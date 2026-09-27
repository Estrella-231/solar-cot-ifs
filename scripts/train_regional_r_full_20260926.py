"""Exploratory full-cache regional COT retrieval; train/val only."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_model(path):
    spec = importlib.util.spec_from_file_location('regional_r_arch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.COTUNet


class PairDataset(Dataset):
    def __init__(self, root, shards, norm, split):
        self.root = root
        self.mean = np.asarray(norm['mean'], np.float32)[:, None, None]
        self.std = np.asarray(norm['std'], np.float32)[:, None, None]
        self.shards = shards
        self.rows = []
        self.cache = OrderedDict()
        for si, shard in enumerate(shards):
            records = json.loads((root / shard['directory'] / 'records.json').read_text())
            for rec in records:
                if rec['split'] == split and rec['included_for_R']:
                    self.rows.append((si, int(rec['array_row'])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        si, row = self.rows[idx]
        if si not in self.cache:
            d = self.root / self.shards[si]['directory']
            self.cache[si] = {
                'x': np.load(d / 'x_raw.npy', mmap_mode='r'),
                'y': np.load(d / 'cot.npy', mmap_mode='r'),
                'm': np.load(d / 'mask.npy', mmap_mode='r')}
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(si)
        a = self.cache[si]
        x = np.asarray(a['x'][row], np.float32)
        x = (x - self.mean) / self.std
        np.nan_to_num(x, copy=False, nan=0., posinf=0., neginf=0.)
        y = np.log1p(np.asarray(a['y'][row], np.float32))[None]
        m = np.asarray(a['m'][row], np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def masked_huber(pred, target, mask):
    v = F.huber_loss(pred.float(), target.float(), reduction='none', delta=.1)
    return (v * mask).sum(), mask.sum()


@torch.no_grad()
def evaluate(model, loader, device, save=False):
    model.eval()
    total = pixels = 0.
    predictions, targets, masks = [], [], []
    for x, y, m in loader:
        x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
        pred = model(x).float()
        if not torch.isfinite(pred).all():
            raise RuntimeError('nonfinite validation prediction')
        s, n = masked_huber(pred, y, m)
        total += float(s)
        pixels += float(n)
        if save:
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(m.cpu().numpy().astype(bool))
    if save:
        return total / pixels, tuple(np.concatenate(v) for v in (predictions, targets, masks))
    return total / pixels, None


def metric_triplet(pred_log, true_log, mask):
    pred, truth = np.expm1(pred_log), np.expm1(true_log)
    e = (pred - truth)[mask]
    return {'n_pixels': int(e.size), 'cot_mae': float(np.mean(np.abs(e))),
            'cot_rmse': float(np.sqrt(np.mean(e ** 2))), 'cot_bias': float(np.mean(e)),
            'predicted_above100_fraction': float(np.mean(pred[mask] > 100))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--r-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('refuse to overwrite output')
    if not torch.cuda.is_available():
        raise RuntimeError('PBS GPU is required')
    idx = json.loads((a.cache / 'index.json').read_text())
    dist = json.loads((a.cache / 'distribution_audit.json').read_text())
    norm_path = a.cache / 'norm.json'
    norm = json.loads(norm_path.read_text())
    assert idx['state'] == 'COMPLETE_EXPLORATORY_REAL_AGRI_CPP_CACHE' and idx['test_used'] is False
    assert dist['state'] == 'COMPLETE_PAIRED_CACHE_DISTRIBUTION' and dist['test_used'] is False
    assert norm['fit_scope'] == 'train-only paired daytime supervised pixels; no val/test'
    assert sha(norm_path) == idx['normalization_sha256']
    for shard in idx['shards']:
        for name, digest in shard['hashes'].items():
            if sha(a.cache / shard['directory'] / name) != digest:
                raise RuntimeError('paired cache SHA mismatch: ' + shard['directory'] + '/' + name)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    train = PairDataset(a.cache, idx['shards'], norm, 'train')
    val = PairDataset(a.cache, idx['shards'], norm, 'val')
    assert len(train) == 6293 and len(val) == 3488, (len(train), len(val))
    train_loader = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                              pin_memory=True, persistent_workers=bool(a.workers),
                              prefetch_factor=2 if a.workers else None, drop_last=False)
    val_loader = DataLoader(val, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                            pin_memory=True, persistent_workers=bool(a.workers),
                            prefetch_factor=2 if a.workers else None)
    model_cls = load_model(a.r_code)
    model = model_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    a.output.mkdir(parents=True)
    metadata = {'state': 'EXPLORATORY_TRAINING', 'purpose': 'regional R feasibility and CPP-reference validation only',
                'test_used': False, 'source_CPP_physical_QA': 'unresolved; not independent truth',
                'train_frames': len(train), 'validation_frames': len(val), 'batch': a.batch,
                'epochs_max': a.epochs, 'patience': a.patience, 'seed': a.seed,
                'loss': 'masked equal-pixel Huber delta=0.1 on log1p(COT)', 'optimizer': 'AdamW lr=1e-3 wd=1e-4',
                'normalization_sha256': sha(norm_path), 'r_arch_sha256': sha(a.r_code),
                'cache_state': idx['state'], 'cache_shards': len(idx['shards']),
                'gpu': torch.cuda.get_device_name(0), 'pbs_job_id': os.environ.get('PBS_JOBID'),
                'cache_path': str(a.cache)}
    atomic(a.output / 'run_metadata.json', metadata)
    best, best_epoch, stale = float('inf'), -1, 0
    start = time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        sum_loss = sum_pixels = 0.
        for x, y, m in train_loader:
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            s, n = masked_huber(pred, y, m)
            loss = s / n.clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            sum_loss += float(s.detach())
            sum_pixels += float(n)
        val_loss, _ = evaluate(model, val_loader, device)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            tmp = a.output / 'best.pt.tmp'
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_huber': best, 'metadata': metadata}, tmp)
            tmp.replace(a.output / 'best.pt')
        else:
            stale += 1
        rec = {'epoch': epoch, 'train_huber': sum_loss / sum_pixels, 'val_huber': val_loss,
               'best_epoch': best_epoch, 'stale_epochs': stale,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'elapsed_seconds': time.monotonic() - start}
        with (a.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        atomic(a.output / 'status.json', {'state': 'TRAINING', **rec})
        print('EPOCH', json.dumps(rec), flush=True)
        if stale >= a.patience:
            break
    ckpt = torch.load(a.output / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    val_loss, arrays = evaluate(model, val_loader, device, save=True)
    pred, true, mask = arrays
    region = metric_triplet(pred, true, mask)
    spatial = {}
    for name, (r, c) in {'sili': (163, 148), 'zhujia': (73, 150)}.items():
        for width in (5, 64):
            h = width // 2
            r0, c0 = max(0, r-h), max(0, c-h)
            sl = (slice(None), slice(None), slice(r0, min(256, r0+width)),
                  slice(c0, min(256, c0+width)))
            spatial[f'{name}_{width}x{width}'] = metric_triplet(pred[sl], true[sl], mask[sl])
    np.savez_compressed(a.output / 'validation_predictions.npz', pred_log1p_cot=pred,
                        reference_log1p_cot=true, valid_mask=mask)
    report = {'state': 'COMPLETE_EXPLORATORY_REGIONAL_R', 'test_used': False,
              'best_epoch': best_epoch, 'best_validation_huber_log1p': val_loss,
              'regional_validation': region, 'station_neighborhoods': spatial,
              'note': 'CPP is a retrieval reference; source producer and independent physical QA remain unresolved; no GHI skill claim',
              'elapsed_seconds': time.monotonic() - start}
    atomic(a.output / 'validation_metrics.json', report)
    atomic(a.output / 'status.json', report)
    print('REGIONAL_R_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
