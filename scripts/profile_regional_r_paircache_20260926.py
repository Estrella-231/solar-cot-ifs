"""Profile full-grid COTUNet batch throughput on the complete paired cache."""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_net(path):
    spec = importlib.util.spec_from_file_location('regional_cotunet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COTUNet


class RegionalDataset(Dataset):
    def __init__(self, root, shards, mean, std, split):
        self.root, self.mean, self.std = root, mean[:, None, None], std[:, None, None]
        self.rows = []
        self.cache = OrderedDict()
        for shard_id, shard in enumerate(shards):
            records = json.loads((root / shard['directory'] / 'records.json').read_text())
            for row in records:
                if row['split'] == split and row['included_for_R']:
                    # Arrays contain only usable frames; records also include
                    # excluded timestamps, so the record position is not an
                    # array index. The packer records the compacted row here.
                    self.rows.append((shard_id, int(row['array_row'])))
        if not self.rows:
            raise RuntimeError('no usable ' + split + ' frames')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        shard_id, row_id = self.rows[index]
        if shard_id not in self.cache:
            item = self.cache.setdefault(shard_id, {})
            directory = self.root / self.index_shards[shard_id]['directory']
            item['x'] = np.load(directory / 'x_raw.npy', mmap_mode='r')
            item['cot'] = np.load(directory / 'cot.npy', mmap_mode='r')
            item['mask'] = np.load(directory / 'mask.npy', mmap_mode='r')
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(shard_id)
        arrays = self.cache[shard_id]
        xraw = np.asarray(arrays['x'][row_id], dtype=np.float32)
        x = (xraw - self.mean) / self.std
        np.nan_to_num(x, copy=False, nan=0., posinf=0., neginf=0.)
        cot = np.log1p(np.asarray(arrays['cot'][row_id], dtype=np.float32))[None]
        mask = np.asarray(arrays['mask'][row_id], dtype=np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(cot), torch.from_numpy(mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--r-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batches', type=lambda s: tuple(int(v) for v in s.split(',')),
                   default=(2, 4, 8, 16, 32, 64, 128))
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('refuse to overwrite profile output')
    if not torch.cuda.is_available():
        raise RuntimeError('one PBS GPU is required')
    status = json.loads((a.cache / 'index.json').read_text())
    distribution = json.loads((a.cache / 'distribution_audit.json').read_text())
    normfile = a.cache / 'norm.json'
    norm = json.loads(normfile.read_text())
    if status['state'] != 'COMPLETE_EXPLORATORY_REAL_AGRI_CPP_CACHE' or status['test_used'] is not False:
        raise RuntimeError('paired cache is incomplete or touched test')
    if distribution['state'] != 'COMPLETE_PAIRED_CACHE_DISTRIBUTION' or distribution['test_used'] is not False:
        raise RuntimeError('paired distribution audit is incomplete')
    if norm['fit_scope'] != 'train-only paired daytime supervised pixels; no val/test':
        raise RuntimeError('normalization scope mismatch')
    if sha(normfile) != status['normalization_sha256']:
        raise RuntimeError('normalization hash mismatch')
    for shard in status['shards']:
        for name, digest in shard['hashes'].items():
            if sha(a.cache / shard['directory'] / name) != digest:
                raise RuntimeError('paired cache hash mismatch: ' + shard['directory'] + '/' + name)
    shards = status['shards']
    train = RegionalDataset(a.cache, shards, np.asarray(norm['mean'], np.float32),
                            np.asarray(norm['std'], np.float32), 'train')
    train.index_shards = shards
    validation = RegionalDataset(a.cache, shards, np.asarray(norm['mean'], np.float32),
                                 np.asarray(norm['std'], np.float32), 'val')
    validation.index_shards = shards
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    model_cls = load_net(a.r_code)
    device_total = torch.cuda.get_device_properties(0).total_memory
    profiles = []
    for batch in a.batches:
        loader = None
        loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=a.workers,
                            pin_memory=True, persistent_workers=bool(a.workers),
                            prefetch_factor=2 if a.workers else None)
        iterator = iter(loader)
        try:
            torch.manual_seed(42)
            model = model_cls().cuda().train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats()
            elapsed = 0.
            seen = 0
            steps = 0
            for step in range(6):
                step_started = time.monotonic()
                try:
                    x, target, mask = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    x, target, mask = next(iterator)
                x, target, mask = x.cuda(non_blocking=True), target.cuda(non_blocking=True), mask.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(x)
                values = torch.nn.functional.huber_loss(pred.float(), target, reduction='none', delta=.1)
                loss = (values * mask).sum() / mask.sum().clamp_min(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                if step >= 2:
                    elapsed += time.monotonic() - step_started
                    seen += batch
                steps += 1
            peak = torch.cuda.max_memory_allocated()
            profiles.append({'batch': batch, 'measured_steps': 4, 'sequences_per_second': seen / elapsed,
                             'peak_allocated_gib': peak / 1024**3,
                             'gpu_memory_fraction': peak / device_total})
            del iterator, model, opt, x, target, mask, pred, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            profiles.append({'batch': batch, 'oom': True})
            torch.cuda.empty_cache()
            break
        finally:
            if loader is not None:
                del loader
    feasible = [r for r in profiles if not r.get('oom') and r['gpu_memory_fraction'] < .75]
    if not feasible:
        raise RuntimeError('no profile batch retained 25% GPU memory headroom')
    selected = max(feasible, key=lambda r: r['sequences_per_second'])
    a.output.mkdir(parents=True)
    report = {'state': 'COMPLETE_SINGLE_GPU_R_PROFILE', 'test_used': False,
              'paired_cache_state': status['state'], 'cache_shards': len(shards),
              'train_frames': len(train), 'validation_frames': len(validation),
              'gpu': torch.cuda.get_device_name(0), 'gpu_total_memory_bytes': device_total,
              'cache_norm_sha256': sha(normfile), 'r_arch_sha256': sha(a.r_code),
              'selected_batch': selected['batch'], 'selection_rule': 'highest measured throughput below 75% allocated GPU memory',
              'profile': profiles,
              'requested_batches': list(a.batches),
              'scope': 'short resource profile only; no full training, checkpoint ranking, or COT skill claim'}
    (a.output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'state': report['state'], 'selected_batch': selected['batch'],
                      'gpu': report['gpu'], 'profile': profiles}), flush=True)


if __name__ == '__main__':
    main()
