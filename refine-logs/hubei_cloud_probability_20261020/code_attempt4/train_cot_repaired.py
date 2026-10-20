"""Train the preregistered Hunan R on the verified CPP pack; never load test payloads."""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b),
                         nn.SiLU(), nn.Conv2d(b, b, 3, padding=1, bias=False),
                         nn.BatchNorm2d(b), nn.SiLU())


class COTUNet(nn.Module):
    """16 inputs, base16, two downsamplings, nonnegative log1p(COT)."""
    def __init__(self):
        super().__init__()
        self.enc1 = block(16, 16)
        self.enc2 = block(16, 32)
        self.middle = block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward_features(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat((self.up2(c), b), 1))
        return self.dec1(torch.cat((self.up1(d), a), 1))

    def forward(self, x):
        return F.softplus(self.head(self.forward_features(x)))


def loss_fn(pred, target, mask):
    # Equal valid-pixel weights, including genuinely valid zero COT.
    values = F.huber_loss(pred.float(), target.float(), reduction='none', delta=0.1)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


def update(model, opt, scaler, x, y, m):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = loss_fn(model(x), y, m)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    scaler.step(opt)
    scaler.update()
    return loss.detach()


@torch.no_grad()
def evaluate(model, x, y, m, indices, batch, save=False):
    model.eval()
    total = torch.zeros((), device=x.device)
    pixels = torch.zeros((), device=x.device)
    outputs = []
    for ids in indices.split(batch):
        with torch.autocast('cuda', dtype=torch.float16):
            pred = model(x[ids]).float()
        if not torch.isfinite(pred).all():
            raise RuntimeError('nonfinite validation predictions')
        count = m[ids].sum()
        total += loss_fn(pred, y[ids], m[ids]) * count
        pixels += count
        if save:
            outputs.append(pred.cpu().numpy())
    return float(total / pixels), (np.concatenate(outputs) if save else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (a.output / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.output / 'run_metadata.json').exists():
        raise RuntimeError('output already used; do not silently restart/overwrite')
    status = json.loads((a.pack / 'pack_status.json').read_text())
    assert status['state'] == 'COMPLETE' and status['test_payloads_read'] == 0
    hashes = json.loads((a.pack / 'SHA256.json').read_text())
    for name, digest in hashes.items():
        assert sha(a.pack / name) == digest, ('pack SHA mismatch', name)
    with (a.pack / 'rows.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [int(r['index']) for r in rows] == list(range(len(rows)))
    assert all(r['split'] in ('train', 'validation') for r in rows)
    norm = json.loads((a.pack / 'norm.json').read_text())
    assert norm['train_samples'] == 11978 and norm['test_used'] is False
    assert norm['cot_scale'] == 100 and len(norm['mean']) == 16
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_all(a.seed)
    raw = np.load(a.pack / 'x_raw.npy')
    raw = (raw - np.asarray(norm['mean'], np.float32)[None, :, None, None]) / np.asarray(norm['std'], np.float32)[None, :, None, None]
    nonfinite = int((~np.isfinite(raw)).sum())
    raw[~np.isfinite(raw)] = 0  # train-mean imputation after normalization
    x = torch.from_numpy(raw).to(device)
    del raw
    target = np.load(a.pack / 'target.npy') * 100.0
    mask = np.load(a.pack / 'mask.npy')
    assert target.shape == mask.shape == (18514, 1, 16, 16)
    assert mask.dtype == np.bool_ and mask.reshape(len(mask), -1).any(1).all()
    assert np.isfinite(target).all() and ((target >= 0) & (target <= 100)).all()
    y = torch.from_numpy(np.log1p(target)).to(device)
    m = torch.from_numpy(mask).to(device)
    train_ids = torch.tensor([i for i, r in enumerate(rows) if r['split'] == 'train'], device=device)
    val_ids = torch.tensor([i for i, r in enumerate(rows) if r['split'] == 'validation'], device=device)
    assert len(train_ids) == 11978 and len(val_ids) == 6536
    meta = dict(model='hunan_cot_unet16_base16_log1p_cpp_repaired_v1', seed=a.seed,
                input_features=norm['feature_names'], target='nonnegative log1p(COT)',
                loss='valid-pixel masked Huber delta=0.1', lr=1e-3, weight_decay=1e-4,
                epochs_max=50, patience=8, train=11978, validation=6536, test_used=False,
                selection='lowest validation valid-pixel Huber', norm=norm,
                source_code_sha256=sha(__file__), pack_sha256=hashes,
                pack=str(a.pack), output=str(a.output), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), pbs_job_id=os.environ.get('PBS_JOBID'),
                nonfinite_input_imputation='normalized zero (train mean)',
                nonfinite_input_values=nonfinite, reference='CPP retrieval, not independent truth',
                source_NC_checkpoint_binding='unverified', mixed_precision='fp16',
                gradient_clip=5.0, clip_output=False)
    atomic(a.output / 'run_metadata.json', meta)
    # Profile only TRAIN; reset model and optimizer before each independent stage.
    profile = []
    for batch in (64, 128, 256, 512, 1024):
        seed_all(a.seed)
        model = COTUNet().to(device).train()
        opt = optimizer(model)
        scaler = torch.cuda.amp.GradScaler()
        ids = train_ids[:batch]
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(5):
                update(model, opt, scaler, x[ids], y[ids], m[ids])
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(20):
                update(model, opt, scaler, x[ids], y[ids], m[ids])
            torch.cuda.synchronize()
            profile.append(dict(batch=batch, samples_per_second=batch * 20 / (time.monotonic() - start),
                                peak_GiB=torch.cuda.max_memory_allocated() / 2**30))
        except torch.cuda.OutOfMemoryError:
            profile.append(dict(batch=batch, error='OOM'))
        del model, opt, scaler
        torch.cuda.empty_cache()
    valid = [r for r in profile if 'error' not in r]
    if not valid:
        raise RuntimeError('no profile batch fits')
    best_speed = max(r['samples_per_second'] for r in valid)
    batch = min(r['batch'] for r in valid if r['samples_per_second'] >= 0.95 * best_speed)
    atomic(a.output / 'profile.json', dict(candidates=profile, selected_batch=batch,
           rule='smallest batch within 5% of best throughput; cap1024 preserves at least12 updates/epoch',
           cache='all train/validation input/target/mask resident on one GPU'))
    print('PROFILE', json.dumps(profile), 'SELECTED', batch, flush=True)
    seed_all(a.seed)
    model = COTUNet().to(device)
    opt = optimizer(model)
    scaler = torch.cuda.amp.GradScaler()
    ids = train_ids[torch.randperm(len(train_ids), device=device)[:8]]
    initial, _ = evaluate(model, x, y, m, ids, 8)
    for step in range(400):
        model.train()
        loss = update(model, opt, scaler, x[ids], y[ids], m[ids])
    final, _ = evaluate(model, x, y, m, ids, 8)
    gate = dict(initial_eval_huber=initial, final_eval_huber=final, updates=400,
                indices=ids.cpu().tolist(), threshold='final <= 0.5 * initial', passed=final <= initial * 0.5)
    atomic(a.output / 'smoke_overfit.json', gate)
    print('OVERFIT', json.dumps(gate), flush=True)
    if not gate['passed']:
        raise RuntimeError('8-sample overfit failed; formal training blocked')
    del model, opt, scaler
    torch.cuda.empty_cache()
    seed_all(a.seed)  # no profile or overfit weights carried into formal training
    model = COTUNet().to(device)
    opt = optimizer(model)
    scaler = torch.cuda.amp.GradScaler()
    meta.update(batch=batch, smoke_passed=True, formal_reinitialized=True,
                parameters=sum(p.numel() for p in model.parameters()))
    atomic(a.output / 'run_metadata.json', meta)
    best, best_epoch, stale = float('inf'), -1, 0
    started = time.monotonic()
    for epoch in range(50):
        model.train()
        ids = train_ids[torch.randperm(len(train_ids), device=device)]
        train_sum, count = 0.0, 0
        for current in ids.split(batch):
            loss = update(model, opt, scaler, x[current], y[current], m[current])
            pixels = int(m[current].sum())
            train_sum += float(loss) * pixels
            count += pixels
        val_loss, _ = evaluate(model, x, y, m, val_ids, batch)
        improved = val_loss < best
        if improved:
            best, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
        ckpt = dict(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                    epoch=epoch, best_val_loss=best, best_epoch=best_epoch, metadata=meta,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        for name in (['last.pt', 'best.pt'] if improved else ['last.pt']):
            temp = a.output / (name + '.tmp')
            torch.save(ckpt, temp)
            temp.replace(a.output / name)
        record = dict(epoch=epoch, train_huber=train_sum / count, validation_huber=val_loss,
                      best_epoch=best_epoch, elapsed_seconds=time.monotonic() - started)
        with (a.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        atomic(a.output / 'status.json', dict(state='TRAINING', **record))
        print('FORMAL', json.dumps(record), flush=True)
        if stale >= 8:
            break
    checkpoint = torch.load(a.output / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    val_loss, predictions = evaluate(model, x, y, m, val_ids, batch, save=True)
    indices = val_ids.cpu().numpy()
    np.savez_compressed(a.output / 'validation_predictions.npz', row_indices=indices,
                        pred_log1p_cot=predictions, reference_cot=target[indices], mask=mask[indices])
    prediction_cot = np.expm1(predictions.astype(np.float64))
    errors = (prediction_cot - target[indices])[mask[indices]]
    assert np.isfinite(errors).all()
    atomic(a.output / 'validation_metrics.json', dict(huber_log1p=val_loss,
           cot_rmse=float(np.sqrt(np.mean(errors**2))), cot_mae=float(np.mean(np.abs(errors))),
           valid_pixels=int(len(errors)), predicted_cot_above100_fraction=float((prediction_cot[mask[indices]] > 100).mean()),
           note='retrieval-reference validation; no forecast AGRI or GHI accuracy claim'))
    atomic(a.output / 'artifact_sha256.json', {n: sha(a.output / n) for n in
           ('best.pt', 'last.pt', 'run_metadata.json', 'validation_predictions.npz', 'validation_metrics.json')})
    atomic(a.output / 'status.json', dict(state='COMPLETE', best_epoch=best_epoch,
           best_validation_huber=best, last_epoch=epoch, test_used=False,
           elapsed_seconds=time.monotonic() - started))
    print('COT_TRAIN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
