#!/usr/bin/env python3
"""Train the fixed-budget, fold-1-U-only corrected-geometry R_U model."""
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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def block(a: int, b: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(),
        nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(),
    )


class COTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.middle = block(16, 32), block(32, 64), block(64, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat((self.up2(c), b), 1))
        return F.softplus(self.head(self.dec1(torch.cat((self.up1(d), a), 1))))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_u_pack(pack: Path):
    audit = json.loads((pack / "pack_audit.json").read_text(encoding="utf-8"))
    if audit.get("state") != "PASS_U_ONLY_R_PACK":
        raise ValueError("U-only R pack audit did not pass")
    if audit.get("h_rows_selected") != 0 or audit.get("outer_validation_rows_selected") != 0:
        raise ValueError("R_U pack includes held-out H/outer-validation rows")
    if audit.get("test_payloads_read") != 0 or audit.get("buffer_payloads_read") != 0:
        raise ValueError("R_U pack violates test/buffer payload gate")
    files = ("x_corrected.npy", "cot.npy", "mask.npy", "rows_u_supervised.csv")
    hashes = audit.get("artifacts_sha256", {})
    for name in files:
        if sha(pack / name) != hashes.get(name):
            raise ValueError(f"R_U pack artifact hash mismatch: {name}")
    x = np.load(pack / "x_corrected.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    m = np.load(pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    rows = list(csv.DictReader((pack / "rows_u_supervised.csv").open(newline="", encoding="utf-8")))
    n = len(rows)
    if x.shape != (n, 16, 16, 16) or y.shape != (n, 1, 16, 16) or m.shape != y.shape:
        raise ValueError("U-only pack arrays do not match row keys")
    if any(r["split"] != "U_train" or r["source_split"] != "train" for r in rows):
        raise ValueError("R_U trainer received non-U training row")
    return audit, x, y, m, rows


def fit_norm(x, mask):
    total = np.zeros(16, np.float64)
    second = np.zeros(16, np.float64)
    count = np.zeros(16, np.int64)
    for i in range(len(x)):
        valid = np.asarray(mask[i, 0], bool)
        xx = np.asarray(x[i], np.float32)
        for ch in range(16):
            v = xx[ch][valid & np.isfinite(xx[ch])]
            total[ch] += v.sum(dtype=np.float64)
            second[ch] += np.square(v, dtype=np.float64).sum()
            count[ch] += v.size
    if np.any(count == 0):
        raise ValueError(f"empty U-only normalization channel: {count.tolist()}")
    mean = total / count
    std = np.sqrt(np.maximum(second / count - mean * mean, 1e-8))
    mean[15], std[15] = 0.0, 1.0
    return mean.astype(np.float64), std.astype(np.float64), count


def safe_masked_huber(pred, target, mask):
    # Mask before arithmetic so invalid NaN payloads cannot propagate as 0*NaN.
    pred = torch.where(mask, pred, torch.zeros_like(pred)).float()
    target = torch.where(mask, target, torch.zeros_like(target)).float()
    per = F.huber_loss(pred, target, reduction="none", delta=0.1)
    denom = mask.sum().clamp_min(1)
    return (per * mask).sum() / denom


def profile_one_gpu_step(xt, yt, mt, out: Path, effective_batch: int) -> dict:
    """Measure a disposable U-only batch; its update is never used for training."""
    device = xt.device
    _, total_mem = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_all(42042)
    model = COTUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    actual = min(effective_batch, len(xt))
    if actual != effective_batch:
        raise ValueError("U-only R profile needs one full effective batch")
    durations = []
    for _ in range(2):
        torch.cuda.synchronize(device)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(xt[:actual])
            loss = safe_masked_huber(pred, yt[:actual], mt[:actual])
        if not torch.isfinite(loss):
            raise RuntimeError("U-only R profile loss is nonfinite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        durations.append(time.monotonic() - started)
    result = {
        "state": "PASS_R_U_SINGLE_GPU_PROFILE",
        "device": torch.cuda.get_device_name(device),
        "profile_rows": actual,
        "profile_updates": len(durations),
        "step_seconds": durations,
        "second_step_samples_per_second": actual / max(durations[-1], 1e-9),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_total_bytes": int(total_mem),
        "profile_update_used_for_training": False,
        "test_payloads_read": 0,
    }
    result["peak_allocated_fraction"] = result["peak_allocated_bytes"] / max(total_mem, 1)
    result["peak_reserved_fraction"] = result["peak_reserved_bytes"] / max(total_mem, 1)
    if result["peak_allocated_fraction"] >= 0.85 or result["peak_reserved_fraction"] >= 0.90:
        raise RuntimeError("batch-1024 R_U profile does not leave the planned GPU headroom")
    atomic_json(out / "profile.json", result)
    del model, optimizer, scaler, pred, loss
    torch.cuda.empty_cache()
    return result


def smoke(pack: Path, output: Path) -> None:
    audit, xraw, yraw, mraw, rows = load_u_pack(pack)
    torch.set_num_threads(4)
    n = min(8, len(rows))
    x = np.asarray(xraw[:n], np.float32).copy()
    y = np.asarray(yraw[:n], np.float32)
    m = np.asarray(mraw[:n], bool).copy()
    mean, std, _ = fit_norm(xraw[:n], mraw[:n])
    x = (x - mean.astype(np.float32)[None, :, None, None]) / std.astype(np.float32)[None, :, None, None]
    x[~np.isfinite(x)] = 0
    target = np.log1p(np.where(m, y, 0.0)).astype(np.float32)
    model = COTUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pred = model(torch.from_numpy(x))
    loss = safe_masked_huber(pred, torch.from_numpy(target), torch.from_numpy(m))
    if not torch.isfinite(loss):
        raise RuntimeError("smoke loss is nonfinite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    opt.step()
    model.eval()
    with torch.no_grad():
        pred0 = model(torch.from_numpy(x))
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "smoke.pt"
    torch.save({"model": model.state_dict(), "pack_audit_sha256": sha(pack / "pack_audit.json")}, checkpoint)
    reloaded = COTUNet()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(ck["model"])
    reloaded.eval()
    with torch.no_grad():
        pred1 = reloaded(torch.from_numpy(x))
    if not torch.equal(pred0, pred1):
        raise RuntimeError("CPU checkpoint reload changed smoke prediction")
    result = {
        "state": "PASS_U_ONLY_R_CPU_SMOKE",
        "pack_audit_sha256": sha(pack / "pack_audit.json"),
        "pack_rows": len(rows),
        "smoke_rows": n,
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "pred_min": float(pred0.min()),
        "pred_max": float(pred0.max()),
        "reload_max_abs_diff": float((pred0 - pred1).abs().max()),
        "test_payloads_read": 0,
        "performance_result": False,
    }
    atomic_json(output / "smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--effective-batch", type=int, default=1024)
    p.add_argument("--microbatch", type=int, default=1024)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.smoke_only:
        smoke(a.pack, a.output)
        return
    if a.epochs != 50 or a.effective_batch != 1024:
        raise ValueError("fixed R_U budget is 50 epochs and effective batch 1024")
    if a.microbatch <= 0 or a.microbatch > a.effective_batch:
        raise ValueError("microbatch must be in 1..effective_batch")
    if a.microbatch != a.effective_batch:
        raise ValueError("this BatchNorm architecture requires microbatch==effective_batch; do not alter BN statistics")
    if not torch.cuda.is_available():
        raise RuntimeError("formal R_U run requires CUDA")
    audit, xraw, yraw, mraw, rows = load_u_pack(a.pack)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    mean, std, count = fit_norm(xraw, mraw)
    norm = {"mean": mean.tolist(), "std": std.tolist(), "count": count.tolist(),
            "fit_scope": "U fold-1 training rows only; common CPP+geometry mask",
            "u_frame_manifest_sha256": audit["u_frame_manifest_sha256"],
            "pack_audit_sha256": sha(a.pack / "pack_audit.json")}
    atomic_json(a.output / "normalization.json", norm)

    x = np.asarray(xraw, np.float32).copy()
    x = (x - mean.astype(np.float32)[None, :, None, None]) / std.astype(np.float32)[None, :, None, None]
    x[~np.isfinite(x)] = 0.0
    y = np.log1p(np.where(np.asarray(mraw, bool), np.asarray(yraw, np.float32), 0.0)).astype(np.float32)
    mask = np.asarray(mraw, bool).copy()
    dev = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    xt = torch.from_numpy(x).to(dev)
    yt = torch.from_numpy(y).to(dev)
    mt = torch.from_numpy(mask).to(dev)
    profile = profile_one_gpu_step(xt, yt, mt, a.output, a.effective_batch)
    seed_all(a.seed)
    model = COTUNet().to(dev)
    init_state_sha = init_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    batches_per_epoch = (len(rows) + a.effective_batch - 1) // a.effective_batch
    meta = {
        "experiment": "agri_direct_cot_20261020",
        "model": "COTUNet from train_paired_r.py; fresh random initialization",
        "seed": a.seed,
        "initialization_state_sha256": init_state_sha,
        "split_id": audit["split_id"],
        "u_frame_manifest_sha256": audit["u_frame_manifest_sha256"],
        "pack_audit_sha256": sha(a.pack / "pack_audit.json"),
        "input": "13 AGRI + cosSOZ + cosRAA + day_mask; native 16x16 station patch",
        "geometry": "corrected EOT radians-to-degrees; frozen Hunan grid_static",
        "supervision": "archived CPP COT, finite-safe common mask, log1p Huber delta=0.1",
        "train_rows": len(rows),
        "train_pixels": int(mask.sum()),
        "epochs": 50,
        "effective_batch": a.effective_batch,
        "microbatch": a.microbatch,
        "updates_per_epoch": batches_per_epoch,
        "planned_updates": batches_per_epoch * 50,
        "optimizer": "AdamW lr=1e-3 wd=1e-4",
        "precision": "CUDA FP16 autocast with GradScaler",
        "normalization_sha256": sha(a.output / "normalization.json"),
        "trainer_source_sha256": sha(Path(__file__)),
        "h_rows_used": 0,
        "outer_validation_rows_used": 0,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "gpu": torch.cuda.get_device_name(0),
        "single_gpu_profile_sha256": sha(a.output / "profile.json"),
    }
    atomic_json(a.output / "run_metadata.json", meta)
    metrics = a.output / "metrics.jsonl"
    n = len(rows)
    for epoch in range(1, 51):
        model.train()
        order = torch.randperm(n, device=dev)
        loss_sum = 0.0
        valid_sum = 0
        epoch_start = time.monotonic()
        for begin in range(0, n, a.effective_batch):
            ids = order[begin:begin + a.effective_batch]
            optimizer.zero_grad(set_to_none=True)
            # Split the logical batch into microbatches; normalize by valid-pixel count.
            denom = int(mt[ids].sum())
            if denom <= 0:
                raise RuntimeError("empty valid mask in an effective batch")
            for mbegin in range(0, len(ids), a.microbatch):
                sub = ids[mbegin:mbegin + a.microbatch]
                submask = mt[sub]
                part = safe_masked_huber(model(xt[sub]), yt[sub], submask)
                scale = int(submask.sum()) / denom
                scaler.scale(part * scale).backward()
                loss_sum += float(part.detach()) * int(submask.sum())
                valid_sum += int(submask.sum())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        record = {"epoch": epoch, "train_huber_log1p": loss_sum / max(valid_sum, 1),
                  "train_valid_pixels": valid_sum, "updates": batches_per_epoch,
                  "epoch_seconds": time.monotonic() - epoch_start,
                  "elapsed_seconds": time.monotonic() - started}
        with metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "metadata": meta, "record": record}
        temp = a.output / "last.pt.tmp"
        torch.save(ck, temp)
        temp.replace(a.output / "last.pt")
        if epoch == 50:
            temp = a.output / "epoch_050.pt.tmp"
            torch.save(ck, temp)
            temp.replace(a.output / "epoch_050.pt")
        atomic_json(a.output / "status.json", {"state": "TRAINING", **record,
                    "test_payloads_read": 0, "buffer_payloads_read": 0})
        print("R_U_EPOCH", json.dumps(record, allow_nan=False), flush=True)
    receipt = {"state": "COMPLETE_R_U_FIXED_EPOCH_50", "last_epoch": 50,
               "checkpoint_sha256": sha(a.output / "epoch_050.pt"),
               "normalization_sha256": sha(a.output / "normalization.json"),
               "run_metadata_sha256": sha(a.output / "run_metadata.json"),
               "test_payloads_read": 0, "buffer_payloads_read": 0,
               "elapsed_seconds": time.monotonic() - started}
    atomic_json(a.output / "status.json", receipt)
    print("R_U_COMPLETE", json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
