#!/usr/bin/env python3
"""Train the separate full-train DropPath-0 SimVP selected by val metrics.

The model sees training AGRI only. Validation AGRI is read only as the target
for checkpoint selection; it is never part of model input. Test is unopened.
The candidate forecast-bank builder is a separate inference-only program.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import train_monthly_simvp_oof_ddp_20261020_rankrng as runner
from monthly_validation_selection import validation_score


def load_base(source: Path):
    return runner.load_base(source)


def evaluate_val(module, base, a, norm, device) -> dict:
    """Use validation AGRI only as label for the frozen checkpoint score."""
    va = base.MonthlyDataset(a, norm, "val")
    loader = DataLoader(va, batch_size=20, shuffle=False, num_workers=a.workers,
                        pin_memory=True, persistent_workers=False)
    module.eval()
    month_sse = np.zeros(12, dtype=np.float64)
    month_count = np.zeros(12, dtype=np.float64)
    with torch.inference_mode():
        for x, g, y, valid, months in loader:
            x, g, y, valid = [z.to(device, non_blocking=True) for z in (x, g, y, valid)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = module(x, g)
            err2 = (pred.float() - y.float()).square()
            mask = valid.to(err2.dtype)
            per_sample_sse = (err2 * mask).sum((1, 2, 3, 4)).double().cpu().numpy()
            per_sample_count = mask.sum((1, 2, 3, 4)).double().cpu().numpy()
            for month, sse, count in zip(months.tolist(), per_sample_sse, per_sample_count):
                month_sse[month - 1] += sse
                month_count[month - 1] += count
    equal_season, seasons = validation_score(month_sse, month_count)
    return {"val_equal_season_rmse": equal_season, "season_rmse": seasons,
            "val_pixel_rmse": float(np.sqrt(month_sse.sum() / month_count.sum())),
            "monthly_rmse": {str(i + 1): (float(np.sqrt(month_sse[i] / month_count[i]))
                                           if month_count[i] > 0 else None)
                             for i in range(12)},
            "validation_labels_used_for_checkpoint_selection": True,
            "validation_future_agri_used_as_model_input": False}


def train(a, base, rank: int, local_rank: int, world: int, device: torch.device) -> None:
    if world != 2:
        raise RuntimeError(f"candidate full-train runner is frozen for two GPUs, got {world}")
    manifests = Path(a.manifests)
    norm_path = manifests / "hunan_train_norm.json"
    csv_path = manifests / "hunan_8to16.csv"
    audit_path = manifests / "manifest_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    if audit.get("frame_disjoint_splits") is not True:
        raise ValueError("full-train candidate requires the audited disjoint monthly manifest")
    if norm.get("validation_test_used") is not False or norm.get("manifest_sha256") != base.sha(csv_path):
        raise ValueError("candidate normalization is not train-only or does not bind to manifest")
    if norm.get("train_frames_sha256") != base.sha(manifests / "train_frames.txt"):
        raise ValueError("candidate train-only normalization/frame-list mismatch")

    run = Path(a.output)
    training = run / "training"
    if rank == 0:
        training.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    seed = a.seed
    if seed != 42:
        raise ValueError("the frozen candidate full-train comparison is seed 42")
    runner.seed_all(seed)
    torch.backends.cudnn.benchmark = True
    tr = base.MonthlyDataset(a, norm, "train")
    net = runner.sync_bn_model(base, a, device, local_rank)
    rank_cuda_seed = runner.seed_rank_cuda_rng(seed, rank)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    metadata = {
        "region": "Hunan", "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "initialization": "random", "seed": seed, "epochs_fixed": 12,
        "batch": 20, "global_batch": 20, "world_size": world, "per_rank_batch": 10,
        "drop_path_prob": 0.0,
        "train_samples": len(tr),
        "rank_cuda_rng_seed": rank_cuda_seed, "batch_norm": "SyncBatchNorm over global batch 20",
        "geometry_contract": "history_8_then_target_16_v2", "input_agri_channels": 13,
        "history_frames": 8, "target_frames": 16,
        "training_loss": "pooled-valid-pixel normalized AGRI MSE",
        "checkpoint_selection": "minimum equal-four-season validation pixel RMSE over fixed 12 epochs",
        "validation_labels_used_for_selection": True,
        "validation_future_agri_used_as_model_input": False,
        "test_payloads_read": 0, "test_payload_opened": False,
        "buffer_payload_opened": False,
        "future_observed_agri_inference_input": False, "cpp_read": False,
        "manifest_sha256": base.sha(csv_path), "normalization_sha256": base.sha(norm_path),
        "manifest_audit_sha256": base.sha(audit_path),
        "model_code_sha256": base.sha(Path(a.source) / "triad_models.py"),
        "data_code_sha256": base.sha(Path(a.source) / "hunan_data.py"),
        "simvp_code_sha256": base.sha(Path(a.source) / "simvp.py"),
        "training_code_sha256": base.sha(Path(__file__)),
        "pbs": os.environ.get("PBS_JOBID"), "node_cache": a.cache,
        "old_checkpoint_loaded": False,
    }
    if rank == 0:
        base.write_json(training / "run_metadata.json", metadata)
    dist.barrier()

    best = float("inf")
    step = 0
    for epoch in range(12):
        started = time.monotonic()
        net.train()
        losses = []
        loader = runner.make_loader(tr, 20, a.workers, seed + epoch, rank, world)
        for k, (x, g, y, valid, _) in enumerate(loader):
            x, g, y, valid = [z.to(device, non_blocking=True) for z in (x, g, y, valid)]
            progress = (epoch + k / max(1, len(loader))) / 12
            for group in opt.param_groups:
                group["lr"] = 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x, g)
                loss, metric = runner.pooled_ddp_loss(pred, y, valid, world)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch={epoch + 1}, step={step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            losses.append(float(metric.detach()))
            step += 1
            if rank == 0 and step % 100 == 0:
                print("FULLTRAIN_DROPPATH0_STEP", json.dumps({"epoch": epoch + 1,
                      "step": step, "global_batch": 20, "loss": losses[-1]}), flush=True)
        dist.barrier()
        if rank == 0:
            val = evaluate_val(net.module, base, a, norm, device)
            score = val["val_equal_season_rmse"]
            rec = {"epoch": epoch + 1, "step": step,
                   "train_loss_mean": float(np.mean(losses)),
                   "elapsed_s": time.monotonic() - started, **val}
            base.write_json(training / f"epoch_{epoch + 1:03d}.json", rec)
            payload = {"model": net.module.state_dict(), "optimizer": opt.state_dict(),
                       "metadata": metadata, "epoch": epoch + 1, "step": step,
                       "best_equal_season_val_rmse": min(best, score)}
            torch.save(payload, training / "last.tmp")
            Path(training / "last.tmp").replace(training / "last.pt")
            if score < best:
                best = score
                torch.save(payload, training / "best.tmp")
                Path(training / "best.tmp").replace(training / "best.pt")
            print("FULLTRAIN_DROPPATH0_EPOCH_END", json.dumps(rec), flush=True)
        dist.barrier()
    if rank == 0:
        result = {"state": "COMPLETE_DROPPATH0_FULLTRAIN_VALIDATION_SELECTED",
                  "epochs": 12, "steps": step, "best_equal_season_val_rmse": best,
                  "best_checkpoint_sha256": base.sha(training / "best.pt"),
                  "best_checkpoint": "best.pt", "test_payloads_read": 0,
                  "buffer_payload_opened": False, "future_observed_agri_inference_input": False,
                  "cpp_read": False, "world_size": world, "global_batch": 20}
        base.write_json(training / "training_complete.json", result)
        print("FULLTRAIN_DROPPATH0_COMPLETE", json.dumps(result), flush=True)
    dist.barrier()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("profile", "train"), required=True)
    for name in ("root", "source", "cache", "manifests", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    triad_text = (Path(a.source) / "triad_models.py").read_text(encoding="utf-8")
    if "drop_path_prob=0.0" not in triad_text or "drop_path_prob=.1" in triad_text:
        raise ValueError("source is not the frozen DropPath-0 candidate snapshot")
    rank, local_rank, world, device = runner.init_dist()
    try:
        base = load_base(Path(a.source))
        if a.mode == "profile":
            runner.profile(a, base, rank, local_rank, world, device)
        else:
            train(a, base, rank, local_rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
