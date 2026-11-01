#!/usr/bin/env python3
"""Two-GPU DDP runner preserving the fixed global-batch OOF training contract."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler


def load_base(source: Path):
    sys.path.insert(0, str(source))
    import train_monthly_simvp_20261002 as base
    return base


def global_batch_order(size: int, batch: int, seed: int) -> list[list[int]]:
    """Match DataLoader(shuffle=True, generator=seed), including its base-seed draw."""
    g = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=g)
    order = torch.randperm(size, generator=g).tolist()
    return [order[i:i + batch] for i in range(0, size, batch)]


class RankBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, global_batch: int, seed: int, rank: int, world: int):
        self.batches = global_batch_order(size, global_batch, seed)
        self.rank = rank
        self.world = world

    def __iter__(self):
        for batch in self.batches:
            n = len(batch)
            lo = (self.rank * n) // self.world
            hi = ((self.rank + 1) * n) // self.world
            part = batch[lo:hi]
            # DDP needs the same number of forwards on all ranks. For a singleton
            # tail, duplicate the sole row on every rank; pooled masked loss makes
            # the averaged gradient exactly the single-row gradient.
            if not part:
                part = [batch[0]]
            yield part

    def __len__(self):
        return len(self.batches)


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2:
        raise RuntimeError(f"this runner is frozen for exactly two GPUs, got WORLD_SIZE={world}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world, torch.device("cuda", local_rank)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch: int, workers: int, seed: int, rank: int, world: int):
    sampler = RankBatchSampler(len(dataset), batch, seed, rank, world)
    worker_generator = torch.Generator().manual_seed(seed + 104729 * rank)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=False,
                      generator=worker_generator)


def pooled_ddp_loss(pred, truth, valid, world: int):
    # SimVP uses a pooled-valid-pixel MSE, not the mean of per-rank means.
    err = (pred.float() - truth.float()).square()
    mask = valid.to(dtype=err.dtype)
    local_num = (err * mask).sum()
    local_den = mask.sum().detach()
    total_den = local_den.clone()
    dist.all_reduce(total_den, op=dist.ReduceOp.SUM)
    if total_den.item() <= 0:
        raise ValueError("DDP batch has no valid target pixels")
    loss = local_num * world / total_den
    total_num = local_num.detach().clone()
    dist.all_reduce(total_num, op=dist.ReduceOp.SUM)
    return loss, total_num / total_den


def sync_bn_model(base, a, device, local_rank: int):
    net = base.model(a)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.to(device)
    return DDP(net, device_ids=[local_rank], output_device=local_rank,
               broadcast_buffers=True, find_unused_parameters=False)


def profile(a, base, rank, local_rank, world, device):
    norm = json.loads((Path(a.manifests) / "hunan_train_norm.json").read_text())
    assert norm["manifest_sha256"] == base.sha(Path(a.manifests) / "hunan_8to16.csv")
    assert norm["train_frames_sha256"] == base.sha(Path(a.manifests) / "train_frames.txt")
    data = base.MonthlyDataset(a, norm, "train")
    seed_all(a.seed)
    net = sync_bn_model(base, a, device, local_rank).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    loader = make_loader(data, 20, a.workers, a.seed, rank, world)
    timings, losses = [], []
    for k, (x, g, y, v, _) in enumerate(loader):
        if k >= 5:
            break
        x, g, y, v = [z.to(device, non_blocking=True) for z in (x, g, y, v)]
        torch.cuda.synchronize(device)
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x, g)
            loss, metric = pooled_ddp_loss(pred, y, v, world)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite DDP profile loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize(device)
        timings.append(time.monotonic() - start)
        losses.append(float(metric))
        if rank == 0:
            print("DDP_PROFILE_STEP", json.dumps({"step": k + 1, "global_batch": 20,
                  "loss": losses[-1], "elapsed_s": timings[-1]}), flush=True)
    dist.barrier()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    peak = torch.tensor([allocated], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {"state": "TWO_GPU_GLOBAL_BATCH20_PROFILE", "world_size": world,
                  "global_batch": 20, "per_rank_batch": 10,
                  "steps_measured": len(timings), "step_seconds": timings,
                  "global_samples_per_second": 20 / float(np.mean(timings[1:] or timings)),
                  "max_rank_peak_allocated_gib": float(peak.item()),
                  "bf16": True, "sync_batchnorm": True,
                  "sample_order_contract": "same global permutation as single-GPU batch20"}
        Path(a.output).mkdir(parents=True, exist_ok=False)
        (Path(a.output) / "ddp_profile.json").write_text(json.dumps(result, indent=2) + "\n")
        print("DDP_PROFILE_COMPLETE", json.dumps(result), flush=True)
    dist.barrier()


def train(a, base, rank, local_rank, world, device):
    norm_path = Path(a.manifests) / "hunan_train_norm.json"
    norm = json.loads(norm_path.read_text())
    assert norm["manifest_sha256"] == base.sha(Path(a.manifests) / "hunan_8to16.csv")
    assert norm["train_frames_sha256"] == base.sha(Path(a.manifests) / "train_frames.txt")
    oof_ids = Path(a.manifests) / "fold_oof_ids.txt"
    assert oof_ids.is_file() and oof_ids.read_text().splitlines()
    split_audit = json.loads((Path(a.manifests) / "manifest_audit.json").read_text())
    assert split_audit["frame_disjoint_splits"] is True
    assert split_audit["train_val_frame_intersection"] == 0
    assert split_audit["sequence_counts"]["test"] == 0
    output = Path(a.output) / "training"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    tr = base.MonthlyDataset(a, norm, "train")
    net = sync_bn_model(base, a, device, local_rank)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    meta = {"region": "Hunan", "initialization": "random", "seed": a.seed,
        "fold": a.fold, "epochs_fixed": 12, "batch": 20, "global_batch": 20,
        "world_size": world, "per_rank_batch": 10,
        "batch_norm": "SyncBatchNorm over the same global batch of 20",
        "sample_order": "single-GPU DataLoader permutation split contiguously across ranks",
        "train_samples": len(tr), "heldout_payload_read": False,
        "test_payloads_read": 0,
        "checkpoint_policy": "use_epoch_12_fixed; no heldout-fold scoring or checkpoint selection",
        "geometry_contract": "history_8_then_target_16_v2", "input_agri_channels": 13,
        "history_frames": 8, "target_frames": 16,
        "training_loss": "pooled-valid-pixel normalized AGRI MSE, globally normalized across ranks",
        "manifest_sha256": base.sha(Path(a.manifests) / "hunan_8to16.csv"),
        "fold_oof_ids_sha256": base.sha(oof_ids),
        "fold_manifest_audit_sha256": base.sha(Path(a.manifests) / "manifest_audit.json"),
        "normalization_sha256": base.sha(norm_path),
        "model_code_sha256": base.sha(Path(a.source) / "triad_models.py"),
        "data_code_sha256": base.sha(Path(a.source) / "hunan_data.py"),
        "simvp_code_sha256": base.sha(Path(a.source) / "simvp.py"),
        "training_code_sha256": base.sha(Path(__file__)),
        "pbs": os.environ.get("PBS_JOBID"), "node_cache": a.cache,
        "old_checkpoint_loaded": False, "future_observed_agri_read": False,
        "cpp_read": False, "test_payload_opened": False, "buffer_payload_opened": False}
    if rank == 0:
        base.write_json(output / "run_metadata.json", meta)
    dist.barrier()
    step = 0
    for epoch in range(12):
        started = time.monotonic()
        net.train()
        batch_losses = []
        loader = make_loader(tr, 20, a.workers, a.seed + epoch, rank, world)
        for k, (x, g, y, v, _) in enumerate(loader):
            x, g, y, v = [z.to(device, non_blocking=True) for z in (x, g, y, v)]
            progress = (epoch + k / len(loader)) / 12
            for group in opt.param_groups:
                group["lr"] = 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x, g)
                loss, metric = pooled_ddp_loss(pred, y, v, world)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch={epoch+1} step={step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            batch_losses.append(float(metric.detach()))
            step += 1
            if rank == 0 and step % 100 == 0:
                print("DDP_TRAIN_STEP", json.dumps({"fold": a.fold, "epoch": epoch + 1,
                      "step": step, "global_batch": 20,
                      "loss": batch_losses[-1]}), flush=True)
        rec = {"fold": a.fold, "epoch": epoch + 1, "step": step,
               "train_loss_mean": float(np.mean(batch_losses)),
               "elapsed_s": time.monotonic() - started,
               "heldout_payload_read": False,
               "checkpoint_selection": "fixed epoch schedule"}
        dist.barrier()
        if rank == 0:
            base.write_json(output / f"epoch_{epoch + 1:03d}.json", rec)
            payload = {"model": net.module.state_dict(), "optimizer": opt.state_dict(),
                       "metadata": meta, "epoch": epoch + 1, "step": step}
            torch.save(payload, output / "last.tmp")
            (output / "last.tmp").replace(output / "last.pt")
            if epoch == 11:
                torch.save(payload, output / "epoch_012.tmp")
                (output / "epoch_012.tmp").replace(output / "epoch_012.pt")
            print("DDP_EPOCH_END", json.dumps(rec), flush=True)
        dist.barrier()
    if rank == 0:
        result = {"state": "COMPLETE_OOF_SIMVP_FIXED_EPOCH_TRAINING", "fold": a.fold,
                  "epochs": 12, "steps": step, "selected_checkpoint": "epoch_012.pt",
                  "checkpoint_sha256": base.sha(output / "epoch_012.pt"),
                  "heldout_payload_read": False, "test_payloads_read": 0,
                  "world_size": world, "global_batch": 20}
        base.write_json(output / "training_complete.json", result)
        print("OOF_SIMVP_DDP_TRAIN_COMPLETE", json.dumps(result), flush=True)
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("profile", "train"), required=True)
    for name in ("root", "source", "cache", "manifests", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--fold", type=int, required=True)
    a = p.parse_args()
    rank, local_rank, world, device = init_dist()
    try:
        base = load_base(Path(a.source))
        if a.mode == "profile":
            profile(a, base, rank, local_rank, world, device)
        else:
            train(a, base, rank, local_rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
