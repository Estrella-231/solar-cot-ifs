#!/usr/bin/env python3
"""CPU tests for global-batch and masked-loss DDP semantics."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_station_patch_cot_forecaster import (
    GlobalBatchSampler,
    distributed_equal_lead_loss,
)


def check_global_batches() -> None:
    size, global_batch, seed = 103, 32, 7
    reference = list(GlobalBatchSampler(size, global_batch, 0, 1, seed, True))
    for world in (2, 4, 8):
        ranks = [list(GlobalBatchSampler(size, global_batch, r, world, seed, True))
                 for r in range(world)]
        if any(len(x) != len(reference) for x in ranks):
            raise AssertionError("rank update counts differ")
        rebuilt = []
        for batch_i in range(len(reference)):
            rebuilt.append([idx for rank in ranks for idx in rank[batch_i]])
        if rebuilt != reference:
            raise AssertionError(f"global sample order changed at world={world}")
    print("GLOBAL_BATCH_ORDER_PASS", flush=True)


def _distributed_gradient_worker(rank: int, world: int, init_file: str,
                                 pred_np, target_np, valid_np, expected_grad_np,
                                 result_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world)
    pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(target_np, dtype=torch.float64)
    valid = torch.tensor(valid_np, dtype=torch.bool)
    per_rank = pred.shape[0] // world
    sl = slice(rank * per_rank, (rank + 1) * per_rank)
    loss = distributed_equal_lead_loss(pred[sl], target[sl], valid[sl], True)
    loss.backward()
    # DDP averages parameter gradients across ranks after backward.
    dist.all_reduce(pred.grad, op=dist.ReduceOp.SUM)
    pred.grad.div_(world)
    if rank == 0:
        expected = torch.tensor(expected_grad_np, dtype=torch.float64)
        torch.save({"loss_local_rank0": float(loss.item()),
                    "gradient": pred.grad.detach().cpu()}, result_file)
        if not torch.allclose(pred.grad, expected, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"DDP gradient differs: max abs={(pred.grad-expected).abs().max()}")
    dist.barrier()
    dist.destroy_process_group()


def check_distributed_masked_loss() -> None:
    torch.manual_seed(123)
    pred = torch.randn(8, 4, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(pred).abs() * 50
    valid = torch.rand_like(pred) > 0.37
    # Exercise the invalid-NaN path used by the actual cache contract.
    target[~valid] = float("nan")
    safe = torch.where(valid, target, torch.zeros_like(target))
    target_log = torch.log1p(safe.clamp_min(0)) / torch.log(torch.tensor(101.0, dtype=torch.float64))
    err = torch.where(valid, pred - target_log, torch.zeros_like(pred))
    counts = valid.sum(dim=(0, 2, 3, 4)).to(torch.float64)
    sums = err.square().sum(dim=(0, 2, 3, 4))
    expected_loss = (sums / counts.clamp_min(1))[counts > 0].mean()
    expected_loss.backward()
    expected_grad = pred.grad.detach().clone()

    with tempfile.TemporaryDirectory(prefix="cot_ddp_semantics_") as tmp:
        init_file = str(Path(tmp) / "dist_init")
        result_file = str(Path(tmp) / "result.pt")
        mp.spawn(_distributed_gradient_worker,
                 args=(2, init_file, pred.detach().numpy(), target.numpy(),
                       valid.numpy(), expected_grad.numpy(), result_file),
                 nprocs=2, join=True)
        got = torch.load(result_file, map_location="cpu", weights_only=False)
        if not torch.isfinite(got["gradient"]).all():
            raise AssertionError("nonfinite DDP gradient")
    print("GLOBAL_MASKED_LEAD_LOSS_GRADIENT_PASS", flush=True)


if __name__ == "__main__":
    check_global_batches()
    check_distributed_masked_loss()
