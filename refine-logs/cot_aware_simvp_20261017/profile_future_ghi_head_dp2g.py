#!/usr/bin/env python3
"""Synthetic-only parity and throughput profile for the fixed future-GHI head."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_future_ghi_heads import IntervalHead


def sync():
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1792)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("profile requires exactly two PBS-assigned visible GPUs")
    if args.output.exists():
        raise FileExistsError(args.output)

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(731942)
    b = args.batch_size
    ag = torch.randn(b, 3, 13, 3, 3, device="cuda:0")
    geo = torch.randn(b, 3, 2, 3, 3, device="cuda:0")
    cot = torch.randn(b, 3, 1, 3, 3, device="cuda:0")
    clear = torch.rand(b, device="cuda:0") * 1000.0
    ghi = torch.rand(b, device="cuda:0") * clear

    torch.manual_seed(271828)
    initial = {k: v.detach().clone() for k, v in IntervalHead().state_dict().items()}
    single = IntervalHead().to("cuda:0")
    single.load_state_dict(initial, strict=True)
    parallel_core = IntervalHead().to("cuda:0")
    parallel_core.load_state_dict(initial, strict=True)
    parallel = torch.nn.DataParallel(parallel_core, device_ids=[0, 1], output_device=0)

    single.eval(); parallel.eval()
    with torch.inference_mode():
        y_single = single(ag, geo, cot)
        y_parallel = parallel(ag, geo, cot)
    output_max_abs = float((y_single - y_parallel).abs().max().item())
    if not torch.allclose(y_single, y_parallel, rtol=5e-4, atol=5e-5):
        raise AssertionError("single/two-GPU forward parity failed")

    def train_steps(model, count):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        sync(); started = time.perf_counter()
        for _ in range(count):
            pred = model(ag, geo, cot)
            loss = (((pred * clear - ghi) / 1000.0) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
        sync()
        elapsed = time.perf_counter() - started
        return {"steps": count, "elapsed_seconds": elapsed,
                "steps_per_second": count / elapsed,
                "samples_per_second": count * b / elapsed,
                "last_loss": float(loss.detach().item())}

    # A short warm-up also lets cuDNN finish algorithm selection before timing.
    train_steps(single, args.warmup)
    train_steps(parallel, args.warmup)
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    single_result = train_steps(single, args.steps)
    single_peaks = [torch.cuda.max_memory_allocated(i) for i in range(2)]
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    parallel_result = train_steps(parallel, args.steps)
    parallel_peaks = [torch.cuda.max_memory_allocated(i) for i in range(2)]
    speedup = single_result["samples_per_second"] and (
        parallel_result["samples_per_second"] / single_result["samples_per_second"])
    report = {
        "state": "COMPLETE_SYNTHETIC_FUTURE_GHI_HEAD_PROFILE",
        "profile_kind": "synthetic implementation only; no project samples or labels read",
        "model": "existing IntervalHead; unchanged forward, loss, AdamW and batch size",
        "batch_size": b, "warmup_steps": args.warmup, "measured_steps": args.steps,
        "single_gpu": single_result, "two_gpu_dataparallel": parallel_result,
        "two_gpu_speedup_ratio": speedup,
        "forward_max_abs_difference": output_max_abs,
        "single_gpu_peak_allocated_bytes": single_peaks,
        "two_gpu_peak_allocated_bytes": parallel_peaks,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(2)],
        "project_payloads_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FUTURE_GHI_HEAD_DP2_PROFILE_COMPLETE", json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
