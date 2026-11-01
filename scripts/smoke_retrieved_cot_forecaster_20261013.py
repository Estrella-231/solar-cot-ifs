"""One-step shape, mask, gradient and reload smoke for RetrievedCOTForecaster."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from retrieved_cot_forecaster_20261013 import (
    RetrievedCOTForecaster,
    cot_supervision_masked_mse,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    size = args.size
    if size % 8:
        raise ValueError("smoke spatial size must be divisible by 8")

    torch.manual_seed(42)
    model = RetrievedCOTForecaster(width=96).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = torch.rand(1, 8, 1, size, size, device=device)
    history_valid = torch.ones_like(history, dtype=torch.bool)
    history_valid[:, :, :, :2, :2] = False
    history[~history_valid] = float("nan")
    geometry = torch.rand(1, 24, 3, size, size, device=device)
    target = torch.rand(1, 16, 1, size, size, device=device) * 100.0
    target_valid = torch.rand_like(target) > 0.1
    target[~target_valid] = float("nan")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = model(history, history_valid, geometry)
    loss = cot_supervision_masked_mse(prediction, target, target_valid)
    loss.backward()
    grad_sum = sum(
        float(param.grad.detach().abs().sum().item())
        for param in model.parameters() if param.grad is not None
    )
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("nonfinite prediction or loss")
    if not grad_sum > 0:
        raise RuntimeError("model received no gradient")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        expected = model(history, history_valid, geometry)
    checkpoint = Path(args.output).with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    reloaded = RetrievedCOTForecaster(width=96).to(device).eval()
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device,
                                        weights_only=True))
    with torch.no_grad():
        restored = reloaded(history, history_valid, geometry)
    reload_max_abs = float((expected - restored).abs().max().item())
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_memory = None
        device_name = str(device)

    report = {
        "status": "PASS_SYNTHETIC_ENGINEERING_SMOKE",
        "scope": "single synthetic batch; not COT accuracy or validation",
        "torch": torch.__version__,
        "device": str(device),
        "device_name": device_name,
        "input_shape": [1, 8, 1, size, size],
        "geometry_shape": [1, 24, 3, size, size],
        "output_shape": list(prediction.shape),
        "loss": float(loss.detach().item()),
        "gradient_absolute_sum": grad_sum,
        "history_masked_nan_count": int((~history_valid).sum().item()),
        "target_masked_nan_count": int((~target_valid).sum().item()),
        "prediction_finite": bool(torch.isfinite(prediction).all().item()),
        "checkpoint_reload_max_abs": reload_max_abs,
        "elapsed_seconds_forward_backward_update_reload": elapsed,
        "peak_allocated_bytes": peak_memory,
        "source_sha256": sha256(Path(args.source)),
        "checkpoint": str(checkpoint),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
