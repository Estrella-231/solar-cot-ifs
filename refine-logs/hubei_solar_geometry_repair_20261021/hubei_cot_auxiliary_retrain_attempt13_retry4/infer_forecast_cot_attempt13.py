#!/usr/bin/env python3
"""Run the frozen Hubei R model on a saved validation SimVP AGRI bank.

This is a validation-only post-training stage. It consumes the physical
16x16 AGRI forecasts and target-time solar-geometry fields written by
code_pipeline_attempt12/infer_S.py, and emits physical COT for the existing
attempt12 evaluator. No future observed AGRI, CPP, test, or buffer payload is
read as model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--simvp-bank", type=Path, required=True,
                   help="infer_S.py output root containing BANK_REPORT.json and val/")
    p.add_argument("--r-source", type=Path, required=True,
                   help="directory containing train_cot_repaired.py")
    p.add_argument("--r-checkpoint", type=Path, required=True)
    p.add_argument("--r-norm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="new .npy output path; existing outputs are never overwritten")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sequence-batch-size", type=int, default=8)
    a = p.parse_args()

    if a.sequence_batch_size < 1:
        raise ValueError("sequence-batch-size must be positive")
    output_meta_path = a.output.with_suffix(a.output.suffix + ".json")
    if a.output.exists() or output_meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    report_path = a.simvp_bank / "BANK_REPORT.json"
    val_dir = a.simvp_bank / "val"
    pred_path = val_dir / "pred_agri_patch.npy"
    geom_path = val_dir / "future_geometry_patch.npy"
    keys_path = val_dir / "keys.csv"
    for path in (report_path, pred_path, geom_path, keys_path,
                 a.r_source / "train_cot_repaired.py", a.r_checkpoint, a.r_norm):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("state") != "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK" or
            report.get("inferred_splits") != ["val"] or
            report.get("sequences_written_by_split", {}).get("val") != 407 or
            report.get("test_payloads_read") != 0 or
            report.get("buffer_payloads_read") != 0):
        raise RuntimeError(f"Unexpected or non-val-only SimVP bank report: {report_path}")

    agri = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    geom = np.load(geom_path, mmap_mode="r", allow_pickle=False)
    expected_agri = (407, 16, 13, 16, 16)
    expected_geom = (407, 16, 3, 16, 16)
    if agri.shape != expected_agri or geom.shape != expected_geom:
        raise ValueError(f"Unexpected input shapes: AGRI={agri.shape}, geometry={geom.shape}")

    rnorm = json.loads(a.r_norm.read_text(encoding="utf-8"))
    mean = np.asarray(rnorm["mean"], dtype=np.float32)
    std = np.asarray(rnorm["std"], dtype=np.float32)
    if mean.shape != (16,) or std.shape != (16,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("R normalization must contain 16 finite train-fitted channels")

    sys.path.insert(0, str(a.r_source))
    from train_cot_repaired import COTUNet

    payload = torch.load(a.r_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("R checkpoint must contain a model state_dict")
    model = COTUNet().to(a.device).float().eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    mu = torch.as_tensor(mean, dtype=torch.float32, device=a.device).view(1, 16, 1, 1)
    sd = torch.as_tensor(std, dtype=torch.float32, device=a.device).view(1, 16, 1, 1)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        a.output, mode="w+", dtype=np.float32, shape=(407, 16, 1, 16, 16)
    )
    with torch.inference_mode():
        for start in range(0, 407, a.sequence_batch_size):
            end = min(start + a.sequence_batch_size, 407)
            # mmap-backed slices are read-only; copy before exposing them to torch.
            x_agri = torch.as_tensor(np.array(agri[start:end], copy=True), device=a.device, dtype=torch.float32)
            x_geom = torch.as_tensor(np.array(geom[start:end], copy=True), device=a.device, dtype=torch.float32)
            x = torch.cat((x_agri, x_geom), dim=2).reshape(-1, 16, 16, 16)
            x = (x - mu) / sd
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            pred_log1p = model(x).float()
            if pred_log1p.shape != (x.shape[0], 1, 16, 16):
                raise RuntimeError(f"Unexpected R output shape {tuple(pred_log1p.shape)}")
            pred_cot = torch.expm1(pred_log1p.clamp(0.0, math.log1p(100.0)))
            block = pred_cot.reshape(end - start, 16, 1, 16, 16).cpu().numpy()
            if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 100.001):
                raise RuntimeError("Physical COT output is non-finite or outside [0,100]")
            out[start:end] = block
            print(json.dumps({"state": "R_INFER_PROGRESS", "windows_completed": end,
                              "windows_total": 407}, separators=(",", ":")), flush=True)
    out.flush()

    metadata = {
        "state": "ATTEMPT13_VALIDATION_FORECAST_PHYSICAL_COT_BANK",
        "scope": "407 validation windows; SimVP predicted AGRI + target-time geometry -> frozen Hubei R",
        "shape": list(out.shape),
        "dtype": "float32 physical COT, clipped through log1p output to [0,100]",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "simvp_bank_report_sha256": sha256(report_path),
        "simvp_agri_sha256": sha256(pred_path),
        "simvp_geometry_sha256": sha256(geom_path),
        "simvp_keys_sha256": sha256(keys_path),
        "r_source_sha256": sha256(a.r_source / "train_cot_repaired.py"),
        "r_checkpoint_sha256": sha256(a.r_checkpoint),
        "r_normalization_sha256": sha256(a.r_norm),
        "cot_bank_sha256": sha256(a.output),
        "model_inputs": "predicted AGRI and known target-time solar geometry only",
        "reference_use": "CPP is used downstream for validation scoring only, never as inference input",
    }
    output_meta_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
