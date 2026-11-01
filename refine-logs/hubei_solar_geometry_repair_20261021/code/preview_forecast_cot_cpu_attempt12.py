#!/usr/bin/env python3
"""Early, validation-only forecast-COT diagnostic for attempt12.

This produces a separately named CPU/FP32 preview from the already completed
full-train SimVP validation bank and frozen full R checkpoint. It does not
read train labels, test/buffer payloads, or write into the production COT/GHI
directories. The later GPU pipeline remains authoritative for final outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    base = args.base.resolve()
    code = Path(__file__).resolve().parent
    sys.path.insert(0, str(code))
    from train_cot_repaired import COTUNet

    data = base / "prepared_attempt12"
    sbank = base / "S_banks_attempt12/full/val"
    rrun = base / "R_runs_attempt12/full"
    rpack = data / "R_packs/full"
    simvp_meta = json.loads((base / "S_runs_attempt12/full/run_metadata.json").read_text())
    rmeta = json.loads((rrun / "run_metadata.json").read_text())
    rstatus = json.loads((rrun / "status.json").read_text())
    s_pred_path = sbank / "pred_agri_patch.npy"
    geo_path = sbank / "future_geometry_patch.npy"
    keys_path = sbank / "keys.csv"
    checkpoint = rrun / "best.pt"
    norm_path = rpack / "norm.json"
    pred = np.load(s_pred_path, mmap_mode="r", allow_pickle=False)
    geo = np.load(geo_path, mmap_mode="r", allow_pickle=False)
    if pred.shape != (407, 16, 13, 16, 16) or geo.shape != (407, 16, 3, 16, 16):
        raise ValueError(f"Unexpected validation shapes: pred={pred.shape}, geometry={geo.shape}")
    with keys_path.open(newline="", encoding="utf-8-sig") as f:
        keys = list(csv.DictReader(f))
    if len(keys) != 407 or any(row.get("split") != "val" for row in keys):
        raise ValueError("Expected exactly 407 validation-only SimVP keys")
    if (not simvp_meta.get("solar_geometry_corrected")
            or not rmeta.get("normalization", {}).get("solar_geometry_corrected")
            or rstatus.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED"):
        raise ValueError("Attempt12 checkpoint metadata is not solar-geometry-corrected")

    norm = json.loads(norm_path.read_text())
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    if mean.shape != (1, 16, 1, 1) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("Malformed train-fitted R normalization")

    device = torch.device("cpu")
    model = COTUNet().to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    raw = np.concatenate((np.asarray(pred), np.asarray(geo)), axis=2).reshape(-1, 16, 16, 16)
    answer = []
    with torch.inference_mode():
        for start in range(0, len(raw), args.batch):
            x = raw[start:start + args.batch]
            valid = (x[:, 15:16] > 0.5) & (x[:, 13:14] > np.cos(np.deg2rad(80.0)))
            z = (x - mean) / std
            y = torch.expm1(model(torch.from_numpy(np.asarray(z, dtype=np.float32))))
            y = y.masked_fill(~torch.from_numpy(valid), 0.0)
            answer.append(y.numpy())
    cot = np.concatenate(answer).reshape(407, 16, 1, 16, 16).astype(np.float32, copy=False)
    if not np.isfinite(cot).all() or np.any(cot < 0):
        raise ValueError("Preview COT contains non-finite or negative values")

    args.output.mkdir(parents=True, exist_ok=False)
    bank_path = args.output / "validation.npy"
    np.save(bank_path, cot, allow_pickle=False)
    manifest = {
        "state": "VALIDATION_ONLY_CPU_FP32_PREVIEW_NOT_FINAL",
        "device": "CPU", "precision": "float32", "batch_size": args.batch,
        "shape": list(cot.shape), "rows": len(keys),
        "corrected_S_checkpoint_sha256": simvp_meta["best_checkpoint_sha256"],
        "corrected_R_checkpoint_sha256": sha256(checkpoint),
        "R_normalization_sha256": sha256(norm_path),
        "SimVP_key_sha256": sha256(keys_path),
        "predicted_AGRI_sha256": sha256(s_pred_path),
        "future_geometry_sha256": sha256(geo_path),
        "COT_preview_sha256": sha256(bank_path),
        "preprocessing": "same 16-channel train-fitted z-score as corrected R; valid=day_mask>0.5 and cosSOZ>cos(80deg)",
        "inputs": "full-train SimVP predictions on validation windows only; no observed future AGRI or CPP input",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "authority": "preview only; canonical GPU attempt12 output remains authoritative",
    }
    (args.output / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
