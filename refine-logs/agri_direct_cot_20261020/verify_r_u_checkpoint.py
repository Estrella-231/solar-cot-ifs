#!/usr/bin/env python3
"""Verify R_U checkpoint reload/interface on one U sample; no scoring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_r_u import COTUNet


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.run / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((a.run / "run_metadata.json").read_text(encoding="utf-8"))
    norm = json.loads((a.run / "normalization.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE_R_U_FIXED_EPOCH_50" or status.get("last_epoch") != 50:
        raise ValueError("R_U fixed-budget training receipt is incomplete")
    if status.get("test_payloads_read") != 0 or status.get("buffer_payloads_read") != 0:
        raise ValueError("R_U run violates the test/buffer gate")
    if metadata.get("h_rows_used") != 0 or metadata.get("outer_validation_rows_used") != 0:
        raise ValueError("R_U trainer metadata includes held-out rows")
    checkpoint = a.run / "epoch_050.pt"
    if sha(checkpoint) != status.get("checkpoint_sha256"):
        raise ValueError("R_U checkpoint hash does not match completion receipt")
    if sha(a.run / "normalization.json") != status.get("normalization_sha256"):
        raise ValueError("R_U normalization hash does not match receipt")
    if sha(a.run / "run_metadata.json") != status.get("run_metadata_sha256"):
        raise ValueError("R_U metadata hash does not match receipt")
    rows = list(csv.DictReader((a.pack / "rows_u_supervised.csv").open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("empty U-only pack")
    xraw = np.load(a.pack / "x_corrected.npy", mmap_mode="r", allow_pickle=False)
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    x = np.asarray(xraw[0:1], np.float32).copy()
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    x[~np.isfinite(x)] = 0.0
    torch.set_num_threads(4)
    model = COTUNet()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x))
    if pred.shape != (1, 1, 16, 16) or not torch.isfinite(pred).all() or torch.any(pred < 0):
        raise ValueError("reloaded R_U failed finite nonnegative 16x16 interface")
    result = {
        "state": "PASS_R_U_CHECKPOINT_RELOAD_INTERFACE",
        "checkpoint_sha256": sha(checkpoint),
        "normalization_sha256": sha(a.run / "normalization.json"),
        "source_row": {k: rows[0][k] for k in ("utc_key", "bjt", "station", "source_array_index")},
        "output_shape": list(pred.shape),
        "output_min": float(pred.min()),
        "output_max": float(pred.max()),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "performance_score": False,
    }
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
