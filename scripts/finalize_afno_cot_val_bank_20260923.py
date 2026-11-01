"""Finalize metadata for a completed AFNO->frozen-R validation bank.

The prior inference completed its full seen-index check and flushed the bank;
only CSV metadata writing failed because the manifest has no init_time_utc.
This utility validates the existing bank and writes the correct manifest join.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triad-root", type=Path, required=True)
    p.add_argument("--hunan-source", type=Path, required=True)
    p.add_argument("--solar-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    a = p.parse_args()
    s = json.loads((a.solar_root / "configs/s_frozen_hunan_seed42.json").read_text())
    r = json.loads((a.solar_root / "configs/r_frozen_repaired_seed42.json").read_text())
    for path, digest in ((Path(s["manifest"]), s["manifest_sha256"]),
                         (Path(s["normalization"]), s["normalization_sha256"]),
                         (Path(s["checkpoint"]), s["checkpoint_sha256"]),
                         (Path(r["checkpoint"]), r["checkpoint_sha256"])):
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("frozen Hunan asset hash mismatch: " + str(path))
    if sha(a.data_root / "grid_static.npz") != s["grid_sha256"]:
        raise RuntimeError("Hunan grid contract mismatch")
    afno_path = a.triad_root / "runs/afno_transformer_hunan_full_e12_b8x8_seed42_v1/best.pt"
    if not afno_path.is_file():
        raise RuntimeError("missing trained Hunan AFNO checkpoint")

    sys.path.insert(0, str(a.hunan_source))
    from hunan_data import HunanDataset
    norm = json.loads(Path(s["normalization"]).read_text())
    dataset = HunanDataset(a.data_root, s["manifest"], norm, "val")
    bank_path = a.bank / "forecast_cot_log1p.npy"
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(dataset), 2, 16, 1, 16, 16)
    if len(dataset) != 4258 or bank.shape != expected_shape or not np.isfinite(bank).all():
        raise RuntimeError(f"AFNO COT bank integrity failure: {bank.shape}, expected {expected_shape}")
    index_path = a.bank / "index.csv"
    with index_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "seq_id", "split", "BJT_start"])
        for i, row in enumerate(dataset.rows):
            writer.writerow([i, row["seq_id"], row["split"], row["BJT_start"]])
    del bank
    contract = {
        "state": "COMPLETE_AFNO_FORECAST_COT_VALIDATION", "samples": len(dataset), "test_used": False,
        "split": "val", "region": "Hunan", "stations": list(s["station_patches"]),
        "shape": list(expected_shape), "layout": "sequence, station, lead, channel, row, col",
        "prediction_source": "Hunan AFNO+Transformer best validation checkpoint",
        "retriever": "frozen Hunan repaired COTUNet; FP32 forward",
        "inputs": "8 observed AGRI frames + deterministic geometry at history and target times",
        "future_AGRI_read": False, "future_CPP_read": False, "GHI_read": False,
        "lead_minutes": s["lead_minutes"], "station_patches": s["station_patches"],
        "afno_checkpoint": str(afno_path), "afno_checkpoint_sha256": sha(afno_path),
        "R_checkpoint": r["checkpoint"], "R_checkpoint_sha256": r["checkpoint_sha256"],
        "manifest": s["manifest"], "manifest_sha256": s["manifest_sha256"],
        "normalization_sha256": s["normalization_sha256"],
        "baseline_forecast_cot_audit_sha256": sha(a.solar_root / "data/forecast_cot_trajectory_trainval_20260918_v1/audit.json"),
        "comparison_scope": "exploratory frozen-head input-swap diagnostic; not a re-trained backbone comparison",
        "completion_note": "AFNO and frozen-R inference passed full seen-index check; first metadata write failed on a nonexistent manifest field; this finalizer rebuilt the index from the same verified validation manifest.",
        "forecast_cot_sha256": sha(bank_path), "index_sha256": sha(index_path),
    }
    temp = a.bank / "complete.tmp"
    temp.write_text(json.dumps(contract, indent=2, sort_keys=True))
    temp.replace(a.bank / "complete.json")
    print(json.dumps({"state": contract["state"], "samples": len(dataset),
                      "bank_sha256": contract["forecast_cot_sha256"],
                      "index_sha256": contract["index_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
