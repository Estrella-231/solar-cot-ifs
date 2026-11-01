#!/usr/bin/env python3
"""Read-only train/val witness of the R-bank value and pixel-mask contract."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
COS80 = float(np.cos(np.deg2rad(80.0)))


def aux_for(root: Path, rel: str) -> Path:
    t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") + timedelta(hours=8)
    matches = list((root / "aux_256" / t.strftime("%Y%m")).glob(
        t.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one aux sidecar, found {len(matches)}: {rel}")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.bank / "bank_manifest.json").read_text())
    if manifest.get("state") != "COMPLETE" or manifest.get("test_payload_opened") is not False:
        raise ValueError("COT bank provenance/test gate failed")
    if manifest.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split")
    cot = np.load(a.bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    rows = list(csv.DictReader((a.bank / "frame_keys.csv").open(encoding="utf-8")))
    if cot.shape != (len(rows), 2, 1, 16, 16):
        raise ValueError(f"unexpected COT layout: {cot.shape}")

    # Deterministic quantiles over train frames, plus train/val boundary checks.
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    va = [i for i, r in enumerate(rows) if r["split"] == "val"]
    chosen = sorted({tr[0], tr[len(tr)//4], tr[len(tr)//2], tr[3*len(tr)//4], tr[-1], va[0], va[len(va)//2], va[-1]})
    out = []
    for i in chosen:
        r = rows[i]
        rel = r["agri_frame_relpath"]
        with np.load(aux_for(a.data_root, rel), allow_pickle=False) as z:
            bits = z["agri_valid_bits"]
            for si, station in enumerate(("sili", "zhujia")):
                row, col = map(int, r[f"{station}_grid_rc"].split(","))
                r0, c0 = row - 8, col - 8
                sl = np.s_[r0:r0+16, c0:c0+16]
                agri_ok = np.ones((16,16), dtype=bool)
                for ch in CHANNELS:
                    agri_ok &= (bits[sl] & (1 << ch)) != 0
                geom = np.stack([z[k][sl] for k in ("cosSOZ", "cosRAA", "day_mask")])
                logcot = np.asarray(cot[i, si, 0])
                physical = np.expm1(logcot.astype(np.float64))
                valid = agri_ok & (geom[2] > 0.5) & (geom[0] > COS80) & np.isfinite(physical)
                out.append({
                    "index": i, "split": r["split"], "target_time_utc": r["target_time_utc"],
                    "station": station, "finite_log1p_fraction": float(np.isfinite(logcot).mean()),
                    "finite_cot_fraction": float(np.isfinite(physical).mean()),
                    "r_input_pixel_valid_fraction": float(valid.mean()),
                    "valid_cot_zero_pixels": int((valid & (physical == 0)).sum()),
                    "valid_cot_gt100_pixels": int((valid & (physical > 100)).sum()),
                    "valid_cot_min_max": [float(physical[valid].min()), float(physical[valid].max())] if valid.any() else None,
                    "cot_log1p_once_expm1": True,
                })
    result = {
        "state": "PASS_SAMPLED_TRAIN_VAL_R_AND_MASK_WITNESS",
        "split_id": manifest["split_id"],
        "R_checkpoint_sha256": manifest["R_checkpoint_sha256"],
        "frames_checked": len(chosen), "station_patches_checked": len(out),
        "test_payload_opened": False,
        "mask_rule": "all 13 AGRI channel bits valid AND day_mask>0.5 AND cosSOZ>cos(80deg) AND finite expm1(COT_log1p)",
        "crop": "16x16; zero-based origin is recorded station center (row,col)-8",
        "samples": out,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
