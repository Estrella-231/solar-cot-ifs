#!/usr/bin/env python3
"""Extract four preselected validation frames and exact scan-time sidecars.

The sample keys are intentionally fixed and must remain validation-only. The
script exports actual AGRI C01-C03, validity, 3x3 frozen-R COT, station solar
geometry, and NOMObsTime-derived per-pixel scan offsets for physical projection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CASES = [
    ("sili", "2024-04-19 04:15:00", "AC_better", 3),
    ("sili", "2024-06-11 06:30:00", "AC_worse", 0),
    ("zhujia", "2024-05-29 03:45:00", "AC_better", 3),
    ("zhujia", "2024-04-20 06:30:00", "AC_worse", 0),
]
RGB_CHANNELS = (2, 1, 0)  # C03, C02, C01, respectively


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--head-pack", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    with a.candidates.open(encoding="utf-8", newline="") as f:
        candidates = list(csv.DictReader(f))
    candidate_by_key = {(r["station"].lower(), r["target_time_utc"]): r for r in candidates}
    with (a.head_pack / "rows.csv").open(encoding="utf-8", newline="") as f:
        pack_rows = list(csv.DictReader(f))
    pack_by_key = {(r["station"].lower(), r["target_time_utc"]): r for r in pack_rows}
    cot = np.load(a.head_pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)

    selected = []
    rgb, valid, offsets, cossoz, cosraa, daymask, cot3 = [], [], [], [], [], [], []
    aux_sources = []
    for station, stamp, expected_group, expected_seed_count in CASES:
        key = (station, stamp)
        if key not in candidate_by_key or key not in pack_by_key:
            raise KeyError(f"missing validation evidence for {key}")
        c, row = candidate_by_key[key], pack_by_key[key]
        if row["split"] != "val" or c["outcome_group"] != expected_group:
            raise ValueError(f"unexpected split/outcome for {key}: {row['split']} / {c['outcome_group']}")
        if int(c["helpful_seed_count"]) != expected_seed_count:
            raise ValueError(f"seed outcome is not unanimous for {key}")
        expected_stem = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
        if Path(row["agri_frame_relpath"]).stem != expected_stem:
            # Keep the explicit per-row source mapping authoritative; only require
            # canonical date/time consistency below, since directory components
            # are part of the source contract.
            raise ValueError(f"candidate/AGRI timestamp mismatch for {key}: {row['agri_frame_relpath']}")
        rel = Path(row["agri_frame_relpath"])
        raw_path = a.data_root / "data" / rel
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
            raise ValueError(f"unexpected raw AGRI shape/dtype {raw_path}: {raw.shape}/{raw.dtype}")

        bjt = datetime.strptime(rel.stem, "%Y%m%d%H%M") + timedelta(hours=8)
        aux_matches = sorted((a.data_root / "aux_256" / bjt.strftime("%Y%m")).glob(bjt.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
        if len(aux_matches) != 1:
            raise FileNotFoundError(f"expected one exact scan sidecar for {key}, found {len(aux_matches)}")
        with np.load(aux_matches[0], allow_pickle=False) as z:
            bits = np.asarray(z["agri_valid_bits"])
            rgb_mask = np.stack([(bits & (1 << ch)) != 0 for ch in RGB_CHANNELS])
            rgb.append(np.asarray(raw[np.asarray(RGB_CHANNELS)], dtype=np.float32))
            valid.append(rgb_mask)
            offsets.append(np.asarray(z["obs_time_offset_s"], dtype=np.float32))
            cossoz.append(np.asarray(z["cosSOZ"], dtype=np.float32))
            cosraa.append(np.asarray(z["cosRAA"], dtype=np.float32))
            daymask.append(np.asarray(z["day_mask"], dtype=np.float32))
            aux_sources.append({
                "path": str(aux_matches[0]),
                "source_hdf": str(z["source_hdf"].item()),
                "source_size_bytes": int(z["source_size_bytes"].item()),
                "offset_min_s": float(z["obs_time_offset_min_s"].item()),
                "offset_max_s": float(z["obs_time_offset_max_s"].item()),
                "station_offset_s": float(z["obs_time_offset_s"][int(row["row"]), int(row["col"])]),
            })
        bi = int(row["bank_index"])
        cot3.append(np.expm1(np.asarray(cot[bi, 0], dtype=np.float32)))

        selected.append({
            "station": station,
            "target_time_utc": stamp,
            "split": row["split"],
            "source_frame": row["agri_frame_relpath"],
            "row": int(row["row"]),
            "col": int(row["col"]),
            "ghi_wm2": float(row["ghi"]),
            "clear_sky_ghi_wm2": float(row["clear"]),
            "center_r_cot": float(c["center_cot"]),
            "ac_minus_a_delta_sse_mean": float(c["delta_sse_mean_wm4"]),
            "outcome_group": c["outcome_group"],
            "helpful_seed_count": int(c["helpful_seed_count"]),
            "a_seed_mean_wm2": float(c["A_seed_mean_wm2"]),
            "ac_seed_mean_wm2": float(c["AC_seed_mean_wm2"]),
            "r_checkpoint_sha256": "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001",
            "source_hdf": aux_sources[-1]["source_hdf"],
            "scan_offset_s_from_source_start_at_station": aux_sources[-1]["station_offset_s"],
        })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        case_id=np.asarray([f"{r['station']}_{r['target_time_utc'].replace(':', '').replace(' ', 'T').replace('-', '')}" for r in selected]),
        agri_rgb_c03_c02_c01=np.stack(rgb),
        rgb_valid=np.stack(valid).astype(np.uint8),
        obs_time_offset_s=np.stack(offsets),
        cosSOZ=np.stack(cossoz),
        cosRAA=np.stack(cosraa),
        day_mask=np.stack(daymask),
        r_cot_3x3=np.stack(cot3),
    )
    receipt = {
        "state": "FOUR_VALIDATION_CASE_PAYLOADS_EXTRACTED",
        "split": "val only; test payloads were not opened",
        "case_count": len(selected),
        "case_keys": selected,
        "auxiliary_sources": aux_sources,
        "array_file": str(a.out),
        "array_sha256": sha256(a.out),
        "R_pack_rows_sha256": sha256(a.head_pack / "rows.csv"),
        "candidate_csv_sha256": sha256(a.candidates),
        "grid_static_sha256": "54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62",
    }
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": receipt["state"], "cases": selected, "array_sha256": receipt["array_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
