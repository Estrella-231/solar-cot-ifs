#!/usr/bin/env python3
"""Extract compact observed-AGRI/geometry patches for frozen T16 val cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


VISIBLE_CHANNELS = (0, 1, 2)  # preprocessed AGRI C01, C02, C03
STATIONS = ("sili", "zhujia")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def geometry_path(root: Path, rel: str) -> Path:
    bjt = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") + timedelta(hours=8)
    parent = root / "aux_256" / bjt.strftime("%Y%m")
    matches = sorted(parent.glob(bjt.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one geometry sidecar for {bjt}: {matches}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    if manifest.get("state") != "PARTIAL_T16_FROZEN_ERROR_BLIND_VALIDATION_CASES":
        raise RuntimeError("T16 manifest is not the expected frozen validation manifest")
    selected = manifest["selected_cases"]
    if not selected or any(not x["future_target_times_utc"] for x in selected):
        raise RuntimeError("empty/incomplete selected cases")

    with a.candidate.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = {}
    for r in rows:
        key = (r["station"].lower(), r["target_time_utc"])
        if r["split"] != "val":
            continue
        if key in by_key:
            raise RuntimeError(f"duplicate candidate key {key}")
        by_key[key] = r

    case_ids = []
    rgb = []
    valid_rgb = []
    geometry = []
    frame_audit = []
    for case_no, case in enumerate(selected):
        st = case["station"].lower()
        if st not in STATIONS:
            raise RuntimeError(f"unexpected station {st}")
        rgb_case, valid_case, geo_case = [], [], []
        for lead, target in enumerate(case["future_target_times_utc"], 1):
            row = by_key.get((st, target))
            if row is None:
                raise RuntimeError(f"target is not a frozen val key: {(st, target)}")
            rel = row["agri_frame_relpath"]
            center_r, center_c = map(int, row["station_grid_rc"].split(","))
            if tuple(map(int, row["station_patch_rc"].split(","))) != (8, 8):
                raise RuntimeError(f"station center drift at {(st, target)}")
            r0, c0 = center_r - 8, center_c - 8
            raw_path = a.data_root / "data" / rel
            aux = geometry_path(a.data_root, rel)
            raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
            if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
                raise RuntimeError(f"unexpected AGRI source: {raw_path} {raw.shape}/{raw.dtype}")
            vis = np.asarray(raw[np.asarray(VISIBLE_CHANNELS), r0:r0 + 16, c0:c0 + 16], dtype=np.float32)
            with np.load(aux, allow_pickle=False) as z:
                bits = np.asarray(z["agri_valid_bits"])[r0:r0 + 16, c0:c0 + 16]
                geo = np.stack([np.asarray(z[k])[r0:r0 + 16, c0:c0 + 16] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
            if vis.shape != (3, 16, 16) or geo.shape != (3, 16, 16) or bits.shape != (16, 16):
                raise RuntimeError(f"bad crop shape at {(st, target)}")
            vm = np.stack([((bits & (1 << k)) != 0) for k in VISIBLE_CHANNELS]) & np.isfinite(vis)
            vis = np.where(vm, vis, np.nan).astype(np.float32)
            if not np.isfinite(geo).all():
                raise RuntimeError(f"nonfinite geometry at {(st, target)}")
            rgb_case.append(vis)
            valid_case.append(vm)
            geo_case.append(geo)
            frame_audit.append({
                "case_index": case_no,
                "station": st,
                "lead_minutes": lead * 15,
                "target_time_utc": target,
                "agri_frame_relpath": rel,
                "agri_frame_sha256": sha256(raw_path),
                "geometry_sidecar_relpath": str(aux.relative_to(a.data_root)),
                "geometry_sidecar_sha256": sha256(aux),
                "center_grid_rc": [center_r, center_c],
                "visible_valid_fraction_by_channel": [float(v.mean()) for v in vm],
            })
        if len(rgb_case) != 16:
            raise RuntimeError(f"case {case_no} has {len(rgb_case)} frames")
        case_ids.append(f"{st}_{case['season_BJT']}_{case['case_type']}_{case['init_time_BJT'].replace(':','').replace(' ','T')}")
        rgb.append(np.stack(rgb_case))
        valid_rgb.append(np.stack(valid_case))
        geometry.append(np.stack(geo_case))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output,
                        case_id=np.asarray(case_ids),
                        visible_agri_c01_c02_c03=np.stack(rgb).astype(np.float32),
                        visible_valid=np.stack(valid_rgb).astype(bool),
                        geometry_cossoz_cosraa_daymask=np.stack(geometry).astype(np.float32))
    audit = {
        "state": "PASS_T16_FROZEN_VALIDATION_AGRI_GEOMETRY_EXTRACTION",
        "case_manifest_sha256": sha256(a.manifest),
        "candidate_manifest_sha256": sha256(a.candidate),
        "output_npz_sha256": sha256(a.output),
        "output_npz_bytes": a.output.stat().st_size,
        "arrays": {
            "visible_agri_c01_c02_c03": list(np.stack(rgb).shape),
            "visible_valid": list(np.stack(valid_rgb).shape),
            "geometry_cossoz_cosraa_daymask": list(np.stack(geometry).shape),
        },
        "visible_channel_order": ["C01", "C02", "C03"],
        "rgb_render_order": ["C03 as R", "C02 as G", "C01 as B"],
        "geometry_order": ["cosSOZ", "cosRAA", "day_mask"],
        "frames": frame_audit,
        "test_payload_opened": False,
        "ghi_values_or_predictions_read": False,
    }
    a.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("state", "output_npz_sha256", "output_npz_bytes", "arrays", "test_payload_opened", "ghi_values_or_predictions_read")}, indent=2))


if __name__ == "__main__":
    main()
