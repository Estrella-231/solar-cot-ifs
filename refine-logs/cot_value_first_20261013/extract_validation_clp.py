#!/usr/bin/env python3
"""Extract CPP CLP for exactly the saved monthly-R validation sample keys."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arr(v, key=Ellipsis):
    return np.asarray(np.ma.filled(v[key], np.nan), dtype=np.float32)


def axes(nc):
    latv, lonv = nc.variables["LAT"], nc.variables["LON"]
    lat = arr(latv, (slice(None), 0)) if len(latv.shape) == 2 else arr(latv)
    lon = arr(lonv, (0, slice(None))) if len(lonv.shape) == 2 else arr(lonv)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("CPP LAT/LON are not rectilinear axes")
    return lat, lon


def cpp_index(lat_axis, lon_axis, lat2d, lon2d):
    lat_row = lat2d[:, lat2d.shape[1] // 2]
    lon_col = lon2d[lon2d.shape[0] // 2, :]
    rr = np.abs(lat_axis[:, None] - lat_row[None, :]).argmin(0)
    cc = np.abs(lon_axis[:, None] - lon_col[None, :]).argmin(0)
    err = max(float(np.max(np.abs(lat2d - lat_axis[rr, None]))),
              float(np.max(np.abs(lon2d - lon_axis[None, cc]))))
    if err > 1e-4 or len(np.unique(rr)) != 16 or len(np.unique(cc)) != 16:
        raise ValueError(f"CPP grid mismatch: {err}")
    return rr, cc, err


def main():
    p = argparse.ArgumentParser()
    for k in ("run", "pack", "candidates", "grid-static", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.run / "validation_predictions.npz", allow_pickle=False) as z:
        ids = z["sample_indices"].astype(np.int64)
        mask = z["mask"].astype(bool)
    with np.load(a.grid_static, allow_pickle=False) as z:
        glat, glon = z["lat"].astype(np.float32), z["lon"].astype(np.float32)
    if glat.shape != (256, 256) or glon.shape != (256, 256):
        raise ValueError("unexpected Hunan grid shape")

    with (a.pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["state"] == "SUPERVISED"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not np.array_equal(ids, np.asarray([int(r["array_index"]) for r in val_rows])):
        raise ValueError("saved validation order differs from immutable pack rows")
    if mask.shape != (len(val_rows), 1, 16, 16):
        raise ValueError(f"unexpected mask shape {mask.shape}")
    with (a.candidates).open(newline="", encoding="utf-8-sig") as f:
        cand = {(r["station"].lower(), r["target_time_utc"]): r for r in csv.DictReader(f)}
    grouped = defaultdict(list)
    for i, r in enumerate(val_rows):
        key = (r["station"].lower(), r["target_time_utc"])
        c = cand.get(key)
        if c is None or c["split"] != "val":
            raise ValueError(f"validation row missing from frozen candidate manifest: {key}")
        grouped[r["target_time_utc"]].append((i, r, c))

    clp = np.full((len(val_rows), 1, 16, 16), 255, np.uint8)
    maxerr = 0.0
    files_read = 0
    for utc, group in sorted(grouped.items()):
        paths = {r["cpp_path"] for _, r, _ in group}
        if len(paths) != 1:
            raise ValueError(f"same timestamp maps to multiple CPP paths: {utc}")
        path = Path(next(iter(paths)))
        if not path.is_file():
            raise FileNotFoundError(path)
        with netCDF4.Dataset(path) as nc:
            if "CLP" not in nc.variables:
                raise ValueError(f"CLP variable missing: {path}")
            lat_axis, lon_axis = axes(nc)
            for i, _, c in group:
                cr, cc = map(int, c["station_grid_rc"].split(","))
                sl = np.s_[cr - 8:cr + 8, cc - 8:cc + 8]
                rr, cols, err = cpp_index(lat_axis, lon_axis, glat[sl], glon[sl])
                maxerr = max(maxerr, err)
                rlo, rhi, clo, chi = int(rr.min()), int(rr.max()) + 1, int(cols.min()), int(cols.max()) + 1
                x = arr(nc.variables["CLP"], (slice(rlo, rhi), slice(clo, chi)))[np.ix_(rr-rlo, cols-clo)]
                if np.diff(rr).mean() < 0:
                    x = x[::-1]
                if np.diff(cols).mean() < 0:
                    x = x[:, ::-1]
                rounded = np.rint(x)
                ok = np.isfinite(x) & (np.abs(x - rounded) < 1e-4) & np.isin(rounded, (0, 1, 2))
                clp[i, 0][ok] = rounded[ok].astype(np.uint8)
        files_read += 1

    common = mask[:, 0] & (clp[:, 0] != 255)
    if not common.any():
        raise RuntimeError("no common valid CLP/COT validation pixels")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, clp=clp, clp_valid=(clp != 255), common_mask=common[:, None],
                        sample_indices=ids, station=np.asarray([r["station"] for r in val_rows]),
                        target_time_utc=np.asarray([r["target_time_utc"] for r in val_rows]))
    audit = {
        "state": "PASS_VALIDATION_CLP_EXTRACTION",
        "split": "validation_only",
        "samples": len(val_rows),
        "unique_cpp_files_read": files_read,
        "valid_cot_mask_pixels": int(mask.sum()),
        "common_valid_clp_pixels": int(common.sum()),
        "clp_coverage_of_cot_mask": float(common.sum() / mask.sum()),
        "clp_class_counts_on_common_mask": {str(k): int(np.count_nonzero(common & (clp[:, 0] == k))) for k in (0, 1, 2)},
        "max_grid_coordinate_error_deg": maxerr,
        "candidate_sha256": sha(a.candidates),
        "grid_static_sha256": sha(a.grid_static),
        "test_payloads_read": 0,
        "note": "CLP 0=clear, 1=water cloud, 2=ice cloud; mask is intersection with saved CPP-COT validation mask",
        "clp_file": str(a.output),
        "clp_file_sha256": sha(a.output),
    }
    audit_path = a.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
