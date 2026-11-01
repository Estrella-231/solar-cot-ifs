#!/usr/bin/env python3
"""Extract correctly time-matched CPP COT/CLP patches for Hubei train/val.

Input sample keys are BJT. CPP NetCDF filenames are UTC, so each source key is
computed as BJT - 8 hours. Spatial selection is tied to the shared FY4B crop
grid only after strict station-coordinate, raw-index, and AGRI-angle checks.
This script never reads test/buffer rows or payloads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np


EXPECTED_SPLITS = {"train", "val"}
PATCH_ORIGIN = (2, 134)
PATCH_SIZE = 16
STATIONS = {
    "Niushou": {"local": (8, 6), "hdf": (557, 1535), "lat": 32.0800, "lon": 112.0400},
    "Wangzhai": {"local": (9, 8), "hdf": (558, 1537), "lat": 32.0400, "lon": 112.1200},
    "Mizhuang": {"local": (7, 10), "hdf": (556, 1538), "lat": 32.1200, "lon": 112.2000},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def map_axis(axis: np.ndarray, target: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    axis = np.asarray(axis, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if axis.ndim != 1 or target.ndim != 1 or not np.isfinite(axis).all() or not np.isfinite(target).all():
        raise ValueError(f"invalid {name} coordinates")
    delta = np.diff(axis)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(f"CPP {name} axis is not strictly monotonic")
    idx = np.abs(axis[:, None] - target[None, :]).argmin(axis=0)
    err = float(np.max(np.abs(axis[idx] - target)))
    if err > 1e-4 or len(np.unique(idx)) != len(idx):
        raise ValueError(f"CPP {name} mapping failed: max_error={err}, unique={len(np.unique(idx))}/{len(idx)}")
    return idx, err


def check_grid(grid_path: Path, agri_samples: list[Path]) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(grid_path, allow_pickle=False)
    grid = {k: np.asarray(z[k]) for k in ("lat", "lon", "saa", "saz", "row_index", "col_index")}
    r0, c0 = PATCH_ORIGIN
    sl = (slice(r0, r0 + PATCH_SIZE), slice(c0, c0 + PATCH_SIZE))
    patch = {k: v[sl] for k, v in grid.items()}
    if any(v.shape != (PATCH_SIZE, PATCH_SIZE) for v in patch.values()):
        raise ValueError("grid crop does not contain a 16x16 patch")
    station_checks = {}
    for name, spec in STATIONS.items():
        rr, cc = spec["local"]
        r, c = rr + r0, cc + c0
        got = (int(grid["row_index"][r, c]), int(grid["col_index"][r, c]))
        expected = tuple(spec["hdf"])
        lat, lon = float(grid["lat"][r, c]), float(grid["lon"][r, c])
        if got != expected or abs(lat - spec["lat"]) > 1e-4 or abs(lon - spec["lon"]) > 1e-4:
            raise ValueError(f"{name} grid witness mismatch: index={got}, latlon={(lat, lon)}")
        station_checks[name] = {"grid_local": [rr, cc], "raw_grid_index": list(got), "lat": lat, "lon": lon}

    # The same AGRI tile must carry the static FY4B satellite angles. Check a
    # stratified set of train/val source frames; do not inspect any other split.
    sample_checks = []
    for p in agri_samples:
        with np.load(p, allow_pickle=False) as a:
            for name, key in (("SAA", "saa"), ("SAZ", "saz")):
                if not np.array_equal(np.asarray(a[name]), patch[key]):
                    raise ValueError(f"AGRI static-angle grid mismatch: {p} {name}")
        sample_checks.append(str(p))
    detail = {
        "grid_sha256": sha256(grid_path),
        "patch_origin_in_grid_static": list(PATCH_ORIGIN),
        "patch_size": PATCH_SIZE,
        "station_witnesses": station_checks,
        "agri_static_angle_exact_checks": sample_checks,
    }
    return patch, detail


def _extract_one_once(task):
    """Worker: read exactly one train/val AGRI key and two CPP variables."""
    i, row, src_text, lat, lon = task
    src = Path(src_text)
    try:
        with netCDF4.Dataset(src, "r") as ds:
            if not all(v in ds.variables for v in ("LAT", "LON", "COT", "CLP")):
                raise ValueError("required LAT/LON/COT/CLP variable missing")
            ri, _ = map_axis(ds.variables["LAT"][:], lat[:, 0], "latitude")
            ci, _ = map_axis(ds.variables["LON"][:], lon[0, :], "longitude")
            lat_err = float(np.max(np.abs(np.asarray(ds.variables["LAT"][ri])[:, None] - lat)))
            lon_err = float(np.max(np.abs(np.asarray(ds.variables["LON"][ci])[None, :] - lon)))
            if max(lat_err, lon_err) > 1e-4:
                raise ValueError(f"2D coordinate mismatch: lat={lat_err}, lon={lon_err}")
            rlo, rhi = int(ri.min()), int(ri.max()) + 1
            clo, chi = int(ci.min()), int(ci.max()) + 1
            cot_src = ds.variables["COT"][rlo:rhi, clo:chi]
            clp_src = ds.variables["CLP"][rlo:rhi, clo:chi]
            cot_block = np.asarray(np.ma.filled(np.ma.asarray(cot_src, dtype=np.float32), np.nan), dtype=np.float32)
            clp_block = np.asarray(np.ma.filled(np.ma.asarray(clp_src, dtype=np.float32), np.nan), dtype=np.float32)
            cot_patch = cot_block[np.ix_(ri - rlo, ci - clo)]
            clp_patch = clp_block[np.ix_(ri - rlo, ci - clo)]
            cv = np.isfinite(cot_patch) & (cot_patch >= 0) & (cot_patch <= 100)
            lv = np.isfinite(clp_patch) & np.isin(clp_patch, (0, 1, 2))
            bjt = datetime.strptime(row["time_bjt"], "%Y%m%d%H%M")
            utc = bjt - timedelta(hours=8)
            stamp = utc.strftime("%Y%m%d%H%M%S")
            return i, {
                "cot": np.where(cv, cot_patch, np.nan).astype(np.float32),
                "cot_mask": cv.astype(np.uint8),
                "clp": np.where(lv, clp_patch, 255).astype(np.uint8),
                "clp_mask": lv.astype(np.uint8),
                "row": {
                    "index": i,
                    "time_bjt": row["time_bjt"],
                    "time_cpp_utc": stamp,
                    "split": row["split"],
                    "path_agri": row["path_agri"],
                    "path_cpp_correct_utc": str(src),
                    "source_bytes": src.stat().st_size,
                    "valid_cot_pixels": int(cv.sum()),
                    "valid_clp_pixels": int(lv.sum()),
                    "cloudy_clp_pixels": int((lv & np.isin(clp_patch, (1, 2))).sum()),
                    "max_lat_error_deg": lat_err,
                    "max_lon_error_deg": lon_err,
                },
            }
    except Exception as e:
        return i, {"failure": {"time_bjt": row["time_bjt"], "split": row["split"], "error": repr(e), "path": str(src)}}


def extract_one(task):
    """Retry transient NetCDF/HDF read failures without hiding final errors."""
    last = None
    for attempt in range(1, 4):
        index, result = _extract_one_once(task)
        if "failure" not in result:
            return index, result
        last = result["failure"]
        if attempt < 3:
            time.sleep(0.25 * attempt)
    last = dict(last)
    last["attempts"] = 3
    return index, {"failure": last}


def run(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    grid_path = Path(args.grid_static)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8-sig")))
    if not rows or any(r.get("split") not in EXPECTED_SPLITS for r in rows):
        raise ValueError("manifest must contain train/val rows only")
    keys = [r["time_bjt"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate BJT keys in R train/val manifest")
    samples = []
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            raise ValueError(f"missing {split} rows")
        for ix in sorted({0, len(subset) // 2, len(subset) - 1}):
            samples.append(Path(subset[ix]["path_agri"]))
    patch, grid_detail = check_grid(grid_path, samples)
    out.mkdir(parents=True, exist_ok=False)
    lat = np.asarray(patch["lat"], dtype=np.float64)
    lon = np.asarray(patch["lon"], dtype=np.float64)
    cot_all = np.full((len(rows), PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
    clp_all = np.full((len(rows), PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
    cot_mask = np.zeros_like(cot_all, dtype=np.uint8)
    clp_mask = np.zeros_like(clp_all, dtype=np.uint8)
    output_rows = []
    failures = []
    max_lat_err = 0.0
    max_lon_err = 0.0
    cpp_last = max(datetime.strptime(r["time_bjt"], "%Y%m%d%H%M") - timedelta(hours=8) for r in rows)

    tasks = []
    for i, row in enumerate(rows):
        bjt = datetime.strptime(row["time_bjt"], "%Y%m%d%H%M")
        utc = bjt - timedelta(hours=8)
        stamp = utc.strftime("%Y%m%d%H%M%S")
        src = Path(args.cpp_root) / utc.strftime("%Y") / utc.strftime("%Y%m%d") / f"FY4B_AGRI_{stamp}.nc"
        if not src.is_file():
            failures.append({"time_bjt": row["time_bjt"], "split": row["split"], "error": "correct_utc_cpp_missing", "path": str(src)})
            continue
        tasks.append((i, row, str(src), lat, lon))

    if args.workers <= 1:
        result_iter = map(extract_one, tasks)
        for done, (i, result) in enumerate(result_iter, start=1):
            if "failure" in result:
                failures.append(result["failure"])
            else:
                cot_all[i] = result["cot"]
                cot_mask[i] = result["cot_mask"]
                clp_all[i] = result["clp"]
                clp_mask[i] = result["clp_mask"]
                output_rows.append(result["row"])
                max_lat_err = max(max_lat_err, result["row"]["max_lat_error_deg"])
                max_lon_err = max(max_lon_err, result["row"]["max_lon_error_deg"])
            if done % 50 == 0 or done == len(tasks):
                print(f"progress {done}/{len(tasks)} success={len(output_rows)} failures={len(failures)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            result_iter = pool.map(extract_one, tasks, chunksize=1)
            for done, (i, result) in enumerate(result_iter, start=1):
                if "failure" in result:
                    failures.append(result["failure"])
                else:
                    cot_all[i] = result["cot"]
                    cot_mask[i] = result["cot_mask"]
                    clp_all[i] = result["clp"]
                    clp_mask[i] = result["clp_mask"]
                    output_rows.append(result["row"])
                    max_lat_err = max(max_lat_err, result["row"]["max_lat_error_deg"])
                    max_lon_err = max(max_lon_err, result["row"]["max_lon_error_deg"])
                if done % 50 == 0 or done == len(tasks):
                    print(f"progress {done}/{len(tasks)} success={len(output_rows)} failures={len(failures)}", flush=True)

    if failures:
        (out / "extraction_failures.json").write_text(
            json.dumps({"state": "FAIL_CORRECT_UTC_CPP_EXTRACTION", "manifest_sha256": sha256(manifest),
                        "rows": len(rows), "successes": len(output_rows), "failures": failures,
                        "test_buffer_rows_read": 0, "test_buffer_payloads_read": 0},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"extraction failed for {len(failures)}/{len(rows)} rows; see extraction_failures.json")
    tmp = out / "cpp_supervision_trainval.tmp.npz"
    final = out / "cpp_supervision_trainval.npz"
    np.savez_compressed(
        tmp,
        time_bjt=np.asarray([r["time_bjt"] for r in rows]),
        split=np.asarray([r["split"] for r in rows]),
        cot=cot_all,
        cot_valid_mask=cot_mask,
        clp=clp_all,
        clp_valid_mask=clp_mask,
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        patch_origin=np.asarray(PATCH_ORIGIN, dtype=np.int32),
        station_local_centers=np.asarray([STATIONS[k]["local"] for k in STATIONS], dtype=np.int32),
        station_names=np.asarray(list(STATIONS)),
    )
    os.replace(tmp, final)
    with (out / "rows_correct_utc.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        w.writeheader()
        w.writerows(output_rows)
    report = {
        "state": "COMPLETE_CORRECT_UTC_CPP_TRAIN_VAL_SIDEcar_TEST_CLOSED" if len(output_rows) == len(rows) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "grid_static": str(grid_path),
        "grid_detail": grid_detail,
        "cpp_root": str(args.cpp_root),
        "rows": len(rows),
        "source_rows_extracted": len(output_rows),
        "counts_by_split": {s: sum(r["split"] == s for r in rows) for s in sorted(EXPECTED_SPLITS)},
        "all_CPP_keys": len(output_rows) == len(rows),
        "test_buffer_rows_read": 0,
        "test_buffer_payloads_read": 0,
        "time_rule": "CPP UTC timestamp = AGRI/manifest BJT timestamp - 8 hours",
        "spatial_rule": "nearest existing CPP LAT/LON axis values; no interpolation; unique 16x16 mapping; <=1e-4 degree",
        "max_coordinate_error_deg": {"lat": max_lat_err, "lon": max_lon_err},
        "valid_cot_pixels_by_split": {s: int(sum(r["valid_cot_pixels"] for r in output_rows if r["split"] == s)) for s in sorted(EXPECTED_SPLITS)},
        "valid_clp_pixels_by_split": {s: int(sum(r["valid_clp_pixels"] for r in output_rows if r["split"] == s)) for s in sorted(EXPECTED_SPLITS)},
        "cloudy_clp_pixels_by_split": {s: int(sum(r["cloudy_clp_pixels"] for r in output_rows if r["split"] == s)) for s in sorted(EXPECTED_SPLITS)},
        "sidecar_sha256": sha256(final),
        "arrays": {"cot": array_sha256(cot_all), "cot_valid_mask": array_sha256(cot_mask), "clp": array_sha256(clp_all), "clp_valid_mask": array_sha256(clp_mask)},
        "failures": failures,
        "latest_cpp_utc": cpp_last.isoformat() if cpp_last else None,
        "interpretation": "CPP CLP/COT are reference supervision, not independent truth; this sidecar is train/validation only",
    }
    (out / "extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--grid-static", required=True)
    p.add_argument("--cpp-root", default="/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1)
    run(p.parse_args())


if __name__ == "__main__":
    main()
