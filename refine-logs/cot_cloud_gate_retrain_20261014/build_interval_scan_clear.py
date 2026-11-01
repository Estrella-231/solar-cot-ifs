#!/usr/bin/env python3
"""Compute scan-time Ineichen clear-sky values for the frozen train/val manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.misc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def import_archived_pvlib(root: Path):
    # The archived pvlib 0.9.1 bundle predates NumPy 2 / SciPy 1.12. These
    # aliases only permit import; neither is used by SPA/Ineichen evaluation.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(scipy.misc, "derivative"):
        def derivative(func, x0, dx=1e-6, n=1, args=(), order=3):
            if n == 1:
                return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2 * dx)
            if n == 2:
                return (func(x0 + dx, *args) - 2 * func(x0, *args) + func(x0 - dx, *args)) / (dx * dx)
            raise ValueError("Archived pvlib import compatibility supports derivative orders 1 and 2 only")
        scipy.misc.derivative = derivative
    sys.path.insert(0, str(root))
    import pvlib
    if pvlib.__version__ != "0.9.1":
        raise RuntimeError(f"Expected archived pvlib 0.9.1, got {pvlib.__version__}")
    return pvlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--pvlib-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pvlib = import_archived_pvlib(args.pvlib_root)
    manifest = pd.read_csv(args.manifest, dtype=str)
    rows = pd.read_csv(args.rows, dtype={"target_time_utc": str})
    if not set(manifest["split"]).issubset({"train", "val"}) or not set(rows["split"]).issubset({"train", "val"}):
        raise ValueError("Only train/validation inputs are allowed")
    merged = manifest.merge(rows, on=["split", "station", "target_time_utc"], how="left", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"Manifest/row join incomplete: {merged['_merge'].value_counts().to_dict()}")
    if len(merged) != 54461:
        raise ValueError(f"Unexpected manifest count: {len(merged)}")
    if not np.isfinite(merged[["ghi", "clear", "kt"]].to_numpy(dtype=np.float64)).all() or (merged["clear"] <= 0).any():
        raise ValueError("Invalid observed label or interval clear-sky denominator")

    offsets = np.stack([np.fromstring(s, sep="|") for s in merged["scan_offsets_relative_target_seconds"]])
    weights = np.stack([np.fromstring(s, sep="|") for s in merged["weights"]])
    if offsets.shape != (len(merged), 3) or weights.shape != offsets.shape:
        raise ValueError("Expected three scan offsets and temporal weights per target")
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > 1e-10 or np.min(weights) < 0:
        raise ValueError("Invalid temporal integration weights")

    target_utc = pd.DatetimeIndex(pd.to_datetime(merged["target_time_utc"], utc=True))
    target_bjt = target_utc.tz_convert("Asia/Shanghai")
    stations = {
        "sili": (25.9478611111, 112.347916667),
        "zhujia": (29.566666667, 112.45),
    }
    scan_clear = np.full((len(merged), 3), np.nan, dtype=np.float64)
    clear15_rebuilt = np.full(len(merged), np.nan, dtype=np.float64)
    for station, (lat, lon) in stations.items():
        ids = np.flatnonzero(merged["station"].to_numpy() == station)
        loc = pvlib.location.Location(lat, lon, tz="Asia/Shanghai", altitude=0.0, name=station)
        times_utc = pd.DatetimeIndex(np.repeat(target_utc[ids].asi8, 3) + np.rint(offsets[ids].reshape(-1) * 1e9).astype(np.int64), tz="UTC")
        values = loc.get_clearsky(times_utc.tz_convert("Asia/Shanghai"), model="ineichen")["ghi"].to_numpy(dtype=np.float64)
        scan_clear[ids] = values.reshape(-1, 3)
        # Reconstruct the existing target-window C15 contract: three 5-minute
        # midpoints at T-12.5, T-7.5 and T-2.5 minutes, averaged equally.
        mids = pd.DatetimeIndex(np.concatenate([
            (target_bjt[ids] - pd.Timedelta(minutes=k)).asi8 for k in (12.5, 7.5, 2.5)
        ]), tz="Asia/Shanghai")
        c15 = loc.get_clearsky(mids, model="ineichen")["ghi"].to_numpy(dtype=np.float64).reshape(3, -1).mean(axis=0)
        clear15_rebuilt[ids] = c15

    if not np.isfinite(scan_clear).all() or (scan_clear < 0).any():
        raise ValueError("Nonfinite or negative scan-time Ineichen clear-sky GHI")
    effective = weights * scan_clear
    if np.any(effective.sum(axis=1) <= 0):
        bad = np.flatnonzero(effective.sum(axis=1) <= 0)[:10].tolist()
        raise ValueError(f"No positive clear-sky temporal support for rows {bad}")
    c15_diff = clear15_rebuilt - merged["clear"].to_numpy(dtype=np.float64)
    max_abs = float(np.max(np.abs(c15_diff)))
    p99_abs = float(np.quantile(np.abs(c15_diff), 0.99))
    if max_abs > 0.05:
        raise ValueError(f"Rebuilt C15 does not reproduce label contract: max abs diff {max_abs:.6g} W/m2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "station", "target_time_utc", "scan_clear_0_wm2", "scan_clear_1_wm2", "scan_clear_2_wm2",
              "clear15_rebuilt_wm2", "clear15_saved_wm2", "max_c15_abs_diff_wm2"]
    out = merged[["split", "station", "target_time_utc"]].copy()
    out[fields[3]] = scan_clear[:, 0]
    out[fields[4]] = scan_clear[:, 1]
    out[fields[5]] = scan_clear[:, 2]
    out[fields[6]] = clear15_rebuilt
    out[fields[7]] = merged["clear"].to_numpy(dtype=np.float64)
    out[fields[8]] = np.abs(c15_diff)
    out.to_csv(args.output, index=False, float_format="%.12g")
    summary = {
        "state": "SCAN_CLEAR_VALUES_READY" if max_abs <= 0.05 else "C15_REPRODUCTION_FAILED",
        "rows": int(len(out)),
        "split_counts": {f"{sp}/{st}": int(((out.split == sp) & (out.station == st)).sum()) for sp in ("train", "val") for st in stations},
        "stations": stations,
        "pvlib_version": pvlib.__version__,
        "clear_sky_model": "Ineichen",
        "altitude_m": 0,
        "target_clear_contract": "mean of pvlib Ineichen at T-12.5, T-7.5, T-2.5 min",
        "scan_clear_contract": "pvlib Ineichen at each actual HDF scan time, nominal target UTC + audited per-frame scan offset",
        "test_payloads_read": 0,
        "max_C15_abs_difference_wm2": max_abs,
        "p99_C15_abs_difference_wm2": p99_abs,
        "scan_clear_min_max_wm2": [float(scan_clear.min()), float(scan_clear.max())],
        "manifest_sha256": sha256(args.manifest),
        "rows_source_sha256": sha256(args.rows),
        "output_sha256": sha256(args.output),
        "source_package": str(args.pvlib_root),
        "source_package_version_file_sha256": sha256(args.pvlib_root / "pvlib" / "version.py"),
    }
    sidecar = args.output.with_suffix(".audit.json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
