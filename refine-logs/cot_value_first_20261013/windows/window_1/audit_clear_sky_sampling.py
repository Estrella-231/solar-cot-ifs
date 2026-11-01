"""Compare clear-sky denominator timestamp conventions on candidate keys.

Requires the project Python environment with pvlib/numpy. Reads only the
train/validation timestamp candidate CSV (no station observations, AGRI, CPP,
or test payload). Compare average Ineichen GHI at three interval endpoints
against the three matching five-minute interval midpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib


STATIONS = {
    "sili": (25.9478611111, 112.347916667),
    "zhujia": (29.566666667, 112.45),
}
LOCAL = ZoneInfo("Asia/Shanghai")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.9)),
        "max": float(x.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)

    grouped = defaultdict(list)
    with a.input.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "val"}:
                raise ValueError("test keys are prohibited")
            grouped[row["station"]].append(row)

    a.output_dir.mkdir(parents=True)
    fields = ["split", "station", "target_time_bjt", "solar_zenith_deg",
              "clear_endpoint_mean_wm2", "clear_midpoint_mean_wm2",
              "difference_midpoint_minus_endpoint_wm2", "absolute_difference_wm2",
              "relative_difference_percent", "endpoint_day_positive",
              "midpoint_day_positive", "endpoint_day_ge100", "midpoint_day_ge100"]
    details_path = a.output_dir / "clear_sky_sampling_by_target.csv"
    reports = {}
    with details_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for name in sorted(grouped):
            if name not in STATIONS:
                raise ValueError(f"unknown station: {name}")
            rows = grouped[name]
            times_bjt = pd.DatetimeIndex(pd.to_datetime([r["target_time_bjt"] for r in rows]))
            if times_bjt.tz is None:
                times_bjt = times_bjt.tz_localize(LOCAL)
            else:
                times_bjt = times_bjt.tz_convert(LOCAL)
            lat, lon = STATIONS[name]
            location = pvlib.location.Location(lat, lon, tz="Asia/Shanghai", altitude=0.0, name=name)

            endpoint_samples, midpoint_samples = [], []
            for minute in (10, 5, 0):
                endpoint_samples.append(location.get_clearsky(
                    times_bjt - pd.Timedelta(minutes=minute), model="ineichen")["ghi"].to_numpy())
            for minute in (12.5, 7.5, 2.5):
                midpoint_samples.append(location.get_clearsky(
                    times_bjt - pd.Timedelta(minutes=minute), model="ineichen")["ghi"].to_numpy())
            endpoint = np.mean(endpoint_samples, axis=0)
            midpoint = np.mean(midpoint_samples, axis=0)
            sza = location.get_solarposition(times_bjt)["apparent_zenith"].to_numpy()
            day = np.isfinite(sza) & (sza < 90.0)
            delta = midpoint - endpoint
            denom = np.maximum(endpoint, 1.0)
            rel = 100.0 * delta / denom
            reports[name] = {}

            for split in ("train", "val"):
                ids = np.array([i for i, r in enumerate(rows) if r["split"] == split], dtype=np.int64)
                usable = ids[day[ids]]
                reports[name][split] = {
                    "candidate_targets": int(len(ids)),
                    "station_day_targets_sza_lt90": int(len(usable)),
                    "endpoint_clear_sky_wm2": stats(endpoint[usable]),
                    "midpoint_clear_sky_wm2": stats(midpoint[usable]),
                    "signed_difference_midpoint_minus_endpoint_wm2": stats(delta[usable]),
                    "absolute_difference_wm2": stats(np.abs(delta[usable])),
                    "relative_difference_percent_endpoint_denominator": stats(rel[usable]),
                    "positive_support_disagreements": int(np.sum((endpoint[usable] > 0) != (midpoint[usable] > 0))),
                    "gte100_support_disagreements": int(np.sum((endpoint[usable] >= 100) != (midpoint[usable] >= 100))),
                }

            for i, row in enumerate(rows):
                writer.writerow({
                    "split": row["split"], "station": name,
                    "target_time_bjt": times_bjt[i].isoformat(),
                    "solar_zenith_deg": f"{sza[i]:.8f}",
                    "clear_endpoint_mean_wm2": f"{endpoint[i]:.8f}",
                    "clear_midpoint_mean_wm2": f"{midpoint[i]:.8f}",
                    "difference_midpoint_minus_endpoint_wm2": f"{delta[i]:.8f}",
                    "absolute_difference_wm2": f"{abs(delta[i]):.8f}",
                    "relative_difference_percent": f"{rel[i]:.8f}",
                    "endpoint_day_positive": bool(day[i] and endpoint[i] > 0),
                    "midpoint_day_positive": bool(day[i] and midpoint[i] > 0),
                    "endpoint_day_ge100": bool(day[i] and endpoint[i] >= 100),
                    "midpoint_day_ge100": bool(day[i] and midpoint[i] >= 100),
                })

    report = {
        "state": "CLEAR_SKY_TIMESTAMP_COMPARISON_COMPLETE",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "input_sha256": sha(a.input),
        "code_sha256": sha(Path(__file__)),
        "test_payload_opened": False,
        "observed_GHI_read": False,
        "cpp_or_agri_payload_opened": False,
        "clear_sky_model": "Ineichen, pvlib",
        "pvlib_version": pvlib.__version__,
        "interval_endpoint_offsets_minutes": [10, 5, 0],
        "interval_midpoint_offsets_minutes": [12.5, 7.5, 2.5],
        "primary_day_support": "station apparent zenith <90 degrees and the selected interval-mean clear-sky GHI >0",
        "reports": reports,
        "details_file": details_path.name,
        "details_sha256": sha(details_path),
        "interpretation_limit": "This compares denominator time sampling only; it does not establish observed-GHI label validity or replace the agreed target interval.",
    }
    report_path = a.output_dir / "clear_sky_sampling_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
