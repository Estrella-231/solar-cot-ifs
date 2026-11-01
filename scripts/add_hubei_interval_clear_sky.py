"""Add Hunan-contract Ineichen interval clear-sky values to Hubei train/val labels."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

STATIONS = {
    "Niushou": (32.0800, 112.0400),
    "Wangzhai": (32.0400, 112.1200),
    "Mizhuang": (32.1200, 112.2000),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    inp, out = Path(a.labels), Path(a.output)
    with inp.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["station"] not in STATIONS for r in rows):
        raise ValueError("Unknown station in label file")
    clear = np.full(len(rows), np.nan, dtype=np.float64)
    group_counts = {}
    for station in STATIONS:
        ids = [i for i, r in enumerate(rows) if r["station"] == station]
        stamps = [rows[i]["target_time_BJT"] for i in ids]
        if len(set(stamps)) != len(stamps):
            raise ValueError(f"Duplicate station-target keys: {station}")
        targets = pd.DatetimeIndex(pd.to_datetime(stamps)).tz_localize("Asia/Shanghai")
        loc = pvlib.location.Location(*STATIONS[station], tz="Asia/Shanghai", altitude=0.0, name=station)
        samples = [loc.get_clearsky(targets - pd.Timedelta(minutes=offset), model="ineichen")["ghi"].to_numpy(dtype=np.float64)
                   for offset in (12.5, 7.5, 2.5)]
        means = np.mean(samples, axis=0)
        if not np.isfinite(means).all() or (means < 0).any():
            raise ValueError(f"Invalid clear-sky values for {station}")
        clear[ids] = means
        group_counts[station] = len(ids)
    counters = Counter()
    with inp.open(encoding="utf-8-sig", newline="") as src, out.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fields = reader.fieldnames + ["clear_sky_ghi_15min_wm2", "valid_daylight_clear_sky_label", "sensitivity_clear_ge_100"]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(reader):
            day = row["day_mask_sza_lt_90"] == "True"
            clear_valid = clear[i] > 0.0
            mask = int(day and clear_valid and int(row["mask"]) == 1)
            sensitivity = int(mask and clear[i] >= 100.0)
            row["clear_sky_ghi_15min_wm2"] = f"{clear[i]:.12g}"
            row["valid_daylight_clear_sky_label"] = mask
            row["sensitivity_clear_ge_100"] = sensitivity
            writer.writerow(row)
            station, split = row["station"], row["split"]
            counters[(station, split, "slots")] += 1
            counters[(station, split, "day_sza_lt90")] += int(day)
            counters[(station, split, "positive_clear_sky")] += int(clear_valid)
            counters[(station, split, "valid_main_mask")] += mask
            counters[(station, split, "clear_ge_100_mask")] += sensitivity
    report = {
        "status": "TRAIN_VAL_DAYLIGHT_INEICHEN_SUPPORT_COMPLETE",
        "input_label_sha256": digest(inp),
        "output_sha256": digest(out),
        "test_values_read": 0,
        "pvlib_version": pvlib.__version__,
        "clear_sky_model": "Ineichen",
        "altitude_assumed_m": 0.0,
        "station_coordinates": {k: {"latitude": v[0], "longitude": v[1]} for k, v in STATIONS.items()},
        "interval_integration": "mean of three 5min midpoints: target-12.5/-7.5/-2.5min",
        "main_mask": "complete GHI15 label AND NOAA solar cosine >0 AND interval-mean Ineichen clear-sky GHI >0",
        "sensitivity_mask": "main mask AND interval-mean clear-sky GHI >=100 W/m2",
        "counts_by_station_split": {
            s: {sp: {metric: counters[(s, sp, metric)] for metric in
                     ("slots", "day_sza_lt90", "positive_clear_sky", "valid_main_mask", "clear_ge_100_mask")}
                for sp in ("train", "val")}
            for s in STATIONS
        },
        "normalization": "not fit",
        "unit": "provisional legacy W/m2 convention; source instrument metadata unresolved",
    }
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "pvlib_version": report["pvlib_version"], "counts": report["counts_by_station_split"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
