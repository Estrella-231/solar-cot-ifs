"""Join train/val-only station GHI labels with the frozen clear-sky sampling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def key(station: str, value: str) -> tuple[str, str]:
    t = datetime.fromisoformat(value.strip())
    if t.tzinfo is not None:
        from datetime import timezone, timedelta
        t = t.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
    return station, t.isoformat(sep=" ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--clear-sky", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    labels = {}
    with a.labels.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = key(row["station"], row["target_time_bjt"])
            if k in labels:
                raise ValueError(f"duplicate label key: {k}")
            labels[k] = row
    clear = {}
    with a.clear_sky.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = key(row["station"], row["target_time_bjt"])
            if k in clear:
                raise ValueError(f"duplicate clear-sky key: {k}")
            clear[k] = row
    if labels.keys() != clear.keys():
        raise ValueError(f"key mismatch labels={len(labels)} clear={len(clear)}")

    a.output_dir.mkdir(parents=True)
    fields = ["new_split", "station", "target_time_bjt", "target_time_utc", "agri_frame_relpath",
              "label_record_times_bjt", "raw_ghi_5min_wm2", "observed_ghi_15min_wm2",
              "day_mask_sza_lt_90", "clear_sky_ghi_15min_wm2", "solar_zenith_apparent_deg",
              "kt", "main_mask", "label_source"]
    counts = Counter()
    out = a.output_dir / "ghi_kt_train_val.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for k, row in sorted(labels.items(), key=lambda x: (x[1]["new_split"], x[0])):
            cs = clear[k]
            ghi = float(row["observed_ghi_15min_wm2"]) if row["label_status"] == "complete_ghi" else math.nan
            clear_mid = float(cs["clear_midpoint_mean_wm2"])
            day = row["day_mask_sza_lt_90"] == "True"
            cs_valid = math.isfinite(clear_mid) and clear_mid > 0.0
            valid = math.isfinite(ghi) and ghi >= 0 and day and cs_valid
            counts[(row["new_split"], "candidate")] += 1
            if day: counts[(row["new_split"], "station_day_noaa")] += 1
            if row["label_status"] == "complete_ghi": counts[(row["new_split"], "complete_ghi")] += 1
            if day and row["label_status"] != "complete_ghi": counts[(row["new_split"], "day_incomplete_ghi")] += 1
            if cs_valid: counts[(row["new_split"], "positive_midpoint_clear_sky")] += 1
            if valid: counts[(row["new_split"], "main_mask")] += 1
            w.writerow({"new_split": row["new_split"], "station": row["station"],
                "target_time_bjt": row["target_time_bjt"], "target_time_utc": row["target_time_utc"],
                "agri_frame_relpath": row["agri_frame_relpath"],
                "label_record_times_bjt": row["label_record_times_bjt"],
                "raw_ghi_5min_wm2": row["raw_ghi_5min_wm2"],
                "observed_ghi_15min_wm2": row["observed_ghi_15min_wm2"],
                "day_mask_sza_lt_90": row["day_mask_sza_lt_90"],
                "clear_sky_ghi_15min_wm2": f"{clear_mid:.12g}",
                "solar_zenith_apparent_deg": cs["solar_zenith_deg"],
                "kt": f"{ghi/clear_mid:.12g}" if valid else "",
                "main_mask": str(valid), "label_source": row["label_source"]})
    report = {"state": "TRAIN_VAL_GHI_KT_LABEL_JOIN_COMPLETE", "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "candidate_keys": len(labels), "counts": {"/".join(k): v for k,v in sorted(counts.items())},
        "label_window": "(target-15min,target]; GHI mean from target-10/-5/target samples",
        "clear_sky": "Ineichen pvlib; mean at 5-minute interval midpoints target-12.5/-7.5/-2.5 min",
        "day_mask_source": "NOAA solar cosine at target station/time > 0, consistent with M0 label production",
        "kt": "observed 15-min mean GHI / midpoint-sampled 15-min mean clear-sky GHI; unclipped",
        "test_payload_opened": False, "input_hashes": {"labels": sha(a.labels), "clear_sky": sha(a.clear_sky)},
        "output": out.name, "output_sha256": sha(out)}
    (a.output_dir / "ghi_kt_join_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
