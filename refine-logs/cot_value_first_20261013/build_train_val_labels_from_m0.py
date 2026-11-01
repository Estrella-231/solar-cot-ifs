"""Join train/val candidates to a split-selected label table; no scope receipt required."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def bjt_wallclock(value: str) -> datetime:
    """Normalize naive BJT or offset-aware BJT strings to one wall-clock key."""
    value = value.strip()
    parsed = dt(value)
    if parsed.tzinfo is None:
        return parsed
    from zoneinfo import ZoneInfo
    return parsed.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--canonical-labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)

    candidates = {}
    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "val"}:
                raise ValueError("candidate split must be train or val")
            key = (row["station"], bjt_wallclock(row["target_time_bjt"]).isoformat(sep=" "))
            if key in candidates:
                raise ValueError(f"duplicate candidate key {key}")
            candidates[key] = row

    labels = {}
    label_columns = None
    with a.canonical_labels.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        label_columns = reader.fieldnames or []
        required = {"station", "target_time_bjt", "label_record_times_bjt",
                    "raw_ghi_5min_wm2", "observed_ghi_15min_wm2",
                    "day_mask_sza_lt_90", "split"}
        missing = sorted(required - set(label_columns))
        if missing:
            raise ValueError(f"label manifest missing columns: {missing}")
        source_rows_scanned = 0
        for row in reader:
            source_rows_scanned += 1
            key = (row["station"], bjt_wallclock(row["target_time_bjt"]).isoformat(sep=" "))
            if key not in candidates:
                # The canonical M0 manifest spans multiple historical splits.
                # Only candidate train/val rows may be used; skip all other
                # keys before accessing any label-value columns.
                continue
            # The old forecast-window manifest repeats a target under several
            # init/lead pairs. Collapse only exact duplicate label payloads;
            # preserve its historical split separately from the new split.
            if key in labels:
                prior = labels[key]
                compared = ("label_record_times_bjt", "raw_ghi_5min_wm2",
                            "observed_ghi_15min_wm2", "day_mask_sza_lt_90")
                if any(prior.get(name) != row.get(name) for name in compared):
                    raise ValueError(f"conflicting repeated source labels for {key}")
                prior["_repeat_count"] = str(int(prior.get("_repeat_count", "1")) + 1)
            else:
                row["_repeat_count"] = "1"
                labels[key] = row

    a.output_dir.mkdir(parents=True)
    out = a.output_dir / "unique_target_labels_train_val.csv"
    fields = ["new_split", "source_split", "source_repeat_count", "station", "target_time_bjt", "target_time_utc",
              "agri_frame_relpath", "label_record_times_bjt", "raw_ghi_5min_wm2",
              "observed_ghi_15min_wm2", "day_mask_sza_lt_90", "label_status"]
    counts = Counter()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, candidate in sorted(candidates.items(), key=lambda item: (item[1]["split"], item[0])):
            normalized_key = (key[0], bjt_wallclock(key[1]).isoformat(sep=" "))
            row = labels.get(normalized_key)
            counts[(candidate["split"], "candidate")] += 1
            base = {name: "" for name in fields}
            base.update({"new_split": candidate["split"], "station": candidate["station"],
                         "target_time_bjt": candidate["target_time_bjt"],
                         "target_time_utc": candidate["target_time_utc"],
                         "agri_frame_relpath": candidate["agri_frame_relpath"]})
            if row is None:
                base["label_status"] = "missing_station_label_key"
                counts[(candidate["split"], "missing_key")] += 1
                writer.writerow(base)
                continue
            base["source_split"] = row.get("split", "")
            base["source_repeat_count"] = row.get("_repeat_count", "1")
            target = bjt_wallclock(candidate["target_time_bjt"])
            expected_times = [target-timedelta(minutes=m) for m in (10, 5, 0)]
            actual_times = [bjt_wallclock(x) for x in row["label_record_times_bjt"].split("|")]
            if actual_times != expected_times:
                raise ValueError(f"interval end-record mismatch for {key}: {actual_times}")
            try:
                raw = [float(x) for x in row["raw_ghi_5min_wm2"].split("|")]
                label = float(row["observed_ghi_15min_wm2"])
            except (TypeError, ValueError):
                raw, label = [], math.nan
            complete = (len(raw) == 3 and all(math.isfinite(x) and x >= 0 for x in raw)
                        and math.isfinite(label) and label >= 0)
            if complete and abs(sum(raw)/3.0-label) > 1e-4:
                raise ValueError(f"15-minute mean mismatch for {key}")
            day = row["day_mask_sza_lt_90"] == "True"
            state = "valid_daylight_label" if complete and day else (
                "invalid_or_missing_ghi" if not complete else "outside_station_day_mask")
            base.update({"label_record_times_bjt": row["label_record_times_bjt"],
                         "raw_ghi_5min_wm2": row["raw_ghi_5min_wm2"],
                         "observed_ghi_15min_wm2": row["observed_ghi_15min_wm2"],
                         "day_mask_sza_lt_90": row["day_mask_sza_lt_90"],
                         "label_status": state})
            writer.writerow(base)
            counts[(candidate["split"], "matched_label_key")] += 1
            if complete:
                counts[(candidate["split"], "complete_ghi")] += 1
            if complete and day:
                counts[(candidate["split"], "valid_daylight_label")] += 1

    report = {
        "state": "UNIQUE_TARGET_LABEL_JOIN_COMPLETE",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "candidates_sha256": sha(a.candidates),
        "canonical_label_manifest": a.canonical_labels.name,
        "canonical_source_rows_scanned_by_key": source_rows_scanned,
        "code_sha256": sha(Path(__file__)),
        "candidate_keys": len(candidates),
        "canonical_label_unique_station_target_keys": len(labels),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "label_window": "(target-15min,target]; 5-minute interval-ending values at target-10/-5/target",
        "source_split_is_preserved_for_audit_only": True,
        "new_monthly_split_is_taken_from_candidate_manifest": True,
        "main_sample_mask": "complete finite nonnegative GHI records AND SZA<90deg AND later paired interval-mean clear-sky GHI>0",
        "CPP_used_for_ghi_cohort_filter": False,
        "scope_receipt_required": False,
        "noncandidate_label_values_used": False,
        "label_keys_within_train_val_candidates": True,
        "output_sha256": sha(out),
        "output": out.name,
    }
    report_path = a.output_dir / "unique_target_label_join_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
