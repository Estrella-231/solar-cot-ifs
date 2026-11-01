"""Recover June/July source paths from raw HDF filenames; metadata only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re


def stamp_from_name(name):
    match = re.search(r"NOM_(\d{14})_", name)
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S") if match else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--existing-inventory", required=True)
    p.add_argument("--archive-index", required=True)
    p.add_argument("--historical-root", required=True)
    p.add_argument("--realtime-root", required=True)
    p.add_argument("--start", default="2026-06-13")
    p.add_argument("--end", default="2026-07-14")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    start = datetime.fromisoformat(a.start)
    end = datetime.fromisoformat(a.end) + timedelta(days=1)
    utc_start, utc_end = start-timedelta(hours=8), end-timedelta(hours=8)
    candidates = defaultdict(list)
    missing_dirs = []
    day = utc_start.replace(hour=0)
    while day < utc_end:
        for priority, root in enumerate((a.historical_root, a.realtime_root)):
            directory = Path(root)/day.strftime("%Y")/day.strftime("%Y%m%d")
            if not directory.is_dir():
                missing_dirs.append(str(directory))
                continue
            for path in sorted(directory.iterdir()):
                if not (path.name.endswith(".HDF") and "_L1-_" in path.name
                        and "MULT_NOM_" in path.name and "4000M" in path.name):
                    continue
                utc = stamp_from_name(path.name)
                if utc is not None and utc_start <= utc < utc_end:
                    bjt = utc + timedelta(hours=8)
                    if bjt.minute % 15 == 0 and bjt.second == 0:
                        candidates[str(bjt)].append((priority, str(path)))
        day += timedelta(days=1)
    selected = {t: sorted(ps)[0][1] for t, ps in candidates.items()}
    daily = Counter(t[:10] for t in selected)
    # Verify archived clock columns against the explicitly parsed UTC filename.
    mismatches, seen, examples = 0, 0, []
    with open(a.archive_index, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            utc = stamp_from_name(row["source_hdf"])
            if utc is None:
                continue
            seen += 1
            canonical = utc + timedelta(hours=8)
            stored = datetime.fromisoformat(row["time_bjt"])
            if canonical != stored:
                mismatches += 1
                if len(examples) < 3:
                    examples.append({"source_hdf": row["source_hdf"],
                                     "stored_BJT": str(stored), "canonical_BJT": str(canonical)})
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    with (out/"hdf_frame_paths.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["target_time_BJT", "source_HDF", "candidate_paths"])
        for t, path in sorted(selected.items()):
            w.writerow([t, path, "|".join(p for _, p in sorted(candidates[t]))])
    original = list(csv.DictReader(open(a.existing_inventory, encoding="utf-8", newline="")))
    if any(row["date_BJT"] in daily for row in original):
        raise ValueError("Extension overlaps existing date inventory; resolve explicitly")
    with (out/"daily_path_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["date_BJT", "agri_frames", "cpp_frames", "agri_duplicate_paths", "cpp_duplicate_paths"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        rows = original + [{"date_BJT": d, "agri_frames": n, "cpp_frames": 0,
                            "agri_duplicate_paths": sum(len(candidates[t])-1 for t in candidates if t[:10] == d),
                            "cpp_duplicate_paths": 0} for d, n in daily.items()]
        w.writerows(sorted(rows, key=lambda r:r["date_BJT"]))
    report = {"status": "RAW_HDF_METADATA_EXTENSION_NOT_PREPROCESSED",
              "requested_BJT": [a.start, a.end], "unique_HDF_times": len(selected),
              "range_BJT": [min(selected), max(selected)] if selected else [],
              "daily_HDF_counts": dict(sorted(daily.items())),
              "multiple_source_times": sum(len(v)>1 for v in candidates.values()),
              "source_precedence": "historical before realtime, then lexical; equivalence not asserted",
              "archived_index_rows": seen, "archived_BJT_clock_mismatches": mismatches,
              "clock_mismatch_examples": examples, "missing_source_directories": missing_dirs,
              "HDF_pixel_payloads_read": 0, "GHI_payloads_read": 0,
              "next": "Verify preprocessing/grid contract and timestamp-only station support; process only frozen train/val."}
    report["daily_inventory_sha256"] = hashlib.sha256((out/"daily_path_inventory.csv").read_bytes()).hexdigest()
    (out/"REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status", "unique_HDF_times", "range_BJT", "multiple_source_times", "archived_BJT_clock_mismatches")}))


if __name__ == "__main__":
    main()
