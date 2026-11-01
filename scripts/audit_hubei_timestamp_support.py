"""Read only Excel headers/time column; never access irradiance cells."""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import openpyxl
from hubei_ghi15_contract import minute_index

STATIONS = {"牛首供电所": "Niushou", "王寨供电所": "Wangzhai", "米庄供电所": "Mizhuang"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame-manifest", required=True)
    p.add_argument("--extension-manifest", required=True)
    p.add_argument("--xlsx-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    targets = set()
    for manifest in (a.frame_manifest, a.extension_manifest):
        with open(manifest, encoding="utf-8", newline="") as f:
            targets.update(datetime.fromisoformat(r["target_time_BJT"]) for r in csv.DictReader(f))
    start, end = min(targets)-timedelta(minutes=15), max(targets)
    times = {s: set() for s in STATIONS.values()}
    counts, invalid_times = Counter(), Counter()
    source_files = []
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    progress_path = out / "PROGRESS.json"
    progress = {"status": "RUNNING_TIME_COLUMN_ONLY", "target_count": len(targets),
                "target_range_BJT": [str(min(targets)), str(max(targets))],
                "completed_workbooks": [], "completed_sheets": [],
                "time_records_by_station": {}, "GHI_cells_accessed": 0}
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    for path in sorted(Path(a.xlsx_root).glob("天气数据_*.xlsx")):
        # Authoritative monthly/June-July segment files only; no alternate exports.
        stem = path.stem[len("天气数据_"):]
        if not ((len(stem) == 6 and stem.isdigit()) or stem in
                ("2026-06-01-2026-07-06", "2026-07-07-2026-07-14")):
            continue
        print("TIME_COLUMN_ONLY", path.name, flush=True)
        w = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if not set(STATIONS).issubset(w.sheetnames):
                raise ValueError(f"Missing station sheet: {path}")
            for sheet_name, station in STATIONS.items():
                s = w[sheet_name]
                header = next(s.iter_rows(min_row=1, max_row=1, values_only=True))
                column = header.index("时间") + 1
                assert "总辐射1瞬时值" in header
                sheet_records = 0
                for (value,) in s.iter_rows(min_row=2, min_col=column, max_col=column, values_only=True):
                    try:
                        t = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
                    except (TypeError, ValueError):
                        invalid_times[station] += 1
                        continue
                    if start < t <= end:
                        times[station].add(t)
                        counts[station] += 1
                        sheet_records += 1
                progress["completed_sheets"].append(f"{path.name}:{sheet_name}")
                progress["time_records_by_station"] = dict(counts)
                progress["last_sheet_records_in_target_range"] = sheet_records
                progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        finally:
            w.close()
        source_files.append(str(path))
        progress["completed_workbooks"] = list(source_files)
        progress["time_records_by_station"] = dict(counts)
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    times = {s: sorted(ts) for s, ts in times.items()}
    daily, totals = Counter(), Counter()
    with (out/"target_timestamp_support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station", "target_time_BJT", "unique_time_records", "minute_bins", "complete15"])
        for station, ts in times.items():
            for target in sorted(targets):
                lo = bisect_right(ts, target-timedelta(minutes=15))
                hi = bisect_right(ts, target)
                bins = {minute_index(t, target) for t in ts[lo:hi]}
                complete = len(bins) == 15
                w.writerow([station, target, hi-lo, len(bins), int(complete)])
                if complete:
                    daily[(target.date().isoformat(), station)] += 1
                    totals[station] += 1
    with (out/"daily_timestamp_support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["date_BJT", "station", "complete15_targets"])
        for d in sorted({t.date().isoformat() for t in targets}):
            for s in STATIONS.values():
                w.writerow([d, s, daily[(d, s)]])
    report = {"status": "TIMESTAMP_SUPPORT_ONLY_NOT_VALUE_QA_OR_DAYLIGHT_MASK",
              "target_count": len(targets), "range_BJT": [str(min(targets)), str(max(targets))],
              "complete15_targets_by_station": dict(totals), "source_files": source_files,
              "time_records_by_station": dict(counts),
              "duplicate_timestamps_by_station": {s:counts[s]-len(ts) for s,ts in times.items()},
              "invalid_time_records_by_station": dict(invalid_times),
              "window": "(target-15min,target], 15 right-closed minute bins",
              "time_column_only": True, "GHI_cells_accessed": 0, "test_GHI_cells_accessed": 0,
              "daylight_filtered": False, "value_quality_checked": False,
              "support_sha256": hashlib.sha256((out/"target_timestamp_support.csv").read_bytes()).hexdigest()}
    (out/"REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress["status"] = "COMPLETE_TIME_COLUMN_ONLY"
    progress["report"] = "REPORT.json"
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status", "target_count", "complete15_targets_by_station", "GHI_cells_accessed")}))


if __name__ == "__main__":
    main()
