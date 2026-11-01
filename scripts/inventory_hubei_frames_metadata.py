"""Inventory raw AGRI/CPP file paths only; never open array payloads."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
import re
from pathlib import Path


def inventory(root, clock, start, end):
    frames = {}
    duplicates = Counter()
    for path in sorted(Path(root).rglob("*.npy")):
        match = re.search(r"(?<!\d)(\d{14}|\d{12})(?!\d)", path.stem)
        if not match:
            continue
        stamp = match.group(1)
        timestamp = datetime.strptime(stamp, "%Y%m%d%H%M%S" if len(stamp) == 14 else "%Y%m%d%H%M")
        if clock == "UTC":
            timestamp += timedelta(hours=8)
        if not start <= timestamp.date().isoformat() <= end:
            continue
        key = timestamp.isoformat(sep=" ")
        if key in frames:
            duplicates[timestamp.date().isoformat()] += 1
            # Inventory only. No claim of byte equality or source acceptance.
            if len(path.parts) <= len(Path(frames[key]).parts):
                continue
        frames[key] = str(path)
    return frames, duplicates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agri-root", required=True)
    p.add_argument("--cpp-root", required=True)
    p.add_argument("--cpp-clock", required=True, choices=("UTC", "BJT"))
    p.add_argument("--start", default="2025-08-01")
    p.add_argument("--end", default="2026-07-14")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    agri, ad = inventory(a.agri_root, "UTC", a.start, a.end)
    cpp, cd = inventory(a.cpp_root, a.cpp_clock, a.start, a.end)
    ac, cc = Counter(t[:10] for t in agri), Counter(t[:10] for t in cpp)
    with (out / "daily_path_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date_BJT", "agri_frames", "cpp_frames", "agri_duplicate_paths", "cpp_duplicate_paths"])
        for day in sorted(set(ac) | set(cc)):
            w.writerow([day, ac[day], cc[day], ad[day], cd[day]])
    with (out / "agri_frame_paths.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target_time_BJT", "agri_path"])
        w.writerows(sorted(agri.items()))
    with (out / "cpp_frame_paths.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target_time_BJT", "cpp_path"])
        w.writerows(sorted(cpp.items()))
    report = {"status": "METADATA_INVENTORY_ONLY_NOT_FROZEN_SPLIT",
              "AGRI_clock": "UTC -> BJT +8h", "CPP_clock": a.cpp_clock,
              "range_requested_BJT": [a.start, a.end],
              "agri_unique_frames": len(agri), "cpp_unique_frames": len(cpp),
              "agri_range_BJT": [min(agri), max(agri)] if agri else [],
              "cpp_range_BJT": [min(cpp), max(cpp)] if cpp else [],
              "agri_duplicate_paths": sum(ad.values()),
              "cpp_duplicate_paths": sum(cd.values()), "array_payloads_read": 0,
              "GHI_payloads_read": 0,
              "next": "Confirm CPP filename clock, timestamp-only GHI support and duplicates before freezing common R/H date map."}
    report["inventory_sha256"] = hashlib.sha256((out / "daily_path_inventory.csv").read_bytes()).hexdigest()
    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
