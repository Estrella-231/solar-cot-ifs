"""Summarize metadata-only station window support by frozen candidate split."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-support", required=True)
    p.add_argument("--assignments", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    assignment_rows = read_csv(a.assignments)
    date_split = {r["date_BJT"]: r["split"] for r in assignment_rows}
    support_rows = read_csv(a.target_support)
    targets = defaultdict(dict)
    daily = Counter()
    all_splits = set(date_split.values())
    for r in support_rows:
        date = r["target_time_BJT"][:10]
        split = date_split.get(date)
        if split is None:
            continue
        station = r["station"]
        complete = int(r["complete15"])
        targets[r["target_time_BJT"]][station] = complete
        daily[(date, split, station)] += complete

    rows = []
    per_split = {}
    for split in sorted(all_splits):
        split_targets = [t for t in targets if date_split.get(t[:10]) == split]
        by_station = {}
        for station in ("Niushou", "Wangzhai", "Mizhuang"):
            complete = sum(targets[t].get(station, 0) for t in split_targets)
            day_counts = [daily[(d, split, station)] for d, s in date_split.items()
                          if s == split]
            by_station[station] = {
                "complete15_targets": complete,
                "target_slots": len(split_targets),
                "support_fraction": complete / len(split_targets) if split_targets else None,
                "days_with_any_complete_target": sum(v > 0 for v in day_counts),
                "assigned_days": len(day_counts),
                "zero_support_days": sum(v == 0 for v in day_counts),
            }
            rows.append({"split": split, "station": station, **by_station[station]})
        common = sum(all(targets[t].get(s, 0) for s in
                         ("Niushou", "Wangzhai", "Mizhuang")) for t in split_targets)
        per_split[split] = {"stations": by_station,
                            "three_station_common_targets": common,
                            "target_slots": len(split_targets)}

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "split_timestamp_support.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "station", "complete15_targets", "target_slots",
                  "support_fraction", "days_with_any_complete_target", "assigned_days",
                  "zero_support_days"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    report = {
        "status": "METADATA_ONLY_SUPPORT_SUMMARY_NOT_LABEL_QA",
        "split_file": str(Path(a.assignments)),
        "target_support_file": str(Path(a.target_support)),
        "candidate_split_counts": {s: sum(v == s for v in date_split.values())
                                   for s in sorted(all_splits)},
        "by_split": per_split,
        "GHI_values_read": 0,
        "test_GHI_values_read": 0,
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
