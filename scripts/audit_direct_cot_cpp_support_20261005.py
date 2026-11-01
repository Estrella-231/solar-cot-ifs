"""Build a train/validation-only CPP timestamp support request for the new split.

The source manifest stores FY-4B frame stems in UTC (BJT_start is carried
separately). This script enumerates only the 16 future target frames per
sequence, preserving their canonical +15..+240 minute leads. It never reads
the test split or any image/COT payload.
"""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def season(month: int) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[month]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    occurrences = Counter()
    metadata = {}
    sequences = Counter()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in {"train", "val"}:
                continue
            paths = row["data_relpaths"].split("|")
            if len(paths) != 24:
                raise RuntimeError(f"expected 24 frames, got {len(paths)}: {row['seq_id']}")
            utc0 = datetime.strptime(Path(paths[0]).stem, "%Y%m%d%H%M")
            bjt0 = datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S")
            for index in range(8, 24):
                utc = datetime.strptime(Path(paths[index]).stem, "%Y%m%d%H%M")
                expected = utc0 + timedelta(minutes=15 * index)
                if utc != expected:
                    raise RuntimeError(f"non-canonical UTC sequence: {row['seq_id']} frame {index}")
                bjt = bjt0 + timedelta(minutes=15 * index)
                stamp = utc.strftime("%Y%m%d%H%M")
                key = (split, stamp)
                occurrences[key] += 1
                metadata[key] = {"split": split, "utc": stamp,
                                 "bjt": bjt.strftime("%Y%m%d%H%M"),
                                 "month_bjt": bjt.strftime("%Y-%m"),
                                 "season_bjt": season(bjt.month)}
                sequences[split] += 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    request = args.out_dir / "expected_cpp_targets_train_val.csv"
    with request.open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "utc", "bjt", "month_bjt", "season_bjt", "target_occurrences"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(occurrences):
            writer.writerow({**metadata[key], "target_occurrences": occurrences[key]})
    # For exact temporal support by lead, write a compact occurrence list.
    lead_rows = []
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in {"train", "val"}:
                continue
            paths = row["data_relpaths"].split("|")
            utc0 = datetime.strptime(Path(paths[0]).stem, "%Y%m%d%H%M")
            bjt0 = datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S")
            for index in range(8, 24):
                utc = utc0 + timedelta(minutes=15 * index)
                bjt = bjt0 + timedelta(minutes=15 * index)
                lead_rows.append((split, utc.strftime("%Y%m%d%H%M"),
                                  bjt.strftime("%Y-%m-%d %H:%M"), bjt.strftime("%Y-%m"),
                                  season(bjt.month), (index - 7) * 15))
    lead_path = args.out_dir / "cpp_target_occurrences_by_lead_train_val.csv"
    with lead_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "utc", "target_bjt", "month_bjt", "season_bjt", "lead_min"])
        writer.writerows(lead_rows)
    print(f"targets_unique={len(occurrences)} targets_by_sequence={len(lead_rows)}")
    print(f"train_unique={sum(s == 'train' for s, _ in occurrences)} val_unique={sum(s == 'val' for s, _ in occurrences)}")
    print(f"train_occurrences={sum(r[0] == 'train' for r in lead_rows)} val_occurrences={sum(r[0] == 'val' for r in lead_rows)}")
    print(f"request={request}")
    print(f"lead_occurrences={lead_path}")


if __name__ == "__main__":
    main()
