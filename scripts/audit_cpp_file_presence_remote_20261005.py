"""Metadata-only CPP product presence audit; does not open NetCDF payloads."""
import argparse
import csv
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--cpp-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    with a.targets.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_day = {}
    out = []
    for row in rows:
        stamp = row["utc"]
        day = stamp[:8]
        if day not in by_day:
            folder = a.cpp_root / day[:4] / day
            try:
                names = set(os.listdir(folder))
            except FileNotFoundError:
                names = set()
            by_day[day] = (folder, names)
        folder, names = by_day[day]
        filename = f"FY4B_AGRI_{stamp}00.nc"
        path = folder / filename
        present = filename in names and path.is_file()
        size = path.stat().st_size if present else 0
        out.append({**row, "cpp_present": int(present), "cpp_size_bytes": size,
                    "cpp_path": str(path) if present else ""})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for split in ("train", "val"):
        block = [x for x in out if x["split"] == split]
        present = sum(x["cpp_present"] for x in block)
        occ = sum(int(x["target_occurrences"]) * int(x["cpp_present"]) for x in block)
        all_occ = sum(int(x["target_occurrences"]) for x in block)
        print(f"{split}: unique={present}/{len(block)} ({present/max(1,len(block)):.6f}); "
              f"sequence_target_occurrences={occ}/{all_occ} ({occ/max(1,all_occ):.6f})")
    print(f"unique_days={len(by_day)} output={a.output}")


if __name__ == "__main__":
    main()
