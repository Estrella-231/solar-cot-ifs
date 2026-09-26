"""Freeze season/time-stratified CPP engineering samples without inspecting errors.

Manifest data_relpaths use UTC filenames; BJT_start is not parsed as UTC.
Day/night strata are clock proxies and must be checked against solar geometry.
"""
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    splits = {}
    with a.manifest.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] not in ("train", "val"):
                continue
            for rel in row["data_relpaths"].split("|"):
                stamp = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
                splits.setdefault(stamp, set()).add(row["split"])
    rng = random.Random(42)
    buckets = {}
    for t in sorted(splits):
        season = (t.month % 12) // 3
        bjt_hour = (t.hour + 8) % 24
        clock = "day_clock" if 8 <= bjt_hour < 17 else "night_or_twilight_clock"
        buckets.setdefault((season, clock), []).append(t)
    selected, references = [], []
    for key, candidates in sorted(buckets.items()):
        if len(candidates) < 6:
            raise ValueError(f"insufficient stratum {key}")
        sample = sorted(rng.sample(candidates, 6))
        selected.extend(sample)
        references.extend(sample[:3])
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, items):
        (a.output / name).write_text(json.dumps(items, indent=2) + "\n")
    save("times48.json", [t.isoformat() for t in sorted(selected)])
    save("references24.json", [t.isoformat() for t in sorted(references)])
    save("sample_identity.json", {
        "seed": 42, "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "time_source": "UTC data_relpaths filename", "test_used": False,
        "strata": "four meteorological seasons x BJT clock proxy; solar/cloud QC pending",
        "unique_frames": len(splits),
        "frames": [{"utc": t.isoformat(), "splits": sorted(splits[t])} for t in sorted(selected)]})


if __name__ == "__main__":
    main()
