"""1000 train/val frames in real contiguous runs, 250 per season; preserve gaps."""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
times = set()
with a.manifest.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["split"] in ("train", "val"):
            times.update(datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
                         for rel in row["data_relpaths"].split("|"))
ordered = sorted(times)
segments = {s: [] for s in range(4)}
run = []
for t in ordered:
    if run and (t-run[-1] != timedelta(minutes=15) or (t.month % 12)//3 != (run[-1].month % 12)//3):
        segments[(run[0].month % 12)//3].append(run)
        run = []
    run.append(t)
if run:
    segments[(run[0].month % 12)//3].append(run)
rng = random.Random(42)
selected, blocks = [], []
for season in range(4):
    candidates = [r for r in segments[season] if len(r) >= 24]
    rng.shuffle(candidates)
    remaining = 250
    for run in candidates:
        length = min(len(run), remaining)
        first = rng.randrange(len(run)-length+1)
        block = run[first:first+length]
        selected.extend(block)
        blocks.append({"season": season, "frames": len(block),
                       "first_utc": block[0].isoformat(), "last_utc": block[-1].isoformat()})
        remaining -= length
        if not remaining:
            break
    if remaining:
        raise RuntimeError(f"insufficient genuine contiguous sequences in season {season}")
if len(set(selected)) != 1000:
    raise RuntimeError("overlapping blocks")
a.output.mkdir(parents=True, exist_ok=False)
(a.output / "times1000.json").write_text(json.dumps([t.isoformat() for t in sorted(selected)], indent=2))
(a.output / "identity.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
    "seed": 42, "time_source": "UTC data_relpaths filename", "test_used": False, "blocks": blocks}, indent=2))
