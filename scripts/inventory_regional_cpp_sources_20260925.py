"""Inventory raw CPP availability for unique train/validation Hunan AGRI frames.

Reads only manifest metadata and filesystem existence. No CPP or test payloads.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cpp-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite source inventory")
    frames = defaultdict(set)
    sequences = defaultdict(int)
    with a.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in ("train", "val"):
                continue
            sequences[split] += 1
            relpaths = row["data_relpaths"].split("|")
            if len(relpaths) != 24:
                raise RuntimeError("expected eight history and sixteen future images")
            for rel in relpaths:
                stem = Path(rel).stem
                if len(stem) != 12 or not stem.isdigit():
                    raise RuntimeError(f"bad UTC AGRI frame name: {rel}")
                frames[split].add(stem)
    result = {"state": "REGIONAL_CPP_SOURCE_INVENTORY", "test_used": False,
              "scope": "unique AGRI frame UTC names in train/val manifest only",
              "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              "cpp_root": str(a.cpp_root), "splits": {}}
    for split in ("train", "val"):
        missing = []
        monthly = defaultdict(lambda: {"unique_frames": 0, "cpp_present": 0, "cpp_missing": 0})
        for stem in sorted(frames[split]):
            path = a.cpp_root / stem[:4] / stem[:8] / f"FY4B_AGRI_{stem}00.nc"
            month = monthly[stem[:6]]
            month["unique_frames"] += 1
            if not path.is_file():
                missing.append(stem)
                month["cpp_missing"] += 1
            else:
                month["cpp_present"] += 1
        result["splits"][split] = {"sequences": sequences[split],
                                   "unique_frames": len(frames[split]),
                                   "cpp_present": len(frames[split])-len(missing),
                                   "cpp_missing": len(missing),
                                   "first_missing_utc": missing[:12],
                                   "by_utc_month": dict(sorted(monthly.items()))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["splits"]))


if __name__ == "__main__":
    main()
