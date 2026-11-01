"""Index all frozen train/val Hubei AGRI NPZ paths without reading payloads."""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


STAMP = re.compile(r"^(\d{12})")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agri-root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--r-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, split_path, r_path, out = map(Path, (a.agri_root, a.split, a.r_manifest, a.output))
    with split_path.open(encoding="utf-8-sig", newline="") as f:
        assignments = {r["date_BJT"]: r["split"] for r in csv.DictReader(f)}
    if not assignments or not {"train", "val", "test", "buffer"}.issubset(set(assignments.values())):
        raise ValueError("Unexpected frozen date split")
    split_sha = sha(split_path)
    expected_split_sha = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
    if split_sha != expected_split_sha:
        raise ValueError(f"Frozen split hash mismatch: {split_sha}")
    timestamp_paths = {}
    count_by_split = Counter()
    all_npz = 0
    for current, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".npz"):
                continue
            all_npz += 1
            match = STAMP.match(name)
            if not match:
                continue
            stamp = match.group(1)
            date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
            split = assignments.get(date)
            if split is None:
                continue
            count_by_split[split] += 1
            if split not in ("train", "val"):
                continue
            if stamp in timestamp_paths:
                raise ValueError(f"Duplicate AGRI timestamp {stamp}: {timestamp_paths[stamp]} / {current}/{name}")
            timestamp_paths[stamp] = str(Path(current) / name)

    with r_path.open(encoding="utf-8-sig", newline="") as f:
        r_rows = list(csv.DictReader(f))
    r_times = {r["time_bjt"] for r in r_rows if r["split"] in ("train", "val")}
    missing_r = sorted(r_times - set(timestamp_paths))
    if missing_r:
        raise ValueError(f"CPP-supported R rows missing in full AGRI index: {missing_r[:10]}")
    rows = sorted(timestamp_paths.items())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_bjt", "split", "path_agri"])
        for stamp, path in rows:
            date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
            w.writerow([stamp, assignments[date], path])
    report = {
        "state": "HUBEI_FULL_AGRI_TRAINVAL_PATH_INDEXED_METADATA_ONLY_TEST_CLOSED",
        "agri_root": str(root), "total_npz_paths_scanned": all_npz,
        "paths_by_frozen_split_metadata_only": {s: int(count_by_split[s]) for s in ("train", "val", "test", "buffer")},
        "trainval_paths_written": len(rows),
        "train_paths_written": sum(assignments[f"{t[:4]}-{t[4:6]}-{t[6:8]}"] == "train" for t, _ in rows),
        "val_paths_written": sum(assignments[f"{t[:4]}-{t[4:6]}-{t[6:8]}"] == "val" for t, _ in rows),
        "r_cpp_supported_rows_checked_as_subset": len(r_times),
        "r_cpp_supported_rows_missing": 0,
        "test_path_metadata_only": True, "test_payloads_read": 0,
        "split_sha256": split_sha, "r_manifest_sha256": sha(r_path),
        "manifest_sha256": sha(out),
    }
    (out.parent / "REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
