#!/usr/bin/env python3
"""Append June/July train-val AGRI paths to the frozen Hubei manifest."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

EXPECTED = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def timekey(v):
    s = v.strip().replace("-", "").replace(":", "").replace(" ", "")
    if len(s) == 14: return s[:12]
    if len(s) == 12: return s
    raise ValueError(f"Bad BJT time: {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--old-manifest", type=Path, required=True)
    p.add_argument("--supplement-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    split_sha = sha(a.split)
    if split_sha != EXPECTED: raise ValueError("Frozen split checksum mismatch")
    with a.split.open(encoding="utf-8-sig", newline="") as f:
        date_split = {r["date_BJT"].replace("-", ""): r["split"] for r in csv.DictReader(f)}
    rows = []
    seen = set()
    source_count = Counter()
    for path, is_new in ((a.old_manifest, False), (a.supplement_manifest, True)):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                t = timekey(r["time_bjt"])
                split = r["split"]
                if split not in ("train", "val") or date_split.get(t[:8]) != split:
                    raise ValueError(f"Non-train/val or split mismatch: {t} {split}")
                if t in seen: raise ValueError(f"Duplicate AGRI timestamp: {t}")
                seen.add(t)
                agri_path = r["path_agri"]
                if not agri_path: raise ValueError(f"Missing AGRI path: {t}")
                if is_new:
                    source_count[r.get("source_kind", "unknown")] += 1
                rows.append((t, split, agri_path))
    rows.sort()
    counts = Counter(split for _, split, _ in rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(("time_bjt", "split", "path_agri")); w.writerows(rows)
    report = {
        "state": "HUBEI_EXTENDED_FROZEN_TRAINVAL_AGRI_PATH_MANIFEST_TEST_CLOSED",
        "split_sha256": split_sha, "rows_by_split": dict(counts), "rows_total": len(rows),
        "old_manifest_rows": sum(1 for _ in csv.DictReader(a.old_manifest.open(encoding="utf-8-sig", newline=""))),
        "supplement_rows": sum(source_count.values()), "supplement_source_counts": dict(source_count),
        "test_path_rows": 0, "test_payloads_read": 0,
        "test_agri_payloads_read": 0, "test_ghi_values_read": 0,
        "manifest_sha256": sha(a.output),
    }
    a.output.with_name("REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
