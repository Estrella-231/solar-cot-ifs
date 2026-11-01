"""Audit saved train/validation labels against the independent time-only support audit."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def quantile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    labels_path = Path(a.labels)
    support_path = Path(a.support)
    assignment_path = Path(a.assignment)
    assigned = {}
    with assignment_path.open(encoding="utf-8-sig", newline="") as f:
        assigned = {r["date_BJT"]: r["split"] for r in csv.DictReader(f)}
    targets = {}
    with support_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            targets[(r["station"], r["target_time_BJT"])] = int(r["complete15"])
    stats = {}
    reasons = Counter()
    masks = Counter()
    seen = set()
    expected = Counter()
    with labels_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            station, ts, split = r["station"], r["target_time_BJT"], r["split"]
            key = (station, ts)
            if key in seen:
                raise ValueError(f"Duplicate label key {key}")
            seen.add(key)
            dt = datetime.fromisoformat(ts)
            if assigned.get(dt.date().isoformat()) != split or split not in {"train", "val"}:
                raise ValueError(f"Unexpected split/date or test row: {key} {split}")
            if key not in targets:
                raise ValueError(f"Target absent from time-support manifest: {key}")
            mask = int(r["mask"])
            support = targets[key]
            if mask != support:
                raise ValueError(f"Label-mask/time-support mismatch: {key} label={mask} support={support}")
            reasons[(station, split, r["reason"])] += 1
            masks[(station, split, mask)] += 1
            expected[station] += 1
            if mask:
                x = float(r["ghi15_raw"])
                if not math.isfinite(x) or x < 0:
                    raise ValueError(f"Invalid valid label {key}={x}")
                stats.setdefault((station, split), []).append(x)
            elif r["ghi15_raw"]:
                raise ValueError(f"Masked label has non-empty value: {key}")
    report = {"status": "TRAIN_VAL_LABEL_AUDIT_PASSED_TEST_CLOSED",
              "test_values_read": 0, "label_sha256": sha(labels_path),
              "time_support_sha256": sha(support_path),
              "split_assignment_sha256": sha(assignment_path),
              "row_count": len(seen), "rows_by_station": dict(expected),
              "mask_matches_time_only_support_for_every_key": True,
              "no_test_or_buffer_keys": True,
              "counts_by_station_split_mask": {
                  s: {sp: {str(m): masks[(s, sp, m)] for m in (0, 1)} for sp in ("train", "val")}
                  for s in sorted(expected)},
              "invalid_reason_counts_by_station_split": {
                  s: {sp: {reason: count for (ss, ssp, reason), count in reasons.items() if ss == s and ssp == sp and reason != "complete"}
                      for sp in ("train", "val")}
                  for s in sorted(expected)},
              "valid_value_distribution": {f"{s}/{sp}": {
                  "n": len(v), "min": min(v), "p50": quantile(v, .5), "p95": quantile(v, .95),
                  "p99": quantile(v, .99), "max": max(v), "zero_n": sum(x == 0 for x in v)}
                  for (s, sp), v in sorted(stats.items())},
              "unit": "provisional legacy W/m2 convention; instrument metadata unresolved",
              "daylight_mask": "not applied", "normalization": "not fit"}
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "row_count", "counts_by_station_split_mask", "valid_value_distribution")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
