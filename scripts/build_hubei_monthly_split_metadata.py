"""Metadata-only Hubei monthly date map, shared by R/S/GHI."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random


def build(inventory, prior, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    counts, cpp_counts = {}, {}
    with open(inventory, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["date_BJT"])
            if int(row["agri_frames"]) > 0:
                counts[d] = int(row["agri_frames"])
                cpp_counts[d] = int(row["cpp_frames"])
    if not counts:
        raise ValueError("No AGRI frames")
    pinned = {}
    with open(prior, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pinned[date.fromisoformat(row["date_BJT"])] = row["split"]
    months = defaultdict(list)
    for d in sorted(counts):
        months[d.strftime("%Y-%m")].append(d)
    assignment = {d: "train" for d in counts}
    blocks = []
    touched = {date(2025, 8, 1)}
    pinned_months = {d.strftime("%Y-%m") for d in pinned}
    for month, days in sorted(months.items()):
        eligible = [d for d in days if counts[d] >= 48]
        if month in pinned_months:
            for d in days:
                if d in pinned:
                    assignment[d] = pinned[d]
            blocks.append({"month": month, "status": "prior_map_preserved",
                           "reason": "Previously read train/val GHI must not be reassigned to test"})
            continue
        candidates = [d for d in eligible if d + timedelta(days=1) in eligible]
        pairs = [(a, b) for a in candidates for b in candidates
                 if abs((a-b).days) >= 6 and not ({b, b+timedelta(days=1)} & touched)]
        if len(eligible) < 20 or not pairs:
            blocks.append({"month": month, "status": "sparse_month_train_only",
                           "eligible_days": len(eligible)})
            continue
        seed = int(hashlib.sha256(f"42:{month}".encode()).hexdigest()[:16], 16)
        val_start, test_start = random.Random(seed).choice(pairs)
        for split, start in (("val", val_start), ("test", test_start)):
            for k in (0, 1):
                d = start + timedelta(days=k)
                if assignment[d] != "train":
                    raise ValueError(f"Cross-month holdout collision: {d}")
                assignment[d] = split
            for k in (-1, 2):
                d = start + timedelta(days=k)
                if d in assignment:
                    if assignment[d] not in ("train", "buffer"):
                        raise ValueError(f"Cross-month buffer collision: {d}")
                    assignment[d] = "buffer"
            blocks.append({"month": month, "split": split, "status": "selected",
                           "start": str(start), "end": str(start+timedelta(days=1))})
    # Preserve prior calendar buffers even when a neighboring month is newly added.
    for d, split in pinned.items():
        if d in assignment:
            if assignment[d] not in ("train", split):
                raise ValueError(f"Prior-map conflict: {d}")
            assignment[d] = split
    assert all(assignment.get(d) != "test" for d in touched)
    assert all(assignment[d] == s for d, s in pinned.items() if d in assignment)
    # Every val/test contiguous block retains a buffer at both external edges.
    for d, s in assignment.items():
        if s not in ("val", "test"):
            continue
        for neighbor in (d-timedelta(days=1), d+timedelta(days=1)):
            if neighbor in assignment and assignment[neighbor] != s:
                assert assignment[neighbor] == "buffer", (d, neighbor)
    out.mkdir(parents=True)
    with (out/"day_assignment.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_BJT", "split", "agri_frames", "cpp_frames"])
        for d in sorted(assignment):
            w.writerow([d, assignment[d], counts[d], cpp_counts[d]])
    report = {"status": "CANDIDATE_PENDING_TIMESTAMP_SUPPORT_AND_SOURCE_ACCEPTANCE",
              "split_id": "hubei_monthly_block_2d_buffer1d_seed42_202508_202607_candidate_v1",
              "date_range_BJT": [str(min(counts)), str(max(counts))],
              "counts": dict(Counter(assignment.values())), "blocks": blocks,
              "seed": 42, "test_ineligible_touched_dates": [str(d) for d in sorted(touched)],
              "prior_map_preserved": str(prior), "shared_R_S_H_date_map": True,
              "selection_uses_GHI_CPP_values": False, "test_payloads_read_this_run": 0,
              "date_map_sha256": hashlib.sha256((out/"day_assignment.csv").read_bytes()).hexdigest(),
              "inventory_sha256": hashlib.sha256(Path(inventory).read_bytes()).hexdigest()}
    (out/"REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--prior-date-map", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    build(a.inventory, a.prior_date_map, a.output)
