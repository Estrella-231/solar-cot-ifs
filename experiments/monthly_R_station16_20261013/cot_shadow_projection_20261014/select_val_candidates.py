#!/usr/bin/env python3
"""Select outcome-defined COT-help/hurt validation examples for shadow audit.

This is a validation-only diagnostic over saved A/AC predictions. It never
reads test rows and does not claim that AC-A is a causal COT-only effect.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader((args.pack / "rows.csv").open(encoding="utf-8-sig")))
    if any(r.get("split") == "test" for r in rows):
        raise ValueError("test rows are forbidden in this validation diagnostic")
    cot = np.load(args.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    output = []
    day_groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for station in ("sili", "zhujia"):
        indexed_rows = [(i, r) for i, r in enumerate(rows) if r["split"] == "val" and r["station"] == station]
        station_rows = [r for _, r in indexed_rows]
        predictions = {
            method: [
                np.load(args.predictions / f"ghi_heads_new_r_seed{seed}_v1" / station / f"{method}_val_prediction_ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
                for seed in (42, 43, 44)
            ]
            for method in ("A", "AC")
        }
        if any(len(a) != len(station_rows) for model in predictions.values() for a in model):
            raise ValueError(f"prediction length mismatch for {station}")
        for i, (pack_index, row) in enumerate(indexed_rows):
            y = float(row["ghi"])
            clear = float(row["clear"])
            c = float(np.expm1(cot[pack_index, 0, 1, 1]))
            a = np.asarray([x[i] for x in predictions["A"]], dtype=np.float64)
            ac = np.asarray([x[i] for x in predictions["AC"]], dtype=np.float64)
            delta_sse = np.square(ac - y) - np.square(a - y)
            helpful_seeds = int(np.sum(np.abs(ac - y) < np.abs(a - y)))
            classification = "AC_better" if helpful_seeds >= 2 else "AC_worse" if helpful_seeds <= 1 else "mixed"
            item = {
                "station": station,
                "target_time_utc": row["target_time_utc"],
                "target_date_utc": row["target_time_utc"][:10],
                "ghi_wm2": y,
                "clear_wm2": clear,
                "center_cot": c,
                "A_seed_mean_wm2": float(a.mean()),
                "AC_seed_mean_wm2": float(ac.mean()),
                "delta_sse_mean_wm4": float(delta_sse.mean()),
                "helpful_seed_count": helpful_seeds,
                "outcome_group": classification,
            }
            output.append(item)
            if clear >= 100 and c >= 3:
                day_groups[(station, item["target_date_utc"])].append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else []
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    print("station,date,n,mean_delta_SSE,AC_better,AC_worse")
    for (station, date), values in sorted(day_groups.items()):
        if len(values) < 5:
            continue
        print(
            station,
            date,
            len(values),
            f"{np.mean([v['delta_sse_mean_wm4'] for v in values]):.3f}",
            sum(v["outcome_group"] == "AC_better" for v in values),
            sum(v["outcome_group"] == "AC_worse" for v in values),
            sep=",",
        )
    print(f"saved={args.output} rows={len(output)} test_rows=0")


if __name__ == "__main__":
    main()
