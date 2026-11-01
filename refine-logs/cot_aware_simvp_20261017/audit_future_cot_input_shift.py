#!/usr/bin/env python3
"""Audit train/validation distribution shift in predicted-COT GHI inputs.

This is an input-distribution diagnostic only. It does not estimate COT
forecast accuracy and must not be interpreted as evidence of a GHI gain.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def month_of(utc: str) -> str:
    return datetime.strptime(utc, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m")


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {k: float("nan") for k in ("mean", "std", "median", "p90", "p99", "frac_le_1", "frac_ge_10", "frac_ge_30")}
    return {
        "mean": float(x.mean()), "std": float(x.std()),
        "median": float(np.quantile(x, .5)), "p90": float(np.quantile(x, .9)),
        "p99": float(np.quantile(x, .99)), "frac_le_1": float(np.mean(x <= 1)),
        "frac_ge_10": float(np.mean(x >= 10)), "frac_ge_30": float(np.mean(x >= 30)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk-rows", type=int, default=8192)
    a = ap.parse_args()
    manifest = json.loads((a.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("refuse incomplete pack")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:
        raise ValueError("pack provenance gate failed")

    row_counts: dict[str, int] = {}
    sums: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)
    textures: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)
    changes: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)

    for split in ("train", "val"):
        d = a.pack / split
        cot = np.load(d / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
        geom = np.load(d / "geometry.npy", mmap_mode="r", allow_pickle=False)
        if cot.ndim != 5 or cot.shape[1:] != (3, 1, 3, 3):
            raise ValueError(f"unexpected COT array shape {cot.shape}")
        if geom.ndim != 5 or geom.shape[1:] != (3, 2, 3, 3):
            raise ValueError(f"unexpected geometry array shape {geom.shape}")
        total_rows = 0
        with (d / "keys.csv").open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            while True:
                rows = list(itertools.islice(reader, a.chunk_rows))
                if not rows:
                    break
                lo, hi = total_rows, total_rows + len(rows)
                if hi > len(cot):
                    raise ValueError(f"keys.csv exceeds arrays in {split}")
                total_rows = hi
            # The final context slot is the COT field at this target time.
                c = np.expm1(np.asarray(cot[lo:hi, 2, 0], dtype=np.float32))
                prev = np.expm1(np.asarray(cot[lo:hi, 1, 0], dtype=np.float32))
                # cos(SOZ) is geometry channel 0 at target time; broad bands avoid
                # claiming that this is a cloud/no-cloud label.
                cosz = np.asarray(geom[lo:hi, 2, 0], dtype=np.float32).mean(axis=(1, 2))
                groups: dict[tuple[str, str, str, int, str], list[int]] = defaultdict(list)
                for local_i, r in enumerate(rows):
                    band = "night" if cosz[local_i] <= 0 else ("low_sun" if cosz[local_i] < .3 else "sun")
                    key = (split, r["station"], month_of(r["target_time_utc"]), int(r["lead_minutes"]), band)
                    groups[key].append(local_i)
                for key, local_rows in groups.items():
                    ix = np.asarray(local_rows, dtype=np.int64)
                    field = c[ix]
                    prior = prev[ix]
                    sums[key].append(field.reshape(-1).astype(np.float32, copy=True))
                    spatial_std = field.std(axis=(1, 2))
                    spatial_gradient = (
                        np.abs(np.diff(field, axis=1)).mean(axis=(1, 2))
                        + np.abs(np.diff(field, axis=2)).mean(axis=(1, 2))
                    ) / 2
                    textures[key].append(np.stack((spatial_std, spatial_gradient), axis=1).astype(np.float32))
                    changes[key].append(np.abs(field - prior).mean(axis=(1, 2)).astype(np.float32))
        if total_rows != len(cot) or total_rows != len(geom):
            raise ValueError(f"row/array mismatch in {split}: {total_rows}, {len(cot)}, {len(geom)}")
        row_counts[split] = total_rows

    out_rows = []
    for key in sorted(sums):
        split, station, month, lead, sun_band = key
        vals = np.concatenate(sums[key])
        stat = summarize(vals)
        tex = np.concatenate(textures[key], axis=0)
        delta = np.concatenate(changes[key], axis=0)
        out_rows.append({
            "split": split, "station": station, "month": month,
            "lead_minutes": lead, "solar_band": sun_band,
            "n_samples": int(len(textures[key])), "n_pixels": int(vals.size),
            **stat, "mean_spatial_std": float(tex[:, 0].mean()),
            "mean_abs_spatial_gradient": float(tex[:, 1].mean()),
            "mean_abs_15min_change": float(delta.mean()),
        })

    a.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = a.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]) if out_rows else ["split"])
        w.writeheader(); w.writerows(out_rows)
    payload = {
        "state": "COMPLETE_PREDICTED_COT_INPUT_DISTRIBUTION_DIAGNOSTIC",
        "interpretation_boundary": "train/validation covariate-shift diagnostic only; not COT accuracy or GHI benefit",
        "bank_train_forecasts_in_sample": manifest.get("in_sample_train_forecasts"),
        "pack_manifest_state": manifest.get("state"),
        "test_payload_opened": False, "buffer_payload_opened": False,
        "rows": row_counts,
        "csv": csv_path.name, "groups": len(out_rows),
        "metrics": ["predicted COT quantiles/fractions", "within-patch spatial texture", "mean absolute context-to-target 15-min COT change"],
    }
    a.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
