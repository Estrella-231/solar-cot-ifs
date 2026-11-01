#!/usr/bin/env python3
"""Compare train/validation predicted-COT bank inputs without opening labels.

This diagnoses covariate shift between in-sample train forecasts and held-out
validation forecasts. It does not measure COT accuracy or downstream GHI gain.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def parse_init_bjt(seq_id: str) -> datetime:
    stamp = seq_id.rsplit("_", 1)[-1]
    return datetime.strptime(stamp, "%Y%m%d%H%M")


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        "mean": float(x.mean()), "std": float(x.std()),
        "median": float(np.quantile(x, .5)), "p90": float(np.quantile(x, .9)),
        "p99": float(np.quantile(x, .99)), "frac_le_1": float(np.mean(x <= 1)),
        "frac_ge_10": float(np.mean(x >= 10)), "frac_ge_30": float(np.mean(x >= 30)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk-sequences", type=int, default=256)
    a = ap.parse_args()
    meta = json.loads((a.bank / "bank_manifest.json").read_text(encoding="utf-8"))
    if meta.get("state") != "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK":
        raise ValueError("forecast bank is incomplete")
    if meta.get("test_payload_opened") is not False or meta.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer exclusion gate failed")
    if meta.get("future_observed_agri_read") is not False or meta.get("cpp_read") is not False:
        raise ValueError("future observed AGRI/CPP exclusion gate failed")
    index_path = a.bank / "index.csv"
    with index_path.open(newline="", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    n = len(index)
    if n != int(meta.get("rows", -1)):
        raise ValueError("index row count differs from manifest")
    bank_idx = np.asarray([int(r["bank_index"]) for r in index], dtype=np.int64)
    if len(np.unique(bank_idx)) != n or bank_idx.min() != 0 or bank_idx.max() != n - 1:
        raise ValueError("bank indices are duplicate or non-contiguous")
    if any(r["split"] not in ("train", "val") for r in index):
        raise ValueError("non-train/validation split found")
    cot = np.load(a.bank / "forecast_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    geom = np.load(a.bank / "forecast_geometry.npy", mmap_mode="r", allow_pickle=False)
    hist = np.load(a.bank / "history_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if cot.shape != (n, 2, 16, 1, 3, 3) or cot.dtype != np.float32:
        raise ValueError(f"unexpected future COT shape/dtype: {cot.shape}/{cot.dtype}")
    if geom.shape != (n, 2, 16, 2, 3, 3) or geom.dtype != np.float32:
        raise ValueError(f"unexpected forecast geometry shape/dtype: {geom.shape}/{geom.dtype}")
    if hist.shape != (n, 2, 2, 1, 3, 3) or hist.dtype != np.float32:
        raise ValueError(f"unexpected history COT shape/dtype: {hist.shape}/{hist.dtype}")

    # Values are accumulated by regime, but arrays remain memory-mapped and
    # samples are processed in bounded chunks.
    values: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)
    textures: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)
    changes: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)
    split_counts = defaultdict(int)
    for lo in range(0, n, a.chunk_sequences):
        hi = min(lo + a.chunk_sequences, n)
        ids = bank_idx[lo:hi]
        c = np.expm1(np.asarray(cot[ids], dtype=np.float32))
        h = np.expm1(np.asarray(hist[ids], dtype=np.float32))
        cosz = np.asarray(geom[ids, :, :, 0], dtype=np.float32).mean(axis=(3, 4))
        groups: dict[tuple[str, str, str, int, str], list[tuple[int, int, int]]] = defaultdict(list)
        for local_seq, row in enumerate(index[lo:hi]):
            init = parse_init_bjt(row["seq_id"])
            split = row["split"]
            split_counts[split] += 2 * 16
            for station_i, station in enumerate(("sili", "zhujia")):
                for lead_i in range(16):
                    lead_minutes = 15 * (lead_i + 1)
                    target_month = (init + timedelta(minutes=lead_minutes)).strftime("%Y-%m")
                    cz = float(cosz[local_seq, station_i, lead_i])
                    solar_band = "night" if cz <= 0 else ("low_sun" if cz < .3 else "sun")
                    key = (split, station, target_month, lead_minutes, solar_band)
                    groups[key].append((local_seq, station_i, lead_i))
        for key, refs in groups.items():
            fields, priors = [], []
            for si, station_i, lead_i in refs:
                fields.append(c[si, station_i, lead_i, 0])
                priors.append(h[si, station_i, 1, 0] if lead_i == 0 else c[si, station_i, lead_i - 1, 0])
            field = np.stack(fields)
            prior = np.stack(priors)
            values[key].append(field.reshape(-1).astype(np.float32, copy=True))
            spatial_std = field.std(axis=(1, 2))
            spatial_gradient = (
                np.abs(np.diff(field, axis=1)).mean(axis=(1, 2))
                + np.abs(np.diff(field, axis=2)).mean(axis=(1, 2))
            ) / 2
            textures[key].append(np.stack((spatial_std, spatial_gradient), axis=1).astype(np.float32))
            changes[key].append(np.abs(field - prior).mean(axis=(1, 2)).astype(np.float32))

    rows = []
    for key in sorted(values):
        split, station, month, lead, solar_band = key
        x = np.concatenate(values[key])
        tex = np.concatenate(textures[key], axis=0)
        delta = np.concatenate(changes[key])
        rows.append({
            "split": split, "station": station, "target_month": month,
            "lead_minutes": lead, "solar_band": solar_band,
            "n_samples": int(len(tex)), "n_patch_pixels": int(x.size),
            **stats(x), "mean_spatial_std": float(tex[:, 0].mean()),
            "mean_abs_spatial_gradient": float(tex[:, 1].mean()),
            "mean_abs_15min_change": float(delta.mean()),
        })
    a.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = a.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "state": "COMPLETE_FORECAST_COT_BANK_INPUT_SHIFT_DIAGNOSTIC",
        "interpretation_boundary": "input-distribution diagnostic only; not COT forecast accuracy or GHI benefit",
        "train_forecasts_in_sample": meta.get("in_sample_train_forecasts"),
        "forecast_source": meta.get("forecast_source"),
        "forecast_cot_source": meta.get("forecast_cot_source"),
        "sequence_rows": n, "input_instances_by_split": dict(split_counts),
        "groups": len(rows), "csv": csv_path.name,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_read": False, "cpp_read": False,
        "strata": "station x target month x lead x broad target-time cosSOZ band",
        "metrics": ["physical predicted-COT distribution", "3x3 spatial texture", "target-to-prior 15-minute change"],
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
