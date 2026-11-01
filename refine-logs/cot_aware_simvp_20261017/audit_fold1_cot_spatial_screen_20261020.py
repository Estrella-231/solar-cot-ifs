#!/usr/bin/env python3
"""Fold-1 3x3 station-neighborhood COT texture screen.

Reads predicted COT and the same keys' two observed-history COT frames from
the same frozen R. Does not read future AGRI, CPP, GHI, test, or buffer data.
This is a descriptive texture diagnostic, not a COT skill or GHI-gain score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_index(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(x: np.ndarray) -> dict:
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "n": int(z.size),
        "mean": float(z.mean()),
        "median": float(np.median(z)),
        "p95": float(np.quantile(z, .95)),
    }


def date_block_ratio(forecast: np.ndarray, history: np.ndarray, days: list[str], seed: int = 42) -> dict:
    """Bootstrap equally weighted BJT init dates, preserving within-day pairs."""
    unique_days = sorted(set(days))
    fday = np.asarray([forecast[np.asarray(days) == d].mean() for d in unique_days])
    hday = np.asarray([history[np.asarray(days) == d].mean() for d in unique_days])
    estimate = float(fday.mean() / max(hday.mean(), 1e-12))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_days), size=(10000, len(unique_days)))
    ratios = fday[draws].mean(axis=1) / np.maximum(hday[draws].mean(axis=1), 1e-12)
    return {
        "n_bjt_dates": len(unique_days),
        "equal_date_ratio_of_means": estimate,
        "bootstrap_95pct_ci": [float(x) for x in np.quantile(ratios, [0.025, 0.975])],
        "bootstrap_unit": "BJT init date",
        "bootstrap_replicates": 10000,
        "seed": seed,
    }


def edge_tv(a: np.ndarray) -> np.ndarray:
    """Mean absolute 4-neighbor edge difference per 3x3 map."""
    return (np.abs(a[..., :, 1:] - a[..., :, :-1]).sum(axis=(-2, -1)) +
            np.abs(a[..., 1:, :] - a[..., :-1, :]).sum(axis=(-2, -1))) / 12.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", type=Path, required=True)
    p.add_argument("--history-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    fm = json.loads((a.forecast_bank / "bank_manifest.json").read_text())
    hm = json.loads((a.history_bank / "bank_manifest.json").read_text())
    assert fm["state"] == "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS" and fm["fold"] == 1
    assert fm["future_observed_agri_read"] is False and fm["cpp_read"] is False
    assert fm["validation_test_payload_opened"] is False and fm["buffer_payload_opened"] is False
    assert hm["state"] == "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK"
    assert hm["future_observed_agri_read"] is False and hm["cpp_read"] is False
    assert hm["buffer_payload_opened"] is False
    assert fm["r_checkpoint_sha256"] == hm["r_checkpoint_sha256"]
    fi, hi = read_index(a.forecast_bank / "index.csv"), read_index(a.history_bank / "index.csv")
    hrows = {r["seq_id"]: int(r["bank_index"]) for r in hi if r["split"] == "train"}
    ids = [r["seq_id"] for r in fi]
    if len({r["seq_id"] for r in fi}) != len(fi):
        raise ValueError("duplicate forecast seq_id")
    if any(s not in hrows for s in ids):
        raise ValueError("history bank missing forecast key")
    days = [s.rsplit("_", 1)[-1][:8] for s in ids]
    if any(len(d) != 8 or not d.isdigit() for d in days):
        raise ValueError("cannot derive BJT init date from sequence IDs")
    findices = np.asarray([int(r["bank_index"]) for r in fi], dtype=np.int64)
    hindices = np.asarray([hrows[s] for s in ids], dtype=np.int64)
    hist = np.load(a.history_bank / "history_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    pred = np.load(a.forecast_bank / "forecast_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if hist.shape[1:] != (2, 2, 1, 3, 3) or pred.shape[1:] != (2, 16, 1, 3, 3):
        raise ValueError(f"unexpected array shapes {hist.shape} {pred.shape}")
    h = np.expm1(np.asarray(hist[hindices, :, :, 0], dtype=np.float32))
    q = np.expm1(np.asarray(pred[findices, :, :, 0], dtype=np.float32))
    if not np.isfinite(h).all() or not np.isfinite(q).all() or (h < 0).any() or (q < 0).any():
        raise ValueError("invalid COT value")
    stations = {}
    for si, name in enumerate(("sili", "zhujia")):
        hist_tv = edge_tv(h[:, si, 1])
        first_tv = edge_tv(q[:, si, 0])
        future_tv = edge_tv(q[:, si])
        # Per-sequence means prevent treating each lead as an independent case.
        hist_by_key = hist_tv
        first_by_key = first_tv
        future_by_key = future_tv.mean(axis=1)
        stations[name] = {
            "history_last_frame_edge_tv_physical_cot": summarize(hist_by_key),
            "forecast_first_lead_edge_tv_physical_cot": summarize(first_by_key),
            "forecast_mean_over_16_leads_edge_tv_physical_cot": summarize(future_by_key),
            "ratio_of_means_first_to_history": float(first_by_key.mean() / max(hist_by_key.mean(), 1e-6)),
            "ratio_of_means_forecast16_to_history": float(future_by_key.mean() / max(hist_by_key.mean(), 1e-6)),
            "date_block_ci_first_physical": date_block_ratio(first_by_key, hist_by_key, days),
            "date_block_ci_forecast16_physical": date_block_ratio(future_by_key, hist_by_key, days),
            "fraction_history_maps_edge_tv_below_0_1": float(np.mean(hist_by_key < 0.1)),
            "history_last_frame_edge_tv_log1p_cot": summarize(edge_tv(np.log1p(h[:, si, 1]))),
            "forecast_first_lead_edge_tv_log1p_cot": summarize(edge_tv(np.log1p(q[:, si, 0]))),
            "forecast_mean_over_16_leads_edge_tv_log1p_cot": summarize(edge_tv(np.log1p(q[:, si])).mean(axis=1)),
            "date_block_ci_first_log1p": date_block_ratio(edge_tv(np.log1p(q[:, si, 0])), edge_tv(np.log1p(h[:, si, 1])), days),
            "date_block_ci_forecast16_log1p": date_block_ratio(edge_tv(np.log1p(q[:, si])).mean(axis=1), edge_tv(np.log1p(h[:, si, 1])), days),
        }
    out = {
        "state": "FOLD1_OOF_COT_SPATIAL_TEXTURE_DESCRIPTIVE_SCREEN",
        "interpretation_limit": "3x3 station neighborhood in a single heldout training-date fold; not COT skill or GHI gain",
        "inputs": {
            "script_sha256": sha(Path(__file__)),
            "forecast_manifest_sha256": sha(a.forecast_bank / "bank_manifest.json"),
            "history_manifest_sha256": sha(a.history_bank / "bank_manifest.json"),
            "forecast_rows": len(fi),
            "unique_bjt_init_dates": len(set(days)),
            "same_key_train_intersection": len(ids),
            "shared_frozen_r_checkpoint_sha256": fm["r_checkpoint_sha256"],
            "future_observed_agri_read": False,
            "cpp_read": False,
            "ghi_read": False,
            "test_or_buffer_payload_read": False,
        },
        "metric": "mean absolute 4-neighbor edge difference over 12 edges in each 3x3 map; both physical-COT and log1p-COT scales",
        "stations": stations,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
