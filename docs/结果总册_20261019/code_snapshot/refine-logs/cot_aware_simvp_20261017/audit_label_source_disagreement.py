#!/usr/bin/env python3
"""Audit CPP vs observed-AGRI->R label differences on shared valid support.

This is a diagnostic only: CPP is a retrieval reference, not independent truth.
It never opens test payloads and does not select model checkpoints or weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summarize(cpp: np.ndarray, r_log: np.ndarray, mask: np.ndarray) -> dict:
    """Summarize matched CPP and R values using only explicitly valid pixels."""
    c = np.asarray(cpp, dtype=np.float64)[mask]
    lr = np.asarray(r_log, dtype=np.float64)[mask]
    finite = np.isfinite(c) & np.isfinite(lr) & (c >= 0)
    c, lr = c[finite], lr[finite]
    if c.size == 0:
        return {"n": 0}
    r = np.expm1(lr)
    d = r - c
    return {
        "n": int(c.size),
        "cpp_mean": float(c.mean()),
        "r_mean": float(r.mean()),
        "r_minus_cpp_bias": float(d.mean()),
        "r_minus_cpp_mae": float(np.abs(d).mean()),
        "r_minus_cpp_rmse": float(np.sqrt(np.mean(np.square(d)))),
        "r_minus_cpp_log1p_bias": float((lr - np.log1p(c)).mean()),
        "cpp_quantiles": [float(x) for x in np.quantile(c, [0.1, 0.5, 0.9])],
        "r_quantiles": [float(x) for x in np.quantile(r, [0.1, 0.5, 0.9])],
        "cpp_thick_fraction_ge_30": float(np.mean(c >= 30)),
        "r_thick_fraction_ge_30": float(np.mean(r >= 30)),
        "cpp_nearzero_fraction_lt_1": float(np.mean(c < 1)),
        "r_nearzero_fraction_lt_1": float(np.mean(r < 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--sidecars", type=Path, required=True)
    p.add_argument("--r-bank", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.sample_size < 1:
        raise ValueError("sample-size must be positive")

    keys_path = a.pack / "sequence_keys.csv"
    seq_path = a.pack / "sequence_indices_val.npy"
    frame_keys_path = a.r_bank / "frame_keys.csv"
    cpp_path = a.pack / "cpp_cot_physical.npy"
    cpp_mask_path = a.pack / "cpp_common_mask.npy"
    valid_path = a.sidecars / "frame_valid_mask.npy"
    r_path = a.r_bank / "cot_log1p.npy"
    arrays = {
        "cpp_cot_physical.npy": cpp_path,
        "cpp_common_mask.npy": cpp_mask_path,
        "frame_valid_mask.npy": valid_path,
        "cot_log1p.npy": r_path,
        "sequence_indices_val.npy": seq_path,
    }
    for path in [keys_path, frame_keys_path, *arrays.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    seq = np.load(seq_path, mmap_mode="r", allow_pickle=False)
    cpp = np.load(cpp_path, mmap_mode="r", allow_pickle=False)
    cpp_mask = np.load(cpp_mask_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    r_log = np.load(r_path, mmap_mode="r", allow_pickle=False)
    expected = (len(seq), 24)
    if seq.ndim != 2 or seq.shape != expected:
        raise ValueError(f"unexpected validation sequence shape: {seq.shape}")
    if cpp.shape != (55733, 2, 1, 16, 16) or cpp_mask.shape != cpp.shape \
            or valid.shape != cpp.shape or r_log.shape != cpp.shape:
        raise ValueError("unexpected CPP/R/mask bank shape")

    rows = list(csv.DictReader(keys_path.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row["split"] == "val"]
    if len(rows) != len(seq):
        raise ValueError("validation sequence-key count mismatch")
    rng = np.random.default_rng(a.seed)
    chosen = np.sort(rng.choice(len(seq), size=min(a.sample_size, len(seq)), replace=False))

    # Accumulate matched samples per station/lead and per source region.
    pair = {(s, t): [[], []] for s in range(2) for t in range(16)}
    cot_bins = (("lt1", -np.inf, 1.0), ("1to10", 1.0, 10.0),
                ("10to30", 10.0, 30.0), ("ge30", 30.0, np.inf))
    pair_by_cot_bin = {(s, name): [[], []] for s in range(2) for name, _, _ in cot_bins}
    source = {(s, region): [] for s in range(2) for region in ("cpp", "fallback")}
    coverage = {(s, t): [0, 0] for s in range(2) for t in range(16)}
    for start in range(0, len(chosen), 32):
        ids = chosen[start:start + 32]
        frame_ids = np.asarray(seq[ids, 8:24], dtype=np.int64)
        c = np.asarray(cpp[frame_ids, :, 0], dtype=np.float32)
        cm = np.asarray(cpp_mask[frame_ids, :, 0], dtype=bool)
        vm = np.asarray(valid[frame_ids, :, 0], dtype=bool)
        lr = np.asarray(r_log[frame_ids, :, 0], dtype=np.float32)
        for s in range(2):
            rvalid = vm[:, :, s] & np.isfinite(lr[:, :, s])
            cvalid = cm[:, :, s] & rvalid & np.isfinite(c[:, :, s]) & (c[:, :, s] >= 0)
            fallback = rvalid & ~cvalid
            if np.any(cvalid):
                source[(s, "cpp")].append(lr[:, :, s][cvalid])
            if np.any(fallback):
                source[(s, "fallback")].append(lr[:, :, s][fallback])
            cv = c[:, :, s][cvalid]
            rv = lr[:, :, s][cvalid]
            for name, lo, hi in cot_bins:
                in_bin = (cv >= lo) & (cv < hi)
                if np.any(in_bin):
                    pair_by_cot_bin[(s, name)][0].append(cv[in_bin])
                    pair_by_cot_bin[(s, name)][1].append(rv[in_bin])
            for t in range(16):
                overlap = cvalid[:, t]
                coverage[(s, t)][0] += int(overlap.sum())
                coverage[(s, t)][1] += int(rvalid[:, t].sum())
                if np.any(overlap):
                    pair[(s, t)][0].append(c[:, t, s][overlap])
                    pair[(s, t)][1].append(lr[:, t, s][overlap])

    station_names = ("Sili", "Zhujia")
    per_station_lead = {}
    per_station = {}
    pooled_cpp, pooled_r = [], []
    for s, name in enumerate(station_names):
        station_cpp, station_r = [], []
        for t in range(16):
            cs, rs = pair[(s, t)]
            cvec = np.concatenate(cs) if cs else np.empty(0, np.float32)
            rvec = np.concatenate(rs) if rs else np.empty(0, np.float32)
            per_station_lead[f"{name}_lead_{(t + 1) * 15:03d}min"] = summarize(
                cvec, rvec, np.ones(cvec.shape, dtype=bool))
            station_cpp.append(cvec)
            station_r.append(rvec)
        sc = np.concatenate(station_cpp) if station_cpp else np.empty(0, np.float32)
        sr = np.concatenate(station_r) if station_r else np.empty(0, np.float32)
        per_station[name] = summarize(sc, sr, np.ones(sc.shape, dtype=bool))
        pooled_cpp.append(sc)
        pooled_r.append(sr)

    all_cpp = np.concatenate(pooled_cpp)
    all_r = np.concatenate(pooled_r)
    source_stats = {}
    cot_bin_stats = {}
    for s, name in enumerate(station_names):
        for region in ("cpp", "fallback"):
            vals = np.concatenate(source[(s, region)]) if source[(s, region)] else np.empty(0, np.float32)
            physical = np.expm1(vals.astype(np.float64))
            source_stats[f"{name}_{region}_R_target_distribution"] = {
                "n": int(vals.size),
                "mean_cot": float(physical.mean()) if vals.size else None,
                "median_cot": float(np.median(physical)) if vals.size else None,
                "fraction_ge_30": float(np.mean(physical >= 30)) if vals.size else None,
                "fraction_lt_1": float(np.mean(physical < 1)) if vals.size else None,
            }
        for bin_name, _, _ in cot_bins:
            cs, rs = pair_by_cot_bin[(s, bin_name)]
            cvec = np.concatenate(cs) if cs else np.empty(0, np.float32)
            rvec = np.concatenate(rs) if rs else np.empty(0, np.float32)
            cot_bin_stats[f"{name}_{bin_name}"] = summarize(
                cvec, rvec, np.ones(cvec.shape, dtype=bool))
    for bin_name, _, _ in cot_bins:
        cs = [np.concatenate(pair_by_cot_bin[(s, bin_name)][0])
              for s in range(2) if pair_by_cot_bin[(s, bin_name)][0]]
        rs = [np.concatenate(pair_by_cot_bin[(s, bin_name)][1])
              for s in range(2) if pair_by_cot_bin[(s, bin_name)][1]]
        cvec = np.concatenate(cs) if cs else np.empty(0, np.float32)
        rvec = np.concatenate(rs) if rs else np.empty(0, np.float32)
        cot_bin_stats[f"pooled_{bin_name}"] = summarize(
            cvec, rvec, np.ones(cvec.shape, dtype=bool))

    output = {
        "status": "DIAGNOSTIC_ONLY",
        "split": "validation",
        "test_payloads_opened": False,
        "sample_size": int(len(chosen)),
        "sample_seed": int(a.seed),
        "sample_indices": [int(i) for i in chosen],
        "sample_sequence_keys": [rows[int(i)] for i in chosen],
        "sample_indices_sha256": hashlib.sha256(chosen.astype(np.int32).tobytes()).hexdigest(),
        "frame_key_sha256": sha256(frame_keys_path),
        "array_sha256": {name: sha256(path) for name, path in arrays.items()},
        "interpretation_limit": "CPP is a retrieval reference, not independent truth; overlap disagreement is not causal proof that R fallback harms GHI.",
        "station_cpp_vs_R_on_common_valid_support": per_station,
        "station_lead_cpp_vs_R_on_common_valid_support": per_station_lead,
        "R_target_distribution_by_source_region": source_stats,
        "R_minus_CPP_error_stratified_by_CPP_reference_cot": cot_bin_stats,
        "cpp_coverage_fraction_within_R_valid_support": {
            f"{station_names[s]}_lead_{(t + 1) * 15:03d}min": (
                float(cov[0] / cov[1]) if cov[1] else None
            ) for (s, t), cov in coverage.items()
        },
        "pooled_common_support": summarize(all_cpp, all_r, np.ones(all_cpp.shape, dtype=bool)),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(a.output)
    print(json.dumps({"status": output["status"], "sample_size": len(chosen),
                      "common_pixels": output["pooled_common_support"]["n"],
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
