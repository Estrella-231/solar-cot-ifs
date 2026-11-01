#!/usr/bin/env python3
"""Compare direct COT forecasts with the SimVP->R bank on exact validation keys.

All methods are scored against the compact, date-keyed CPP reference and the
same CPP-common AND frame-valid mask. CPP is a retrieval reference, not truth.
This is a read-only validation analysis; it never opens test payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_utc(s: str) -> datetime:
    value = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--sidecars", type=Path, required=True)
    ap.add_argument("--simvp-bank", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--amplitude-run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # Validate full row-level key/time correspondence before reading predictions.
    keys = list(csv.DictReader((a.run / "validation_prediction_keys.csv").open(encoding="utf-8")))
    seqkeys = list(csv.DictReader((a.pack / "sequence_keys.csv").open(encoding="utf-8")))
    valkeys = [r for r in seqkeys if r["split"] == "val"]
    ids = np.load(a.pack / "sequence_indices_val.npy", mmap_mode="r", allow_pickle=False)
    if len(valkeys) != len(ids) or len(keys) != 2 * len(ids):
        raise ValueError("sequence/prediction key counts do not match")
    station_order = ("sili", "zhujia")
    for q, seq in enumerate(valkeys):
        for s, st in enumerate(station_order):
            row = keys[q * 2 + s]
            if row["station"] != st or row["init_time_utc"] != seq["init_time_utc"]:
                raise ValueError(f"direct prediction key mismatch at sequence={q}, station={s}: {row} vs {seq}")

    z = np.load(a.simvp_bank, allow_pickle=False)
    simvp_log = z["forecast_log1p_cot"]
    simvp_target_times = z["target_time_utc"]
    simvp_seq_ids = z["sequence_ids"]
    if simvp_log.shape != (len(ids), 2, 16, 16, 16):
        raise ValueError(f"unexpected SimVP bank shape {simvp_log.shape}")
    time_mismatches = []
    for q, seq in enumerate(valkeys):
        if str(simvp_seq_ids[q]) != seq["seq_id"]:
            time_mismatches.append((q, seq["seq_id"], str(simvp_seq_ids[q])))
            if len(time_mismatches) >= 5:
                break
        init = parse_utc(seq["init_time_utc"])
        expected = [init + timedelta(minutes=15 * (k + 1)) for k in range(16)]
        got = [parse_utc(t) for t in simvp_target_times[q]]
        if got != expected:
            time_mismatches.append((q, str(expected[0]), str(got[0])))
            if len(time_mismatches) >= 5:
                break
    if time_mismatches:
        raise ValueError(f"SimVP target-time mismatch (first five): {time_mismatches}")

    bank = np.load(a.bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(a.pack / "cpp_cot_physical.npy", mmap_mode="r", allow_pickle=False)
    cppmask = np.load(a.pack / "cpp_common_mask.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(a.sidecars / "frame_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    pred_files = {
        "P_R": a.run / "P_R_val_pred_epoch03.npy",
        "P_Hybrid": a.run / "P_Hybrid_val_pred_epoch09.npy",
    }
    amp_epoch = None
    if a.amplitude_run is not None:
        amp_history = json.loads((a.amplitude_run / "P_Hybrid_metrics.json").read_text(encoding="utf-8"))
        amp_epoch = min(amp_history, key=lambda x: x["cpp_common_rmse_equal_station_lead"])["epoch"]
        pred_files["P_Hybrid_amp01"] = a.amplitude_run / f"P_Hybrid_val_pred_epoch{amp_epoch:02d}.npy"
    preds = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in pred_files.items()}
    if any(x.shape != (2 * len(ids), 16, 1, 16, 16) for x in preds.values()):
        raise ValueError("direct COT prediction shape/key support mismatch")

    # Existing SimVP->R, plus persistence, are evaluated against the exact same
    # compact CPP pixels used by the direct-forecaster evaluation.
    methods = {"SimVP_to_R": np.expm1(np.asarray(simvp_log, dtype=np.float32))}
    # Trainer rows are sequence-major, then station-major. Restore the bank's
    # explicit [sequence, station, lead, row, col] layout before scoring.
    methods.update({name: np.asarray(arr[:, :, 0]).reshape(len(ids), 2, 16, 16, 16)
                    for name, arr in preds.items()})
    persistence = np.empty((len(ids), 2, 16, 16, 16), dtype=np.float32)
    for q in range(len(ids)):
        for s in range(2):
            last = np.expm1(np.asarray(bank[int(ids[q, 7]), s, 0], dtype=np.float32))
            persistence[q, s] = np.broadcast_to(last, (16, 16, 16))
    methods["Persistence"] = persistence
    if methods["Persistence"].shape != (len(ids), 2, 16, 16, 16):
        raise ValueError(f"unexpected persistence shape {methods['Persistence'].shape}")

    bins = [("all", -np.inf, np.inf), ("near_zero_lt1", -np.inf, 1.0),
            ("thin_1_5", 1.0, 5.0), ("moderate_5_30", 5.0, 30.0),
            ("thick_ge30", 30.0, np.inf)]
    # [method, scale, bin, station, lead]
    scales = {"patch_16x16": (slice(None), slice(None)),
              "station_local_3x3": (slice(7, 10), slice(7, 10))}
    scores: dict[str, dict] = {m: {sc: {} for sc in scales} for m in methods}
    counts_by_stratum: dict[str, dict[str, list[int]]] = {sc: {} for sc in scales}
    for sc, (rs, cs) in scales.items():
        for bname, lo, hi in bins:
            sums = {m: np.zeros((2, 16), np.float64) for m in methods}
            bias_sums = {m: np.zeros((2, 16), np.float64) for m in methods}
            counts = np.zeros((2, 16), np.int64)
            target_sums = np.zeros((2, 16), np.float64)
            for q in range(len(ids)):
                fi = np.asarray(ids[q, 8:], dtype=np.int64)
                for s in range(2):
                    ref = np.asarray(cpp[fi, s, 0], dtype=np.float32)[..., rs, cs]
                    cm = np.asarray(cppmask[fi, s, 0], dtype=bool)[..., rs, cs]
                    vm = np.asarray(valid[fi, s, 0], dtype=bool)[..., rs, cs]
                    m = cm & vm
                    m &= (ref >= lo) & (ref < hi)
                    counts[s] += m.sum(axis=(-2, -1))
                    target_sums[s] += np.where(m, ref, 0.0).sum(axis=(-2, -1), dtype=np.float64)
                    for method, p in methods.items():
                        pp = np.asarray(p[q, s], dtype=np.float32)[..., rs, cs]
                        err = np.where(m, pp.astype(np.float64) - ref.astype(np.float64), 0.0)
                        sums[method][s] += np.square(err).sum(axis=(-2, -1))
                        bias_sums[method][s] += err.sum(axis=(-2, -1), dtype=np.float64)
            rmse = {method: np.sqrt(np.divide(ss, counts, out=np.full_like(ss, np.nan), where=counts > 0))
                    for method, ss in sums.items()}
            bias = {method: np.divide(bs, counts, out=np.full_like(bs, np.nan), where=counts > 0)
                    for method, bs in bias_sums.items()}
            target_mean = np.divide(target_sums, counts, out=np.full_like(target_sums, np.nan), where=counts > 0)
            for method, vals in rmse.items():
                station_means = [float(np.nanmean(vals[s])) for s in range(2)]
                station_bias = [float(np.nanmean(bias[method][s])) for s in range(2)]
                scores[method][sc][bname] = {
                    "station_lead_macro_rmse": float(np.nanmean(vals)),
                    "station_rmse": station_means,
                    "station_lead_macro_bias": float(np.nanmean(bias[method])),
                    "station_macro_bias": station_bias,
                    "station_lead_macro_target_mean": float(np.nanmean(target_mean)),
                    "lead_rmse_station_mean": [float(np.nanmean(vals[:, k])) for k in range(16)],
                    "lead_bias_station_mean": [float(np.nanmean(bias[method][:, k])) for k in range(16)],
                    "pixels": [int(counts[s].sum()) for s in range(2)],
                    "supported_station_lead_cells": int(np.isfinite(vals).sum()),
                }
            counts_by_stratum[sc][bname] = [int(counts[s].sum()) for s in range(2)]

    # Difference from SimVP->R, negative is lower RMSE.
    deltas = {}
    candidate_methods = ["P_R", "P_Hybrid", "Persistence"]
    if "P_Hybrid_amp01" in methods:
        candidate_methods.append("P_Hybrid_amp01")
    for method in candidate_methods:
        deltas[method] = {sc: {b: scores[method][sc][b]["station_lead_macro_rmse"] -
                                     scores["SimVP_to_R"][sc][b]["station_lead_macro_rmse"]
                              for b, _, _ in bins} for sc in scales}
    result = {
        "state": "PASS_EXACT_DIRECT_KEY_AND_LEAD_MATCH" if not time_mismatches else "FAIL",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "n_sequences": len(ids), "n_station_samples": 2 * len(ids), "leads": 16,
        "forecast_target_leads_minutes": list(range(15, 241, 15)),
        "direct_prediction_key_file_sha256": sha256(a.run / "validation_prediction_keys.csv"),
        "simvp_bank_sha256": sha256(a.simvp_bank),
        "direct_prediction_sha256": {k: sha256(v) for k, v in pred_files.items()},
        "amplitude_loss_best_epoch": amp_epoch,
        "amplitude_loss_run": str(a.amplitude_run) if a.amplitude_run else None,
        "sequence_id_and_all_16_target_time_keys_exact_match": True,
        "score_mask": "compact CPP common mask AND current new-R frame-valid mask",
        "strata_source": "compact CPP physical COT, [0,1), [1,5), [5,30), [30,+inf)",
        "scores": scores,
        "delta_rmse_vs_SimVP_to_R": deltas,
        "stratum_pixel_counts_station_order_sili_zhujia": counts_by_stratum,
        "cpp_is_independent_truth": False,
        "test_payload_opened": False,
        "notes": ["Scores are CPP-reference consistency, not physical COT truth.",
                  "SimVP_to_R is the deployable forecast-AGRI-to-frozen-R reference.",
                  "Negative delta means lower validation RMSE than SimVP_to_R.",
                  "Single-seed best checkpoints were selected on this same validation cohort; point differences are screening evidence, not independent inference."]
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "sequence_time_key_match": True,
                      "deltas": deltas}, indent=2))
    print(a.out)


if __name__ == "__main__":
    main()
