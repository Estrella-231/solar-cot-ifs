#!/usr/bin/env python3
"""Independent FP64 common-support audit for COT-aware SimVP forecasts.

Scores saved log1p(COT) banks against the same CPP-reference mask for the
original SimVP, equal-budget MSE update, and COT-auxiliary candidate. Bootstrap
clusters are BJT target dates, paired across all methods/stations/leads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


METHODS = ("original_simvp", "mse_4000", "cot_aux_4000")
EXPECTED_HASHES = {
    "cpp_cot_physical.npy": "2791e21aea1ace67fa43c69520bcb3732092d37fc8c2fb866438c9a00e56cb65",
    "cpp_common_mask.npy": "6659a170a6c087ffe56962d4362117b93ffed5bc2bcf2659320a00a87299d3c5",
    "frame_valid_mask.npy": "b04a8a62c2ac266ddcb7d7ef1f949a607f3f82c6045dcf1cacd80e929df23a6e",
    "frame_keys.csv": "75fbd1c17cd0a1ddd7c645ea78b3718e045b7fdafbc2ffdf970a34679e623727",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_utc(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def metric(sse: np.ndarray, count: np.ndarray) -> float:
    # sse/count have [station,lead] axes; macro-average supported cells equally.
    rmse = np.sqrt(np.divide(sse, count, out=np.full_like(sse, np.nan, dtype=np.float64), where=count > 0))
    if not np.isfinite(rmse).any():
        return float("nan")
    return float(np.nanmean(rmse))


def macro_ratio(total: np.ndarray, count: np.ndarray) -> float:
    value = np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)
    return float(np.nanmean(value)) if np.isfinite(value).any() else float("nan")


def aggregate_scale(pred: dict[str, np.ndarray], ref: np.ndarray, mask: np.ndarray,
                    date_index: np.ndarray, dates: list[str], bootstrap: int, seed: int) -> dict:
    # Inputs have [sequence, station, lead, row, col]. Sufficient statistics
    # are retained by target BJT date so each paired resample is reproducible.
    mcount, ndate = len(METHODS), len(dates)
    shape = (ndate, mcount, 2, 16)
    sse = np.zeros(shape, np.float64)
    sae = np.zeros(shape, np.float64)
    bias = np.zeros(shape, np.float64)
    count = np.zeros(shape, np.int64)
    tp = np.zeros(shape, np.int64); fp = np.zeros(shape, np.int64); fn = np.zeros(shape, np.int64)
    low_n = np.zeros(shape, np.int64); low_fp = np.zeros(shape, np.int64)
    for q in range(ref.shape[0]):
        for st in range(2):
            for lead in range(16):
                d = int(date_index[q, lead])
                y = np.asarray(ref[q, st, lead], dtype=np.float64)
                valid = np.asarray(mask[q, st, lead], dtype=bool)
                if not valid.any():
                    continue
                for mi, name in enumerate(METHODS):
                    yhat = np.asarray(pred[name][q, st, lead], dtype=np.float64)
                    e = yhat[valid] - y[valid]
                    sse[d, mi, st, lead] += np.dot(e, e)
                    sae[d, mi, st, lead] += np.abs(e).sum(dtype=np.float64)
                    bias[d, mi, st, lead] += e.sum(dtype=np.float64)
                    count[d, mi, st, lead] += e.size
                    cloud = y >= 30
                    forecast = yhat >= 30
                    tp[d, mi, st, lead] += np.count_nonzero(valid & cloud & forecast)
                    fp[d, mi, st, lead] += np.count_nonzero(valid & ~cloud & forecast)
                    fn[d, mi, st, lead] += np.count_nonzero(valid & cloud & ~forecast)
                    low = valid & (y < 1)
                    low_n[d, mi, st, lead] += np.count_nonzero(low)
                    low_fp[d, mi, st, lead] += np.count_nonzero(low & (yhat >= 1))

    totals = {"sse": sse.sum(axis=0), "sae": sae.sum(axis=0), "bias": bias.sum(axis=0),
              "count": count.sum(axis=0), "tp": tp.sum(axis=0), "fp": fp.sum(axis=0),
              "fn": fn.sum(axis=0), "low_n": low_n.sum(axis=0), "low_fp": low_fp.sum(axis=0)}
    names = list(METHODS)
    point_rmse = {name: metric(totals["sse"][i], totals["count"][i]) for i, name in enumerate(names)}
    deltas = {"cot_minus_original": point_rmse["cot_aux_4000"] - point_rmse["original_simvp"],
              "cot_minus_equal_budget_mse": point_rmse["cot_aux_4000"] - point_rmse["mse_4000"]}

    rng = np.random.default_rng(seed)
    dcount = np.zeros((bootstrap, ndate), np.int16)
    for b in range(bootstrap):
        dcount[b] = np.bincount(rng.integers(0, ndate, size=ndate), minlength=ndate)
    boot_sse = np.tensordot(dcount.astype(np.float64), sse, axes=(1, 0))
    boot_n = np.tensordot(dcount.astype(np.float64), count, axes=(1, 0))
    boot_score = np.full((bootstrap, mcount), np.nan, np.float64)
    for b in range(bootstrap):
        for mi in range(mcount):
            boot_score[b, mi] = metric(boot_sse[b, mi], boot_n[b, mi])
    delta_indices = {"cot_minus_original": (2, 0), "cot_minus_equal_budget_mse": (2, 1)}
    intervals = {}
    for key, (a, b) in delta_indices.items():
        vals = boot_score[:, a] - boot_score[:, b]
        vals = vals[np.isfinite(vals)]
        intervals[key] = [float(x) for x in np.quantile(vals, [0.025, 0.975])]

    def rate_boot(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
        bn = np.tensordot(dcount.astype(np.float64), numer, axes=(1, 0)).sum(axis=(-2, -1))
        bd = np.tensordot(dcount.astype(np.float64), denom, axes=(1, 0)).sum(axis=(-2, -1))
        return np.divide(bn, bd, out=np.full_like(bn, np.nan), where=bd > 0)

    boot_csi = rate_boot(tp, tp + fp + fn)
    boot_far = rate_boot(low_fp, low_n)
    rate_intervals = {}
    for label, boot_values, point_numer, point_denom in (
        ("pooled_thick_csi_cot_ge30", boot_csi, tp, tp + fp + fn),
        ("near_zero_false_alarm_rate_cpp_lt1_pred_ge1", boot_far, low_fp, low_n),
    ):
        point_rate = np.divide(point_numer.sum(axis=(0, 2, 3)), point_denom.sum(axis=(0, 2, 3)),
                               out=np.full((mcount,), np.nan), where=point_denom.sum(axis=(0, 2, 3)) > 0)
        rate_intervals[label] = {}
        for key, (a, b) in delta_indices.items():
            vals = boot_values[:, a] - boot_values[:, b]
            vals = vals[np.isfinite(vals)]
            rate_intervals[label][key] = {
                "point_delta": float(point_rate[a] - point_rate[b]),
                "95ci": [float(x) for x in np.quantile(vals, [0.025, 0.975])],
            }

    scores = {}
    for i, name in enumerate(names):
        n = totals["count"][i]
        tp_i, fp_i, fn_i = (totals[k][i] for k in ("tp", "fp", "fn"))
        cell_rmse = np.sqrt(np.divide(totals["sse"][i], n,
            out=np.full_like(n, np.nan, dtype=np.float64), where=n > 0))
        scores[name] = {
            "station_lead_macro_rmse": point_rmse[name],
            "station_lead_rmse": [[float(x) if np.isfinite(x) else None for x in row] for row in cell_rmse],
            "station_lead_macro_mae": macro_ratio(totals["sae"][i], n),
            "station_lead_macro_bias": macro_ratio(totals["bias"][i], n),
            "pooled_thick_csi_cot_ge30": float(tp_i.sum() / max(1, tp_i.sum() + fp_i.sum() + fn_i.sum())),
            "near_zero_false_alarm_rate_cpp_lt1_pred_ge1": float(totals["low_fp"][i].sum() / max(1, totals["low_n"][i].sum())),
            "valid_pixels_by_station": [int(x) for x in n.sum(axis=1)],
            "supported_station_lead_cells": int(np.isfinite(np.sqrt(np.divide(totals["sse"][i], n,
                out=np.full_like(n, np.nan, dtype=np.float64), where=n > 0))).sum()),
        }
    return {"scores": scores, "point_delta": deltas, "paired_bjt_target_date_bootstrap_95ci": intervals,
            "paired_rate_deltas": rate_intervals,
            "bootstrap_replicates": bootstrap, "bootstrap_seed": seed,
            "bootstrap_date_count": ndate, "dates": dates}


def main() -> None:
    p = argparse.ArgumentParser()
    for n in ("pack", "sidecars", "r-bank", "run", "mse-control", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261017)
    a = p.parse_args()

    pack_meta = json.loads((a.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    side_meta = json.loads((a.sidecars / "pack_manifest.json").read_text(encoding="utf-8"))
    if pack_meta.get("test_payload_opened") is not False or side_meta.get("test_payload_opened") is not False:
        raise ValueError("test exclusion is not certified")
    seq_all = list(csv.DictReader((a.pack / "sequence_keys.csv").open(encoding="utf-8")))
    seq = [r for r in seq_all if r["split"] == "val"]
    ids = np.load(a.pack / "sequence_indices_val.npy", mmap_mode="r", allow_pickle=False)
    if len(seq) != len(ids) or len(seq) != 3140:
        raise ValueError("validation sequence key count mismatch")
    for root, name in ((a.pack, "cpp_cot_physical.npy"), (a.pack, "cpp_common_mask.npy"),
                       (a.sidecars, "frame_valid_mask.npy"), (a.r_bank, "frame_keys.csv")):
        if sha(root / name) != EXPECTED_HASHES[name]:
            raise ValueError(f"frozen input hash mismatch: {root / name}")
    keys = list(csv.DictReader((a.r_bank / "frame_keys.csv").open(encoding="utf-8")))
    if len(keys) != 55733:
        raise ValueError("unexpected frame-key count")
    frame_times = [r["target_time_utc"] for r in keys]
    frame_splits = [r["split"] for r in keys]
    date_index = np.empty((len(seq), 16), np.int16)
    date_set = set()
    for q, row in enumerate(seq):
        init = parse_utc(row["init_time_utc"])
        for k in range(16):
            fi = int(ids[q, 8 + k])
            t = parse_utc(frame_times[fi])
            expected = init + timedelta(minutes=15 * (k + 1))
            if t != expected or frame_splits[fi] != "val":
                raise ValueError(f"forecast target key/split mismatch at sequence={q}, lead={k+1}")
            date_bjt = (t + timedelta(hours=8)).date().isoformat()
            date_set.add(date_bjt)
    dates = sorted(date_set); date_lookup = {d: i for i, d in enumerate(dates)}
    for q, row in enumerate(seq):
        init = parse_utc(row["init_time_utc"])
        for k in range(16):
            d = (init + timedelta(minutes=15 * (k + 1)) + timedelta(hours=8)).date().isoformat()
            date_index[q, k] = date_lookup[d]

    score_dirs = {"original_simvp": a.run / "cot_score_original_simvp",
                  "mse_4000": a.run / "cot_score_mse_4000",
                  "cot_aux_4000": a.run / "cot_score_aux_4000"}
    pred = {}
    pred_hashes = {}
    for name, d in score_dirs.items():
        score_path = d / "cot_aux_validation_score.json"
        complete = a.run / "training_complete.json"
        if not score_path.is_file() or not complete.is_file():
            raise FileNotFoundError(f"missing completed score: {score_path}")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        if score.get("status") != "PASS_CPP_COMMON_SUPPORT_FP64" or score.get("test_payloads_read") != 0:
            raise ValueError(f"score contract failed for {name}")
        bank_path = d / "forecast_log1p_cot.npy"
        if score.get("bank_sha256") != sha(bank_path):
            raise ValueError(f"prediction bank hash does not match score receipt for {name}")
        arr = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        row_ids = np.load(d / "validation_dataset_indices.npy", mmap_mode="r", allow_pickle=False)
        if arr.shape != (len(seq), 2, 16, 16, 16) or not np.array_equal(row_ids, np.arange(len(seq))):
            raise ValueError(f"prediction bank shape/keys mismatch for {name}")
        pred[name] = np.expm1(arr.astype(np.float64))
        pred_hashes[name] = sha(bank_path)

    cpp = np.load(a.pack / "cpp_cot_physical.npy", mmap_mode="r", allow_pickle=False)
    cm = np.load(a.pack / "cpp_common_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    valid = np.load(a.sidecars / "frame_valid_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    if cpp.shape != (len(keys), 2, 1, 16, 16) or cm.shape != cpp.shape or valid.shape != cpp.shape:
        raise ValueError("CPP/reference mask shape mismatch")
    ref = np.empty((len(seq), 2, 16, 16, 16), np.float64)
    mask = np.empty_like(ref, dtype=bool)
    for q in range(len(seq)):
        fi = np.asarray(ids[q, 8:], dtype=np.int64)
        for st in range(2):
            ref[q, st] = np.asarray(cpp[fi, st, 0], dtype=np.float64)
            mask[q, st] = np.asarray(cm[fi, st, 0], dtype=bool) & np.asarray(valid[fi, st, 0], dtype=bool)
    if int(mask.sum()) == 0:
        raise ValueError("empty common CPP support")

    # Bootstrap only dates that actually contribute CPP-valid pixels; target
    # dates without reference support must not dilute the date-block resample.
    active = set()
    for q in range(len(seq)):
        for k in range(16):
            fi = int(ids[q, 8 + k])
            if (cm[fi].astype(bool) & valid[fi].astype(bool)).any():
                active.add(int(date_index[q, k]))
    active = sorted(active)
    if not active:
        raise ValueError("no BJT target date contributes to CPP common support")
    compact = {old: new for new, old in enumerate(active)}
    date_index = np.vectorize(lambda x: compact.get(int(x), -1), otypes=[np.int16])(date_index)
    dates = [dates[i] for i in active]

    run_meta = json.loads((a.run / "run_metadata.json").read_text(encoding="utf-8"))
    preflight = json.loads((a.run / "preflight.json").read_text(encoding="utf-8"))
    complete = json.loads((a.run / "training_complete.json").read_text(encoding="utf-8"))
    control_meta = json.loads((a.mse_control / "run_metadata.json").read_text(encoding="utf-8"))
    control_done = json.loads((a.mse_control / "training_complete.json").read_text(encoding="utf-8"))
    fixed_rows_path = a.mse_control.parent / "fixed_training_rows.json"
    fixed_rows = json.loads(fixed_rows_path.read_text(encoding="utf-8"))
    control_ckpt = a.mse_control / "step_4000.pt"
    if (run_meta.get("fixed_training_rows_sha256") != fixed_rows.get("sha256") or
            run_meta.get("equal_budget_mse_step4000_sha256") != sha(control_ckpt) or
            run_meta.get("split_id") != pack_meta.get("split_id") or run_meta.get("seed") != 42 or
            run_meta.get("steps") != 4000 or run_meta.get("batch") != 8 or
            preflight.get("status") != "PASS" or preflight.get("R_frozen") is not True or
            complete.get("status") != "PASS_TRAIN_AND_COT_SCORE" or complete.get("test_payloads_read") != 0):
        raise ValueError("candidate run provenance/completion check failed")
    if (control_meta.get("arm") != "A" or control_meta.get("loss") != "mse" or
            control_meta.get("split_id") != pack_meta.get("split_id") or
            control_meta.get("seed") != 42 or control_meta.get("steps") != 4000 or
            control_meta.get("effective_batch") != 8 or control_done.get("test_payloads_read") != 0):
        raise ValueError("equal-budget MSE control provenance check failed")
    for name, d in score_dirs.items():
        score = json.loads((d / "cot_aux_validation_score.json").read_text(encoding="utf-8"))
        if score.get("valid_cpp_dates") != len(dates) or score.get("validation_sequences") != len(seq):
            raise ValueError(f"score date/sequence support mismatch for {name}")

    all_scores = {}
    for scale, sl in (("patch_16x16", (slice(None), slice(None))),
                      ("station_local_3x3", (slice(7, 10), slice(7, 10)))):
        ps = {k: v[..., sl[0], sl[1]] for k, v in pred.items()}
        ys, ms = ref[..., sl[0], sl[1]], mask[..., sl[0], sl[1]]
        all_scores[scale] = aggregate_scale(ps, ys, ms, date_index, dates, a.bootstrap, a.seed)

    # The candidate's AGRI guard is computed from the saved 4000-step
    # sufficient statistics on the same validation cohort as its MSE control.
    cand_agri = json.loads((a.run / "validation_step_4000.json").read_text(encoding="utf-8"))
    mse_agri = json.loads((a.mse_control / "validation_step_4000.json").read_text(encoding="utf-8"))
    agri_change = float(cand_agri["val_equal_season_rmse"] / mse_agri["val_equal_season_rmse"] - 1.0)
    gates = {}
    for scale, result in all_scores.items():
        rmse_ok = all(result["point_delta"][key] < 0 and
                      result["paired_bjt_target_date_bootstrap_95ci"][key][1] < 0
                      for key in ("cot_minus_original", "cot_minus_equal_budget_mse"))
        csi_ok = all(result["paired_rate_deltas"]["pooled_thick_csi_cot_ge30"][key]["point_delta"] >= -0.02
                     for key in ("cot_minus_original", "cot_minus_equal_budget_mse"))
        far_ok = all(result["paired_rate_deltas"]["near_zero_false_alarm_rate_cpp_lt1_pred_ge1"][key]["95ci"][0] <= 0
                     for key in ("cot_minus_original", "cot_minus_equal_budget_mse"))
        gates[scale] = {"rmse_improves_both_refs_with_upper_ci_below_zero": rmse_ok,
                        "thick_csi_drop_within_0p02": csi_ok,
                        "no_significant_near_zero_false_alarm_increase": far_ok,
                        "pass": rmse_ok and csi_ok and far_ok}
    gate_pass = all(v["pass"] for v in gates.values()) and agri_change <= 0.01
    result = {
        "state": "PASS_INDEPENDENT_FP64_PAIRED_COT_AUDIT",
        "split_id": pack_meta.get("split_id"), "validation_sequences": len(seq),
        "unique_bjt_target_dates": len(dates), "common_cpp_valid_pixels_patch": int(mask.sum()),
        "test_payloads_read": 0, "cpp_is_independent_truth": False,
        "score_basis": "physical COT expm1(log bank), common CPP mask AND R-frame-valid mask, FP64 SSE, equal station-lead macro",
        "date_block_unit": "BJT target date, sampled with replacement and paired across all methods/stations/leads",
        "source_hashes": {"pack_manifest": sha(a.pack / "pack_manifest.json"),
                          "sidecar_manifest": sha(a.sidecars / "pack_manifest.json"),
                          "frame_keys": sha(a.r_bank / "frame_keys.csv"),
                          "equal_budget_mse_step4000": sha(control_ckpt),
                          "fixed_training_rows": sha(fixed_rows_path),
                          **{f"prediction_{k}": v for k, v in pred_hashes.items()}},
        "agri_guard": {"candidate_val_equal_season_rmse": cand_agri["val_equal_season_rmse"],
                       "mse_val_equal_season_rmse": mse_agri["val_equal_season_rmse"],
                       "relative_change_candidate_vs_mse": agri_change,
                       "pass_under_1pct_limit": agri_change <= 0.01},
        "screening_gate": {"pass": gate_pass,
                           "rule": "patch and centre-3x3 COT RMSE improves against original SimVP and equal-budget MSE with paired date-block 95% upper CI below zero; thick CSI drop <=0.02; near-zero false-alarm increase not significant; AGRI seasonal RMSE worsening <=1%",
                           "by_scale": gates},
        "scales": all_scores,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({s: {"point_delta": x["point_delta"],
                         "ci": x["paired_bjt_target_date_bootstrap_95ci"],
                         "gate": gates[s],
                         "rmse": {k: v["station_lead_macro_rmse"] for k, v in x["scores"].items()}}
                      for s, x in all_scores.items()}, indent=2))
    print(a.out)


if __name__ == "__main__":
    main()
