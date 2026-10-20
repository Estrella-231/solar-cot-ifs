#!/usr/bin/env python3
"""Validation-only lead-wise COT forecast diagnostics on common CPP support.

CPP is a retrieval reference, not independent COT truth. The future observed
AGRI->R series is an offline retrieval reference and is never a deployable
input. Only frozen validation sequences and validation R outputs are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LEADS = tuple(range(15, 241, 15))
METHODS = ("forecast_simvp_to_R", "history_COT_persistence", "offline_observed_future_AGRI_to_R")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stamp(value: str) -> str:
    value = "".join(ch for ch in str(value) if ch.isdigit())
    if len(value) != 12:
        raise ValueError(f"Expected canonical YYYYMMDDHHMM BJT timestamp, got {value!r}")
    datetime.strptime(value, "%Y%m%d%H%M")
    return value


def metric(sse: float, sae: float, signed: float, count: int) -> dict[str, float | int]:
    if count <= 0:
        return {"n_pixels": 0, "rmse_cot": float("nan"), "mae_cot": float("nan"), "bias_cot": float("nan")}
    return {"n_pixels": int(count), "rmse_cot": float(np.sqrt(sse / count)),
            "mae_cot": float(sae / count), "bias_cot": float(signed / count)}


def bootstrap_delta(date_stats: dict[str, dict[str, tuple[float, int]]],
                    left: str, right: str, draws: int, seed: int) -> dict[str, object]:
    dates = sorted(date_stats)
    if len(dates) < 2:
        return {"n_target_dates": len(dates), "delta_rmse_left_minus_right": None,
                "ci95": None, "state": "INSUFFICIENT_DATE_BLOCKS"}
    a = np.asarray([date_stats[d][left][0] for d in dates], dtype=np.float64)
    b = np.asarray([date_stats[d][right][0] for d in dates], dtype=np.float64)
    n = np.asarray([date_stats[d][left][1] for d in dates], dtype=np.float64)
    if not np.array_equal(n, np.asarray([date_stats[d][right][1] for d in dates], dtype=np.float64)):
        raise ValueError("Paired methods do not share exactly the same valid-pixel support")
    delta = float(np.sqrt(a.sum() / n.sum()) - np.sqrt(b.sum() / n.sum()))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(dates), size=(draws, len(dates)))
    count = n[sampled].sum(axis=1)
    boot = np.sqrt(a[sampled].sum(axis=1) / count) - np.sqrt(b[sampled].sum(axis=1) / count)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n_target_dates": len(dates), "delta_rmse_left_minus_right": delta,
            "ci95": [float(lo), float(hi)], "bootstrap_draws": draws,
            "block": "BJT target date", "state": "PAIRED_DATE_BLOCK_BOOTSTRAP"}


def evaluate(cot_bank: Path, s_keys_path: Path, r_val: Path, output: Path,
             draws: int = 2000, seed: int = 20261021) -> dict[str, object]:
    cot = np.load(cot_bank, mmap_mode="r", allow_pickle=False)
    if cot.ndim != 5 or cot.shape[1:] != (16, 1, 16, 16) or cot.shape[0] != 407:
        raise ValueError(f"Unexpected full validation COT bank shape {cot.shape}")
    s_rows = read_rows(s_keys_path)
    if len(s_rows) != len(cot):
        raise ValueError("COT bank and SimVP key count differ")
    target_to_r: dict[str, int] = {}
    for i, row in enumerate(read_rows(r_val / "val_keys.csv")):
        if row.get("split") != "val":
            raise ValueError(f"Non-validation R key in validation bank: {row}")
        key = stamp(row["time_bjt"])
        if key in target_to_r:
            raise ValueError(f"Duplicate R validation time key: {key}")
        target_to_r[key] = i
    r_pred = np.load(r_val / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(r_val / "val_reference_cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(r_val / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    if r_pred.shape != cpp.shape or r_pred.shape != valid.shape or r_pred.shape[1:] != (1, 16, 16):
        raise ValueError(f"R validation arrays disagree: pred={r_pred.shape}, cpp={cpp.shape}, mask={valid.shape}")
    if len(target_to_r) != len(r_pred):
        raise ValueError("R validation keys and arrays are not one-to-one")

    acc: dict[tuple[int, str, str], dict[str, object]] = {}
    skipped_no_target = skipped_no_history = 0
    paired_target_keys: set[tuple[str, str, int]] = set()

    def scope_specs():
        specs = [("all_patch", "ALL_PATCH", (slice(0, 16), slice(0, 16)))]
        for name, (y, x) in CENTERS.items():
            specs.append(("station_pixel", name, (slice(y, y + 1), slice(x, x + 1))))
            specs.append(("station_3x3", name, (slice(y - 1, y + 2), slice(x - 1, x + 2))))
        return specs

    for si, row in enumerate(s_rows):
        if row.get("split") != "val":
            raise ValueError(f"S key bank contains non-validation row: {row}")
        init = stamp(row["init_time_bjt"])
        target_stamps = [stamp(x) for x in row["target_times_bjt"].split("|")]
        if len(target_stamps) != 16:
            raise ValueError(f"Expected 16 target keys for sequence {si}")
        init_dt = datetime.strptime(init, "%Y%m%d%H%M")
        for li, lead in enumerate(LEADS):
            target = target_stamps[li]
            expected = (init_dt + timedelta(minutes=lead)).strftime("%Y%m%d%H%M")
            if target != expected:
                raise ValueError(f"Lead/time mismatch: init={init} lead={lead} target={target} expected={expected}")
            ri = target_to_r.get(target)
            if ri is None:
                skipped_no_target += 1
                continue
            hi = target_to_r.get(init)
            if hi is None:
                skipped_no_history += 1
                continue
            forecast = np.asarray(cot[si, li, 0], dtype=np.float64)
            persistence = np.asarray(r_pred[hi, 0], dtype=np.float64)
            offline = np.asarray(r_pred[ri, 0], dtype=np.float64)
            reference = np.asarray(cpp[ri, 0], dtype=np.float64)
            mask = np.asarray(valid[ri, 0], dtype=bool)
            mask &= np.isfinite(forecast) & np.isfinite(persistence) & np.isfinite(offline) & np.isfinite(reference)
            if not mask.any():
                continue
            paired_target_keys.add((init, target, lead))
            date = target[:8]
            for scope, station, (ys, xs) in scope_specs():
                m = mask[ys, xs]
                if not m.any():
                    continue
                values = {
                    "forecast_simvp_to_R": forecast[ys, xs][m] - reference[ys, xs][m],
                    "history_COT_persistence": persistence[ys, xs][m] - reference[ys, xs][m],
                    "offline_observed_future_AGRI_to_R": offline[ys, xs][m] - reference[ys, xs][m],
                }
                key = (lead, scope, station)
                if key not in acc:
                    acc[key] = {"sum_sq": defaultdict(float), "sum_abs": defaultdict(float),
                                "sum_signed": defaultdict(float), "count": defaultdict(int),
                                "dates": defaultdict(lambda: {m: [0.0, 0] for m in METHODS}),
                                "target_keys": set()}
                cell = acc[key]
                cell["target_keys"].add((init, target))
                for method, err in values.items():
                    ss = float(np.square(err).sum(dtype=np.float64))
                    sa = float(np.abs(err).sum(dtype=np.float64))
                    sg = float(err.sum(dtype=np.float64))
                    n = int(err.size)
                    cell["sum_sq"][method] += ss
                    cell["sum_abs"][method] += sa
                    cell["sum_signed"][method] += sg
                    cell["count"][method] += n
                    cell["dates"][date][method][0] += ss
                    cell["dates"][date][method][1] += n

    rows_out = []
    summary = {}
    for key, cell in sorted(acc.items()):
        lead, scope, station = key
        result = {m: metric(cell["sum_sq"][m], cell["sum_abs"][m], cell["sum_signed"][m], cell["count"][m]) for m in METHODS}
        date_stats = cell["dates"]
        ci_persist = bootstrap_delta(date_stats, "forecast_simvp_to_R", "history_COT_persistence", draws, seed + lead)
        ci_offline = bootstrap_delta(date_stats, "forecast_simvp_to_R", "offline_observed_future_AGRI_to_R", draws, seed + lead + 100)
        record = {"lead_minutes": lead, "scope": scope, "station": station,
                  "n_target_pairs": len(cell["target_keys"]), "n_target_dates": len(date_stats),
                  "cpp_common_support_pixels": int(cell["count"]["forecast_simvp_to_R"]),
                  "forecast": result["forecast_simvp_to_R"],
                  "history_persistence": result["history_COT_persistence"],
                  "offline_future_observed_agri_to_R": result["offline_observed_future_AGRI_to_R"],
                  "delta_forecast_minus_persistence_rmse": ci_persist,
                  "delta_forecast_minus_offline_future_rmse": ci_offline}
        rows_out.append(record)
        summary[f"{lead}|{scope}|{station}"] = record

    if not rows_out:
        raise RuntimeError("No paired validation target-time rows on common CPP support")
    output.mkdir(parents=True, exist_ok=False)
    (output / "forecast_cot_metrics.json").write_text(json.dumps({
        "state": "VALIDATION_ONLY_FORECAST_COT_DIAGNOSTIC_CPP_RETRIEVAL_REFERENCE",
        "scope": "frozen corrected-SimVP -> frozen corrected-R; canonical +15..+240 min; common target CPP-valid support; reused validation, not independent test",
        "reference_warning": "CPP is a satellite retrieval reference, not independent COT truth; offline observed-future AGRI->R is non-deployable and is a retrieval reference only.",
        "pairing": "target_time_bjt exact key; persistence uses exact history_end/init_time_bjt R prediction; date-block bootstrap resamples target BJT dates",
        "n_simvp_validation_sequences": len(s_rows), "n_r_validation_keys": len(target_to_r),
        "n_paired_init_target_lead_keys": len(paired_target_keys),
        "skipped_no_target_reference_key": skipped_no_target,
        "skipped_no_history_reference_key": skipped_no_history,
        "cpp_reference_keys_sha256": sha256(r_val / "val_keys.csv"),
        "cpp_reference_array_sha256": sha256(r_val / "val_reference_cpp_cot.npy"),
        "cpp_mask_sha256": sha256(r_val / "val_valid_mask.npy"),
        "simvp_key_sha256": sha256(s_keys_path), "forecast_cot_bank_sha256": sha256(cot_bank),
        "metrics": summary, "test_payloads_read": 0, "buffer_payloads_read": 0,
    }, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "forecast_cot_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lead_minutes", "scope", "station", "n_target_pairs", "n_target_dates", "cpp_common_support_pixels", "method", "rmse_cot", "mae_cot", "bias_cot", "delta_rmse_vs_persistence", "delta_rmse_vs_persistence_ci95_low", "delta_rmse_vs_persistence_ci95_high", "delta_rmse_vs_offline_future", "delta_rmse_vs_offline_future_ci95_low", "delta_rmse_vs_offline_future_ci95_high"])
        writer.writeheader()
        for item in rows_out:
            for method_key, name in (("forecast", METHODS[0]), ("history_persistence", METHODS[1]), ("offline_future_observed_agri_to_R", METHODS[2])):
                m = item[method_key]
                p = item["delta_forecast_minus_persistence_rmse"]
                o = item["delta_forecast_minus_offline_future_rmse"]
                writer.writerow({"lead_minutes": item["lead_minutes"], "scope": item["scope"], "station": item["station"],
                    "n_target_pairs": item["n_target_pairs"], "n_target_dates": item["n_target_dates"],
                    "cpp_common_support_pixels": item["cpp_common_support_pixels"], "method": name,
                    "rmse_cot": m["rmse_cot"], "mae_cot": m["mae_cot"], "bias_cot": m["bias_cot"],
                    "delta_rmse_vs_persistence": p["delta_rmse_left_minus_right"],
                    "delta_rmse_vs_persistence_ci95_low": (p["ci95"] or [None, None])[0],
                    "delta_rmse_vs_persistence_ci95_high": (p["ci95"] or [None, None])[1],
                    "delta_rmse_vs_offline_future": o["delta_rmse_left_minus_right"],
                    "delta_rmse_vs_offline_future_ci95_low": (o["ci95"] or [None, None])[0],
                    "delta_rmse_vs_offline_future_ci95_high": (o["ci95"] or [None, None])[1]})
    return {"rows": len(rows_out), "paired_keys": len(paired_target_keys), "output": str(output)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cot-bank", type=Path, required=True)
    p.add_argument("--simvp-keys", type=Path, required=True)
    p.add_argument("--r-validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261021)
    a = p.parse_args()
    if a.bootstrap_draws < 100:
        raise ValueError("Use at least 100 date-block bootstrap draws")
    print(json.dumps(evaluate(a.cot_bank, a.simvp_keys, a.r_validation, a.output,
                              a.bootstrap_draws, a.seed), indent=2), flush=True)


if __name__ == "__main__":
    main()
