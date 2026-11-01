#!/usr/bin/env python3
"""Paired FP64 validation scoring for attempt13 forecast-COT banks.

Scores the MSE-control and COT-auxiliary SimVP forecasts, exact-history COT
persistence, and the offline observed-future-AGRI->R diagnostic on the same
CPP-supported pixels. CPP remains a retrieval reference, not independent
truth. Test and buffer inputs are not accepted by this scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


METHODS = ("mse_control", "cot_aux", "history_persistence", "offline_future_agri_to_R")
LEADS = tuple(range(15, 241, 15))
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
THICKNESS = (("near_zero", 0.0, 1.0), ("thin", 1.0, 10.0),
             ("moderate", 10.0, 30.0), ("thick_ge30", 30.0, float("inf")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stamp(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 12:
        raise ValueError(f"Invalid BJT time key: {value!r}")
    return digits[:12]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def scopes():
    result = [("all_patch", "ALL_PATCH", (slice(0, 16), slice(0, 16)))]
    for name, (y, x) in CENTERS.items():
        result.append(("station_pixel", name, (slice(y, y + 1), slice(x, x + 1))))
        result.append(("station_3x3", name, (slice(y - 1, y + 2), slice(x - 1, x + 2))))
    return result


def bootstrap_macro(date_stats, left: str, right: str, draws: int, seed: int):
    dates = sorted(date_stats)
    if len(dates) < 2:
        return {"state": "INSUFFICIENT_DATE_BLOCKS", "n_dates": len(dates), "ci95": None}
    # date_stats[date] = arrays [16, n_methods] for SSE and count
    sse = np.stack([date_stats[d][0] for d in dates]).astype(np.float64)
    count = np.stack([date_stats[d][1] for d in dates]).astype(np.float64)
    if not np.array_equal(count[:, :, 0], count[:, :, 1]) or not np.array_equal(count[:, :, 0], count[:, :, 2]) or not np.array_equal(count[:, :, 0], count[:, :, 3]):
        raise RuntimeError("Paired methods do not have identical date/lead sample support")

    def score(ss, nn, method_i):
        n_by_lead = nn[:, method_i]
        s_by_lead = ss[:, method_i]
        present = n_by_lead > 0
        if not np.any(present):
            return np.nan
        return float(np.mean(np.sqrt(s_by_lead[present] / n_by_lead[present])))

    point_left = score(sse.sum(axis=0), count.sum(axis=0), METHODS.index(left))
    point_right = score(sse.sum(axis=0), count.sum(axis=0), METHODS.index(right))
    delta = point_left - point_right
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(dates), size=(draws, len(dates)))
    boot = np.empty(draws, dtype=np.float64)
    li, ri = METHODS.index(left), METHODS.index(right)
    for j, idx in enumerate(sampled):
        ss = sse[idx].sum(axis=0)
        nn = count[idx].sum(axis=0)
        boot[j] = score(ss, nn, li) - score(ss, nn, ri)
    lo, hi = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
    return {"state": "PAIRED_BJT_DATE_BLOCK_BOOTSTRAP", "n_dates": len(dates),
            "delta_rmse_left_minus_right": float(delta), "ci95": [float(lo), float(hi)],
            "bootstrap_draws": draws, "aggregation": "mean of physical-COT RMSE across leads; dates resampled as blocks"}


def bootstrap_station_macro(date_stats_by_station, left: str, right: str, draws: int, seed: int):
    """Equal-station, equal-lead RMSE difference on dates shared by all sites."""
    stations = ("Niushou", "Wangzhai", "Mizhuang")
    date_sets = [set(date_stats_by_station.get(st, {})) for st in stations]
    common_dates = sorted(set.intersection(*date_sets)) if all(date_sets) else []
    if len(common_dates) < 2:
        return {"state": "INSUFFICIENT_COMMON_DATE_BLOCKS", "n_common_dates": len(common_dates), "ci95": None}
    li, ri = METHODS.index(left), METHODS.index(right)

    def score_for(station, date_indices, method_i):
        sse = np.zeros((16,), dtype=np.float64)
        count = np.zeros((16,), dtype=np.float64)
        blocks = date_stats_by_station[station]
        for idx in date_indices:
            ss, nn = blocks[common_dates[int(idx)]]
            sse += ss[:, method_i]
            count += nn[:, method_i]
        present = count > 0
        if not present.any():
            return np.nan
        return float(np.mean(np.sqrt(sse[present] / count[present])))

    all_idx = np.arange(len(common_dates), dtype=np.int64)
    station_scores = {
        st: (score_for(st, all_idx, li), score_for(st, all_idx, ri)) for st in stations
    }
    point = float(np.mean([l - r for l, r in station_scores.values()]))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(common_dates), size=(draws, len(common_dates)))
    boot = np.empty(draws, dtype=np.float64)
    for j, idx in enumerate(sampled):
        diffs = []
        for st in stations:
            lscore, rscore = score_for(st, idx, li), score_for(st, idx, ri)
            if np.isfinite(lscore) and np.isfinite(rscore):
                diffs.append(lscore - rscore)
        boot[j] = float(np.mean(diffs)) if diffs else np.nan
    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return {"state": "NO_VALID_BOOTSTRAP_DRAWS", "n_common_dates": len(common_dates), "ci95": None}
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {
        "state": "PAIRED_BJT_DATE_BLOCK_BOOTSTRAP_EQUAL_STATION_EQUAL_LEAD",
        "stations": list(stations), "n_common_dates": len(common_dates),
        "station_rmse_left_right": {st: [float(v[0]), float(v[1])] for st, v in station_scores.items()},
        "delta_rmse_left_minus_right": point, "ci95": [float(lo), float(hi)],
        "bootstrap_draws": draws,
        "aggregation": "common BJT dates resampled jointly; RMSE averaged equally over leads and three stations",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mse-cot-bank", type=Path, required=True)
    p.add_argument("--aux-cot-bank", type=Path, required=True)
    p.add_argument("--mse-keys", type=Path, required=True)
    p.add_argument("--aux-keys", type=Path, required=True)
    p.add_argument("--r-validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261021)
    a = p.parse_args()
    if a.bootstrap_draws < 100:
        raise ValueError("Use at least 100 paired date-block bootstrap draws")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    required = (a.mse_keys, a.aux_keys, a.r_validation / "val_keys.csv",
                a.r_validation / "val_prediction_cot.npy",
                a.r_validation / "val_reference_cpp_cot.npy",
                a.r_validation / "val_valid_mask.npy")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    banks = [np.load(path, mmap_mode="r", allow_pickle=False)
             for path in (a.mse_cot_bank, a.aux_cot_bank)]
    if any(z.shape != (407, 16, 1, 16, 16) for z in banks):
        raise ValueError(f"Unexpected attempt13 COT bank shapes: {[z.shape for z in banks]}")
    s_rows = read_csv(a.mse_keys)
    aux_rows = read_csv(a.aux_keys)
    if len(s_rows) != 407 or any(row.get("split") != "val" for row in s_rows):
        raise ValueError("Expected exactly 407 validation-only sequence keys")
    if aux_rows != s_rows:
        raise RuntimeError("MSE and COT-aux SimVP validation keys or row order differ")
    bank_meta_paths = [Path(str(path) + ".json") for path in (a.mse_cot_bank, a.aux_cot_bank)]
    bank_metas = []
    for bank_meta, keys_path in zip(bank_meta_paths, (a.mse_keys, a.aux_keys)):
        if not bank_meta.is_file():
            raise FileNotFoundError(f"Missing COT inference receipt: {bank_meta}")
        meta = json.loads(bank_meta.read_text(encoding="utf-8"))
        if (meta.get("state") != "ATTEMPT13_VALIDATION_FORECAST_PHYSICAL_COT_BANK" or
                meta.get("test_payloads_read") != 0 or meta.get("buffer_payloads_read") != 0 or
                meta.get("simvp_keys_sha256") != sha256(keys_path)):
            raise RuntimeError(f"COT bank receipt does not bind to the supplied validation keys: {bank_meta}")
        bank_metas.append(meta)
    for field in ("r_source_sha256", "r_checkpoint_sha256", "r_normalization_sha256"):
        if bank_metas[0].get(field) != bank_metas[1].get(field):
            raise RuntimeError(f"MSE and COT-aux COT banks used different frozen R provenance: {field}")
    for path, bank in zip((a.mse_cot_bank, a.aux_cot_bank), banks):
        if not np.isfinite(bank).all() or np.any(bank < 0) or np.any(bank > 100.001):
            raise ValueError(f"Forecast COT contains non-finite or non-physical values: {path}")
    for row in s_rows:
        if len(row.get("target_times_bjt", "").split("|")) != 16:
            raise ValueError("Every validation sequence must have 16 target timestamps")

    r_keys = read_csv(a.r_validation / "val_keys.csv")
    target_to_r = {}
    for i, row in enumerate(r_keys):
        if row.get("split") != "val":
            raise ValueError("R validation key table contains a non-validation row")
        key = stamp(row["time_bjt"])
        if key in target_to_r:
            raise ValueError(f"Duplicate R validation timestamp: {key}")
        target_to_r[key] = i
    r_pred = np.load(a.r_validation / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(a.r_validation / "val_reference_cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(a.r_validation / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    if r_pred.shape != cpp.shape or cpp.shape != valid.shape or cpp.shape[1:] != (1, 16, 16) or len(target_to_r) != len(cpp):
        raise ValueError("R prediction, CPP, mask and key arrays disagree")

    method_order = METHODS
    accum = defaultdict(lambda: {m: [0.0, 0.0, 0.0, 0] for m in method_order})
    # date_stats[scope][date] -> [SSE, count], each [16,4].
    date_stats = defaultdict(dict)
    classification = defaultdict(lambda: {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "nearzero_fp": 0, "nearzero_n": 0} for m in method_order})
    paired_keys = set()
    skipped_target = skipped_history = 0
    scope_list = scopes()

    for si, row in enumerate(s_rows):
        init = stamp(row["init_time_bjt"])
        init_dt = datetime.strptime(init, "%Y%m%d%H%M")
        targets = [stamp(x) for x in row["target_times_bjt"].split("|")]
        for li, lead in enumerate(LEADS):
            target = targets[li]
            expected = (init_dt + timedelta(minutes=lead)).strftime("%Y%m%d%H%M")
            if target != expected:
                raise ValueError(f"Lead/key mismatch at sample={si}: {init}+{lead} -> {target}, expected {expected}")
            ri = target_to_r.get(target)
            hi = target_to_r.get(init)
            if ri is None:
                skipped_target += 1
                continue
            if hi is None:
                skipped_history += 1
                continue
            forecasts = [np.asarray(banks[0][si, li, 0], dtype=np.float64),
                         np.asarray(banks[1][si, li, 0], dtype=np.float64)]
            persistence = np.asarray(r_pred[hi, 0], dtype=np.float64)
            offline = np.asarray(r_pred[ri, 0], dtype=np.float64)
            reference = np.asarray(cpp[ri, 0], dtype=np.float64)
            common = np.array(valid[ri, 0], dtype=bool, copy=True)
            for z in (*forecasts, persistence, offline, reference):
                common &= np.isfinite(z)
            if not common.any():
                continue
            paired_keys.add((init, target, lead))
            date = target[:8]
            for scope_kind, scope_name, (ys, xs) in scope_list:
                y = reference[ys, xs]
                m = common[ys, xs]
                preds = [forecasts[0][ys, xs], forecasts[1][ys, xs],
                         persistence[ys, xs], offline[ys, xs]]
                if not m.any():
                    continue
                scope_key = (scope_kind, scope_name)
                if date not in date_stats[scope_key]:
                    date_stats[scope_key][date] = [np.zeros((16, 4), dtype=np.float64),
                                                   np.zeros((16, 4), dtype=np.float64)]
                d_sse, d_count = date_stats[scope_key][date]
                for stratum, low, high in (("all", -float("inf"), float("inf")), *THICKNESS):
                    sm = m & (y >= low) & (y < high)
                    if not sm.any():
                        continue
                    key = (scope_kind, scope_name, li, stratum)
                    for mi, method in enumerate(method_order):
                        err = preds[mi][sm] - y[sm]
                        cell = accum[key][method]
                        cell[0] += float(np.square(err).sum(dtype=np.float64))
                        cell[1] += float(np.abs(err).sum(dtype=np.float64))
                        cell[2] += float(err.sum(dtype=np.float64))
                        cell[3] += int(err.size)
                        if stratum == "all":
                            d_sse[li, mi] += float(np.square(err).sum(dtype=np.float64))
                            d_count[li, mi] += int(err.size)
                thick_ref = y >= 30.0
                nearzero_ref = y < 1.0
                for mi, method in enumerate(method_order):
                    pred = preds[mi]
                    cm = classification[scope_key][method]
                    cm["tp"] += int(np.sum(m & thick_ref & (pred >= 30.0)))
                    cm["fp"] += int(np.sum(m & ~thick_ref & (pred >= 30.0)))
                    cm["fn"] += int(np.sum(m & thick_ref & (pred < 30.0)))
                    cm["tn"] += int(np.sum(m & ~thick_ref & (pred < 30.0)))
                    cm["nearzero_fp"] += int(np.sum(m & nearzero_ref & (pred >= 1.0)))
                    cm["nearzero_n"] += int(np.sum(m & nearzero_ref))

    if not paired_keys:
        raise RuntimeError("No forecast/init target pairs have common CPP support")
    rows = []
    for (scope_kind, scope_name, li, stratum), methods in sorted(accum.items()):
        row = {"scope": scope_kind, "station": scope_name, "lead_minutes": LEADS[li], "cpp_cot_stratum": stratum}
        for method, (sse, sae, bias_sum, n) in methods.items():
            row[f"{method}_n"] = n
            row[f"{method}_rmse"] = float(np.sqrt(sse / n)) if n else None
            row[f"{method}_mae"] = float(sae / n) if n else None
            row[f"{method}_bias"] = float(bias_sum / n) if n else None
        rows.append(row)

    ci = {}
    for (scope_kind, scope_name), stats in date_stats.items():
        if scope_kind not in ("all_patch", "station_3x3"):
            continue
        ci[f"{scope_kind}|{scope_name}"] = {
            "cot_aux_minus_mse": bootstrap_macro(stats, "cot_aux", "mse_control", a.bootstrap_draws, a.seed),
            "cot_aux_minus_history_persistence": bootstrap_macro(stats, "cot_aux", "history_persistence", a.bootstrap_draws, a.seed + 1),
        }
    site_macro_ci = {
        "station_3x3": {
            "cot_aux_minus_mse": bootstrap_station_macro(
                {st: date_stats.get(("station_3x3", st), {}) for st in ("Niushou", "Wangzhai", "Mizhuang")},
                "cot_aux", "mse_control", a.bootstrap_draws, a.seed + 2),
            "cot_aux_minus_history_persistence": bootstrap_station_macro(
                {st: date_stats.get(("station_3x3", st), {}) for st in ("Niushou", "Wangzhai", "Mizhuang")},
                "cot_aux", "history_persistence", a.bootstrap_draws, a.seed + 3),
        }
    }
    class_out = {}
    for (scope_kind, scope_name), by_method in classification.items():
        scope_label = f"{scope_kind}|{scope_name}"
        class_out[scope_label] = {}
        for method, cm in by_method.items():
            tp, fp, fn, tn = (cm[k] for k in ("tp", "fp", "fn", "tn"))
            class_out[scope_label][method] = {
                **cm,
                "thick_cloud_pod": tp / (tp + fn) if tp + fn else None,
                "thick_cloud_far": fp / (tp + fp) if tp + fp else None,
                "thick_cloud_csi": tp / (tp + fp + fn) if tp + fp + fn else None,
                "nearzero_false_alarm_rate": cm["nearzero_fp"] / cm["nearzero_n"] if cm["nearzero_n"] else None,
            }

    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / "lead_stratum_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "state": "ATTEMPT13_PAIRED_VALIDATION_FORECAST_COT_FP64",
        "scope": "Hubei train/validation contract only; three stations; canonical +15..+240 min",
        "reference_warning": "CPP is a satellite retrieval reference, not independent COT truth.",
        "aggregation": "physical COT; exact common CPP mask; FP64 metric accumulation; per-site and equal-station/equal-lead BJT date-block bootstrap",
        "paired_init_target_lead_keys": len(paired_keys),
        "target_reference_missing": skipped_target,
        "history_reference_missing": skipped_history,
        "paired_date_block_ci_equal_lead_macro": ci,
        "paired_date_block_ci_equal_station_equal_lead": site_macro_ci,
        "classification_guards": class_out,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "sources_sha256": {
            "mse_cot_bank": sha256(a.mse_cot_bank), "aux_cot_bank": sha256(a.aux_cot_bank),
            "mse_keys": sha256(a.mse_keys), "aux_keys": sha256(a.aux_keys),
            "mse_bank_receipt": sha256(bank_meta_paths[0]),
            "aux_bank_receipt": sha256(bank_meta_paths[1]),
            "r_val_keys": sha256(a.r_validation / "val_keys.csv"),
            "r_val_prediction": sha256(a.r_validation / "val_prediction_cot.npy"),
            "cpp_reference": sha256(a.r_validation / "val_reference_cpp_cot.npy"),
            "cpp_mask": sha256(a.r_validation / "val_valid_mask.npy"),
        },
        "metrics_csv": "lead_stratum_metrics.csv",
        "per_lead_metrics": rows,
    }
    (a.output / "summary_fp64.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "paired_init_target_lead_keys": len(paired_keys),
                      "output": str(a.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
