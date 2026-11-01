#!/usr/bin/env python3
"""Score the frozen Hubei station-patch COT forecaster on common val support.

CPP is used only as the retrieval reference. The score is validation-only and
does not certify independent physical COT truth or full-region skill.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LEADS = tuple(range(15, 241, 15))
SITES = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LOG_DENOM = math.log(101.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bjt(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y%m%d%H%M")


def rmse(sum_sq: float, n: int) -> float:
    return math.sqrt(sum_sq / n) if n else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cache = args.cache
    z = np.load(args.predictions, allow_pickle=False)
    pred_times = [str(x) for x in z["init_time_bjt"].tolist()]
    pred = np.asarray(z["pred_log_scaled"], dtype=np.float64)
    if pred.shape != (len(pred_times), 16, 1, 16, 16):
        raise ValueError(f"unexpected saved prediction shape {pred.shape}")
    pred_i = {t: i for i, t in enumerate(pred_times)}
    if len(pred_i) != len(pred_times):
        raise ValueError("duplicate validation prediction init keys")

    rows = read_csv(cache / "bank_rows.csv")
    bank_i: dict[dt.datetime, int] = {}
    split_by_time: dict[dt.datetime, str] = {}
    for r in rows:
        t = parse_bjt(r["time_bjt"])
        if t in bank_i:
            raise ValueError(f"duplicate bank time {t}")
        bank_i[t] = int(r["index"])
        split_by_time[t] = r["split"].lower()

    pairs = [r for r in read_csv(cache / "forecast_pairs.csv") if r["split"].lower() == "val"]
    by_init: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for r in pairs:
        init_s = r["init_time_bjt"]
        init_t, target_t = parse_bjt(init_s), parse_bjt(r["target_time_bjt"])
        lead = int(r["lead_minutes"])
        if target_t - init_t != dt.timedelta(minutes=lead) or lead not in LEADS:
            raise ValueError(f"bad key/lead in validation pair {r}")
        if r["split"].lower() != split_by_time.get(init_t) or r["split"].lower() != split_by_time.get(target_t):
            raise ValueError(f"cross-split validation pair {r}")
        if init_s not in pred_i:
            raise ValueError(f"no saved forecast for init key {init_s}")
        if lead in by_init[init_s]:
            raise ValueError(f"duplicate init/lead pair {init_s} {lead}")
        by_init[init_s][lead] = r

    arr = {n: np.load(cache / f"{n}.npy", mmap_mode="r", allow_pickle=False)
           for n in ("history_cot_log", "history_valid", "cpp_cot", "cpp_valid")}
    # Sufficient statistics keyed by station, lead, and target date enable a
    # paired date-block bootstrap without treating overlapping windows as iid.
    stats: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0.0] * 7)
    stratum: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0.0] * 8)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    pair_keys: set[tuple[str, int, int]] = set()
    target_dates: set[str] = set()
    any_valid = 0
    out_of_range_pred = 0
    thick_confusion = {"forecast": [0, 0, 0], "persistence": [0, 0, 0]}

    for init_s, lead_rows in by_init.items():
        init_t = parse_bjt(init_s)
        init_i = bank_i[init_t]
        hist_physical = np.expm1(np.asarray(arr["history_cot_log"][init_i, 0], dtype=np.float64) * LOG_DENOM)
        hist_valid = np.asarray(arr["history_valid"][init_i, 0], dtype=bool)
        for lead, row in lead_rows.items():
            target_t = parse_bjt(row["target_time_bjt"])
            target_i = bank_i[target_t]
            k = lead // 15 - 1
            truth = np.asarray(arr["cpp_cot"][target_i, 0], dtype=np.float64)
            target_valid = np.asarray(arr["cpp_valid"][target_i, 0], dtype=bool)
            pred_cot = np.expm1(pred[pred_i[init_s], k, 0] * LOG_DENOM)
            if not np.isfinite(pred_cot).all():
                raise ValueError(f"non-finite forecast COT for {init_s} +{lead}")
            out_of_range_pred += int(np.count_nonzero((pred_cot < 0) | (pred_cot > 100)))
            date = target_t.strftime("%Y%m%d")
            target_dates.add(date)
            for site, (rr, cc) in SITES.items():
                # Score only where the target reference and the persistence
                # input are both available; both methods use identical pixels.
                if not (target_valid[rr, cc] and hist_valid[rr, cc]):
                    continue
                any_valid += 1
                e_f = float(pred_cot[rr, cc] - truth[rr, cc])
                e_p = float(hist_physical[rr, cc] - truth[rr, cc])
                target_thick = bool(truth[rr, cc] >= 30.0)
                for name, estimate in (("forecast", pred_cot[rr, cc]),
                                       ("persistence", hist_physical[rr, cc])):
                    predicted_thick = bool(estimate >= 30.0)
                    if predicted_thick and target_thick:
                        thick_confusion[name][0] += 1
                    elif predicted_thick and not target_thick:
                        thick_confusion[name][1] += 1
                    elif target_thick and not predicted_thick:
                        thick_confusion[name][2] += 1
                key = (site, lead, date)
                v = stats[key]
                v[0] += e_f * e_f
                v[1] += e_p * e_p
                v[2] += abs(e_f)
                v[3] += abs(e_p)
                v[4] += 1
                v[5] += e_f
                v[6] += e_p
                counts[(site, lead)] += 1
                pair_keys.add((init_s, lead, target_i))
                tbin = "near_zero_lt1" if truth[rr, cc] < 1 else (
                    "thin_1_10" if truth[rr, cc] < 10 else (
                        "moderate_10_30" if truth[rr, cc] < 30 else "thick_ge30"))
                sv = stratum[(site, lead, tbin)]
                sv[0] += e_f * e_f
                sv[1] += e_p * e_p
                sv[2] += abs(e_f)
                sv[3] += abs(e_p)
                sv[4] += 1
                sv[5] += e_f
                sv[6] += e_p
                sv[7] += float(truth[rr, cc])

    if not stats:
        raise ValueError("empty common validation support")

    sites = list(SITES)
    leads = list(LEADS)
    per_site_lead = []
    eq_cell_rmse_f, eq_cell_rmse_p = [], []
    per_site = {s: [] for s in sites}
    per_lead = {l: [] for l in leads}
    date_cell: dict[str, dict[tuple[str, int], list[float]]] = defaultdict(dict)
    for (site, lead, date), v in sorted(stats.items()):
        date_cell[date][(site, lead)] = v
    for site in sites:
        for lead in leads:
            vals = [stats[(site, lead, d)] for d in target_dates if (site, lead, d) in stats]
            if not vals:
                continue
            sf = sum(x[0] for x in vals); sp = sum(x[1] for x in vals)
            af = sum(x[2] for x in vals); ap = sum(x[3] for x in vals)
            n = int(sum(x[4] for x in vals))
            rf, rp = rmse(sf, n), rmse(sp, n)
            eq_cell_rmse_f.append(rf); eq_cell_rmse_p.append(rp)
            per_site[site].append((rf, rp))
            per_lead[lead].append((rf, rp))
            per_site_lead.append({"station": site, "lead_minutes": lead, "n_station_pixel_pairs": n,
                                  "forecast_rmse_cot": rf, "persistence_rmse_cot": rp,
                                  "forecast_mae_cot": af / n, "persistence_mae_cot": ap / n,
                                  "forecast_bias_cot": sum(x[5] for x in vals) / n,
                                  "persistence_bias_cot": sum(x[6] for x in vals) / n,
                                  "delta_rmse_forecast_minus_persistence": rf - rp})

    score_f = float(np.mean(eq_cell_rmse_f))
    score_p = float(np.mean(eq_cell_rmse_p))
    delta = score_f - score_p
    dates = sorted(target_dates)
    rng = np.random.default_rng(args.seed)
    boot_deltas = np.empty(args.bootstrap, dtype=np.float64)
    for b in range(args.bootstrap):
        draw = rng.choice(dates, size=len(dates), replace=True)
        bf, bp = [], []
        for site in sites:
            for lead in leads:
                pieces = [date_cell[d].get((site, lead)) for d in draw]
                pieces = [x for x in pieces if x is not None]
                if not pieces:
                    continue
                n = int(sum(x[4] for x in pieces))
                if n:
                    bf.append(rmse(sum(x[0] for x in pieces), n))
                    bp.append(rmse(sum(x[1] for x in pieces), n))
        boot_deltas[b] = float(np.mean(bf) - np.mean(bp))

    strata_out = []
    for (site, lead, name), v in sorted(stratum.items()):
        n = int(v[4])
        strata_out.append({"station": site, "lead_minutes": lead, "target_cpp_cot_stratum": name,
                           "n_station_pixel_pairs": n,
                           "forecast_rmse_cot": rmse(v[0], n), "persistence_rmse_cot": rmse(v[1], n),
                           "forecast_mae_cot": v[2] / n, "persistence_mae_cot": v[3] / n,
                           "forecast_bias_cot": v[5] / n, "persistence_bias_cot": v[6] / n,
                           "target_cpp_cot_mean": v[7] / n})
    thickness_summary = {}
    for name in ("near_zero_lt1", "thin_1_10", "moderate_10_30", "thick_ge30"):
        cells = [x for x in strata_out if x["target_cpp_cot_stratum"] == name]
        if cells:
            thickness_summary[name] = {
                "equal_station_lead_forecast_rmse_cot": float(np.mean([x["forecast_rmse_cot"] for x in cells])),
                "equal_station_lead_persistence_rmse_cot": float(np.mean([x["persistence_rmse_cot"] for x in cells])),
                "equal_station_lead_forecast_bias_cot": float(np.mean([x["forecast_bias_cot"] for x in cells])),
                "equal_station_lead_persistence_bias_cot": float(np.mean([x["persistence_bias_cot"] for x in cells])),
                "equal_station_lead_target_cpp_cot_mean": float(np.mean([x["target_cpp_cot_mean"] for x in cells])),
                "station_lead_cells": len(cells),
                "station_pixel_pair_count": int(sum(x["n_station_pixel_pairs"] for x in cells)),
            }

    thick_detection = {}
    for name, (tp, fp, fn) in thick_confusion.items():
        thick_detection[name] = {
            "threshold_cpp_cot_ge30": True,
            "tp": tp, "fp": fp, "fn": fn,
            "pod": tp / (tp + fn) if tp + fn else float("nan"),
            "far": fp / (tp + fp) if tp + fp else float("nan"),
            "csi": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        }

    result = {
        "state": "VALIDATION_ONLY_STATION_PATCH_COT_FORECAST_VS_PERSISTENCE",
        "scope": "Hubei shared 16x16 station-neighborhood patch; center pixels only; CPP retrieval is reference, not independent truth.",
        "split": "monthly train/validation; test and buffer payloads closed",
        "prediction_keys": len(pred_times), "common_init_lead_pairs_scored": len(pair_keys),
        "distinct_target_dates": len(dates), "target_dates_bjt": dates,
        "station_center_pixels_row_col": SITES,
        "common_support": "CPP target valid AND initialization-time historical COT valid; identical for forecast and persistence",
        "common_station_center_pixel_sample_count": any_valid,
        "out_of_training_range_forecast_pixel_count_cot_lt0_or_gt100": out_of_range_pred,
        "equal_station_equal_lead": {
            "forecast_rmse_cot": score_f, "persistence_rmse_cot": score_p,
            "forecast_bias_cot": float(np.mean([x["forecast_bias_cot"] for x in per_site_lead])),
            "persistence_bias_cot": float(np.mean([x["persistence_bias_cot"] for x in per_site_lead])),
            "delta_rmse_forecast_minus_persistence": delta,
            "relative_rmse_reduction_percent_vs_persistence": 100.0 * (score_p - score_f) / score_p,
            "date_block_bootstrap_replicates": args.bootstrap,
            "delta_rmse_95pct_percentile_ci": np.quantile(boot_deltas, [0.025, 0.975]).tolist(),
            "bootstrap_seed": args.seed,
        },
        "by_station_equal_lead": {
            s: {"forecast_rmse_cot": float(np.mean([x[0] for x in per_site[s]])),
                "persistence_rmse_cot": float(np.mean([x[1] for x in per_site[s]]))}
            for s in sites if per_site[s]
        },
        "by_lead_equal_station": {
            str(l): {"forecast_rmse_cot": float(np.mean([x[0] for x in per_lead[l]])),
                     "persistence_rmse_cot": float(np.mean([x[1] for x in per_lead[l]]))}
            for l in leads if per_lead[l]
        },
        "one_to_four_hour": {
            str(h): {"forecast_rmse_cot": float(np.mean([x["forecast_rmse_cot"] for x in per_site_lead if x["lead_minutes"] == 60*h])),
                     "persistence_rmse_cot": float(np.mean([x["persistence_rmse_cot"] for x in per_site_lead if x["lead_minutes"] == 60*h])),
                     "n_station_lead_cells": sum(x["lead_minutes"] == 60*h for x in per_site_lead)}
            for h in (1, 2, 3, 4)
        },
        "by_station_lead": per_site_lead,
        "by_target_cpp_thickness": strata_out,
        "by_target_cpp_thickness_equal_station_lead": thickness_summary,
        "thick_cloud_detection_at_cpp_cot_ge30": thick_detection,
        "forecast_agri_to_r_comparison": "not run: no exact lineage-matched prediction bank was joined in this scoring step",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["equal_station_equal_lead"], indent=2))
    print(json.dumps(result["one_to_four_hour"], indent=2))
    print(f"saved={args.output} pairs={len(pair_keys)} dates={len(dates)} center_pixel_pairs={any_valid}")


if __name__ == "__main__":
    main()
