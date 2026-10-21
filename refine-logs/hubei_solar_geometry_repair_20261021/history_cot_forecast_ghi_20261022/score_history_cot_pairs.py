#!/usr/bin/env python3
"""FP64 common-key scoring and BJT-date paired bootstrap for history-COT GHI arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)
ARMS = ("FA_HIST", "FAC_HIST")
HOURS = ((15, 60, "0-1h"), (75, 120, "1-2h"), (135, 180, "2-3h"), (195, 240, "3-4h"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(pred, truth):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {"n": int(err.size), "rmse_wm2": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
            "mae_wm2": float(np.mean(np.abs(err), dtype=np.float64)),
            "bias_wm2": float(np.mean(err, dtype=np.float64))}


def load_pred(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def keyed(a):
    return list(zip(a["station"].astype(str), a["init_time_bjt"].astype(str),
                    a["target_time_bjt"].astype(str), a["lead_minutes"].astype(int).tolist()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261022)
    a = p.parse_args()
    orchestration = json.loads((a.training_root / "orchestrator_receipt.json").read_text(encoding="utf-8"))
    if orchestration.get("state") != "HISTORY_COT_PAIRED_TRAINING_COMPLETE" or len(orchestration.get("completed_pairs", [])) != 9:
        raise ValueError("All nine paired station/seed trainings must finish before scoring")
    pack_report = json.loads((a.pack / "pack_report.json").read_text(encoding="utf-8"))
    if pack_report.get("test_and_buffer_values_read") != 0 or pack_report.get("test_payloads_read") != 0 or pack_report.get("buffer_payloads_read") != 0:
        raise ValueError("Scoring source pack is not test/buffer closed")
    rows = list(csv.DictReader((a.pack / "rows.csv").open("r", newline="", encoding="utf-8-sig")))
    expected_support = {}
    for st in STATIONS:
        expected_support[st] = {(r["station"], r["init_time_bjt"], r["target_time_bjt"], int(r["lead_minutes"]))
                               for r in rows if r["split"] == "val" and r["station"] == st}
    all_pairs = {}
    station_seed_metrics = []
    raw_records = []
    for station in STATIONS:
        for seed in SEEDS:
            pair = a.training_root / "pairs" / f"{station}_seed{seed}"
            arm_data = {}
            for arm in ARMS:
                result = json.loads((pair / f"{arm}_result.json").read_text(encoding="utf-8"))
                if result.get("state") != "HISTORY_COT_PAIRED_ARM_COMPLETE" or result.get("test_payloads_read") != 0:
                    raise ValueError(f"Incomplete or noncausal arm result: {station}/{seed}/{arm}")
                path = pair / f"{arm}_validation_predictions.npz"
                x = load_pred(path)
                keys = keyed(x)
                if len(keys) != len(set(keys)) or set(keys) != expected_support[station]:
                    raise ValueError(f"Validation support mismatch for {station}/{seed}/{arm}")
                if x["split"].astype(str).tolist() != ["val"] * len(keys):
                    raise ValueError("Prediction bundle contains a non-validation row")
                if not all(np.isfinite(x[k]).all() for k in ("pred_ghi_wm2", "truth_ghi_wm2", "clear_sky_wm2")):
                    raise ValueError("Nonfinite saved validation prediction/label")
                arm_data[arm] = (x, keys, path)
            xa, ka, pa = arm_data[ARMS[0]]
            xc, kc, pc = arm_data[ARMS[1]]
            if ka != kc:
                raise ValueError(f"Paired arm row order differs for {station}/{seed}")
            if not np.array_equal(xa["truth_ghi_wm2"], xc["truth_ghi_wm2"]) or not np.array_equal(xa["clear_sky_wm2"], xc["clear_sky_wm2"]):
                raise ValueError(f"Paired labels or clear-sky support differ for {station}/{seed}")
            arm_met = {}
            for arm, (x, _, path) in arm_data.items():
                pred = np.asarray(x["pred_ghi_wm2"], np.float64)
                truth = np.asarray(x["truth_ghi_wm2"], np.float64)
                met = metrics(pred, truth)
                arm_met[arm] = met
                raw_records.append({"station": station, "seed": seed, "arm": arm,
                                    "n": met["n"], "rmse_wm2": met["rmse_wm2"],
                                    "mae_wm2": met["mae_wm2"], "bias_wm2": met["bias_wm2"],
                                    "prediction_sha256": sha(path)})
            d = arm_met[ARMS[1]]["rmse_wm2"] - arm_met[ARMS[0]]["rmse_wm2"]
            station_seed_metrics.append({"station": station, "seed": seed,
                                         "FA_HIST": arm_met[ARMS[0]], "FAC_HIST": arm_met[ARMS[1]],
                                         "delta_rmse_fac_minus_fa_wm2": d,
                                         "relative_improvement_pct": 100.0 * (arm_met[ARMS[0]]["rmse_wm2"] - arm_met[ARMS[1]]["rmse_wm2"]) / arm_met[ARMS[0]]["rmse_wm2"]})
            all_pairs[(station, seed)] = {arm: arm_data[arm][0] for arm in ARMS}

    # Equal-site/equal-seed point estimate: average the nine paired per-site RMSEs.
    deltas = np.asarray([x["delta_rmse_fac_minus_fa_wm2"] for x in station_seed_metrics], dtype=np.float64)
    point_delta = float(deltas.mean())
    per_arm = {arm: float(np.mean([x[arm]["rmse_wm2"] for x in station_seed_metrics])) for arm in ARMS}
    all_val = {"FA_HIST": [], "FAC_HIST": []}
    for (station, seed), pair in all_pairs.items():
        for arm in ARMS:
            all_val[arm].append((pair[arm], station, seed))
    n_dates = sorted({str(value)[:8]
                      for records in all_val.values()
                      for x, _, _ in records
                      for value in x["target_time_bjt"]})
    date_index = {d: i for i, d in enumerate(n_dates)}
    sse = np.zeros((2, len(STATIONS), len(SEEDS), len(n_dates)), dtype=np.float64)
    sae = np.zeros_like(sse)
    bias = np.zeros_like(sse)
    count = np.zeros_like(sse, dtype=np.int64)
    for ai, arm in enumerate(ARMS):
        for (station, seed), pair in all_pairs.items():
            si, ri = STATIONS.index(station), SEEDS.index(seed)
            x = pair[arm]
            err = np.asarray(x["pred_ghi_wm2"], np.float64) - np.asarray(x["truth_ghi_wm2"], np.float64)
            dates = x["target_time_bjt"].astype(str)
            ddate = np.asarray([v[:8] for v in dates])
            for day in n_dates:
                m = ddate == day
                if np.any(m):
                    di = date_index[day]
                    sse[ai, si, ri, di] = np.sum(err[m] ** 2, dtype=np.float64)
                    sae[ai, si, ri, di] = np.sum(np.abs(err[m]), dtype=np.float64)
                    bias[ai, si, ri, di] = np.sum(err[m], dtype=np.float64)
                    count[ai, si, ri, di] = int(m.sum())
    rng = np.random.default_rng(a.seed)
    boot_delta = np.empty(a.bootstrap_draws, np.float64)
    boot_improvement = np.empty(a.bootstrap_draws, np.float64)
    arm_index = {arm: i for i, arm in enumerate(ARMS)}
    for b in range(a.bootstrap_draws):
        draw = rng.integers(0, len(n_dates), size=len(n_dates))
        n = count[:, :, :, draw].sum(axis=-1)
        ss = sse[:, :, :, draw].sum(axis=-1)
        rmse = np.sqrt(ss / np.maximum(n, 1))
        # first average seeds within station, then stations equally
        eq = rmse.mean(axis=2).mean(axis=1)
        boot_delta[b] = eq[arm_index["FAC_HIST"]] - eq[arm_index["FA_HIST"]]
        boot_improvement[b] = 100.0 * (eq[arm_index["FA_HIST"]] - eq[arm_index["FAC_HIST"]]) / eq[arm_index["FA_HIST"]]
    global_summary = {"equal_station_equal_seed_mean_rmse_wm2": per_arm,
                      "paired_delta_rmse_fac_minus_fa_wm2": point_delta,
                      "relative_rmse_improvement_pct": 100.0 * (per_arm["FA_HIST"] - per_arm["FAC_HIST"]) / per_arm["FA_HIST"],
                      "paired_bjt_date_block_bootstrap_95pct_delta_ci_wm2": [float(x) for x in np.quantile(boot_delta, [0.025, 0.975])],
                      "paired_bjt_date_block_bootstrap_95pct_improvement_ci_pct": [float(x) for x in np.quantile(boot_improvement, [0.025, 0.975])],
                      "bootstrap_draws": a.bootstrap_draws, "bootstrap_blocks": n_dates,
                      "bootstrap_unit": "BJT target date; resample dates jointly across station and seed; equal seeds within station then equal stations"}

    lead_results = []
    hour_results = []
    grouping = [(int(r["lead_minutes"]), str(r["lead_minutes"])) for r in rows if r["split"] == "val"]
    leads = sorted({x[0] for x in grouping})
    for lead in leads:
        local = []
        for (station, seed), pair in all_pairs.items():
            vals = {}
            for arm in ARMS:
                x = pair[arm]
                m = x["lead_minutes"].astype(int) == lead
                vals[arm] = metrics(x["pred_ghi_wm2"][m], x["truth_ghi_wm2"][m])
            local.append((station, seed, vals))
        lead_results.append({"lead_minutes": lead,
                             "FA_HIST_equal_station_seed_rmse_wm2": float(np.mean([z[2]["FA_HIST"]["rmse_wm2"] for z in local])),
                             "FAC_HIST_equal_station_seed_rmse_wm2": float(np.mean([z[2]["FAC_HIST"]["rmse_wm2"] for z in local])),
                             "delta_fac_minus_fa_wm2": float(np.mean([z[2]["FAC_HIST"]["rmse_wm2"] - z[2]["FA_HIST"]["rmse_wm2"] for z in local])),
                             "n_per_site_seed_min": int(min(z[2]["FA_HIST"]["n"] for z in local))})
    for lo, hi, name in HOURS:
        local = []
        for (station, seed), pair in all_pairs.items():
            vals = {}
            for arm in ARMS:
                x = pair[arm]
                m = (x["lead_minutes"].astype(int) >= lo) & (x["lead_minutes"].astype(int) <= hi)
                vals[arm] = metrics(x["pred_ghi_wm2"][m], x["truth_ghi_wm2"][m])
            local.append((station, seed, vals))
        hour_results.append({"hour_bin": name, "lead_min": lo, "lead_max": hi,
                             "FA_HIST_equal_station_seed_rmse_wm2": float(np.mean([z[2]["FA_HIST"]["rmse_wm2"] for z in local])),
                             "FAC_HIST_equal_station_seed_rmse_wm2": float(np.mean([z[2]["FAC_HIST"]["rmse_wm2"] for z in local])),
                             "delta_fac_minus_fa_wm2": float(np.mean([z[2]["FAC_HIST"]["rmse_wm2"] - z[2]["FA_HIST"]["rmse_wm2"] for z in local])),
                             "n_per_site_seed_min": int(min(z[2]["FA_HIST"]["n"] for z in local))})

    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "station_seed_metrics.json").write_text(json.dumps(station_seed_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (a.output / "global_summary.json").write_text(json.dumps(global_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, items in (("by_lead.csv", lead_results), ("by_hour.csv", hour_results), ("arm_metrics.csv", raw_records)):
        with (a.output / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0]) if items else ["empty"])
            w.writeheader(); w.writerows(items)
    conclusion = "validation_point_improvement_date_interval_excludes_zero" if global_summary["paired_bjt_date_block_bootstrap_95pct_delta_ci_wm2"][1] < 0 else (
        "validation_point_regression_date_interval_excludes_zero" if global_summary["paired_bjt_date_block_bootstrap_95pct_delta_ci_wm2"][0] > 0 else "validation_uncertain_date_interval_crosses_zero")
    receipt = {"state": "HISTORY_COT_GHI_VALIDATION_FP64_SCORED",
               "conclusion_label": conclusion, "global_summary": global_summary,
               "station_seed_metrics": station_seed_metrics, "by_lead": lead_results, "by_hour": hour_results,
               "source_training_receipt_sha256": sha(a.training_root / "orchestrator_receipt.json"),
               "pack_report_sha256": sha(a.pack / "pack_report.json"),
               "prediction_hashes_verified": True,
               "validation_only": True, "test_payloads_read": 0, "buffer_payloads_read": 0}
    (a.output / "score_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": receipt["state"], "conclusion": conclusion,
                      "global": global_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
