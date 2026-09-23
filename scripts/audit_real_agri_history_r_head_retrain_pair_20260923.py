"""Independent paired-row audit for old-R/new-R GHI head retraining pilot."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

STATIONS = ("Sili", "Zhujia")
GROUPS = ("A_ST", "AC_H", "AC_F", "AC_ST")
ARMS = ("original_R", "continued_R")


def metrics(pred, truth, clear):
    error = pred.astype(np.float64) * clear.astype(np.float64) - truth.astype(np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise RuntimeError("empty/nonfinite saved GHI errors")
    return {"rows": int(len(error)), "rmse_wm2": float(np.sqrt(np.mean(error ** 2))),
            "mae_wm2": float(np.mean(np.abs(error))), "bias_wm2": float(np.mean(error))}


def main(root, run, repetitions, seed, original_run=None):
    root, run = root.resolve(), run.resolve()
    original_run = (original_run or run / "original_R").resolve()
    continued_run = (run / "continued_R").resolve()
    completions = {}
    for arm, arm_run in (("original_R", original_run), ("continued_R", continued_run)):
        complete = json.loads((arm_run / "complete.json").read_text())
        if complete["state"] != "COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY" or complete["test_used"]:
            raise RuntimeError(f"incomplete or test-contaminated run: {arm}")
        completions[arm] = complete
    old_norm = completions["original_R"]["cot_normalization"]
    new_norm = completions["continued_R"]["cot_normalization"]
    if old_norm != new_norm:
        raise RuntimeError("old/new R head runs do not share an identical COT normalization")
    with (root / "data/head_pack_trainval_20260908_v1/rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 637902 or any(row["split"] not in ("train", "validation") for row in rows):
        raise RuntimeError("base row manifest drift")

    report = {"state": "PASS_REAL_AGRI_HISTORY_R_HEAD_RETRAIN_AUDIT", "test_used": False,
              "comparison": "same saved validation pack rows, GHI/clear truth, stations and leads; independent FP64 metric recomputation",
              "cot_normalization": {"identical": True, "shared_train_only_contract": old_norm,
                                    "continued_R_source": completions["continued_R"].get("cot_normalization_source")},
              "bootstrap": {"unit": "shared BJT initialization day across stations and all leads",
                            "repetitions": repetitions, "seed": seed}, "stations": {}, "equal_station_primary": {}}
    saved = {}
    for station in STATIONS:
        report["stations"][station] = {}
        for group in GROUPS:
            cells = {}
            for arm in ARMS:
                arm_run = original_run if arm == "original_R" else continued_run
                path = arm_run / f"{station}_{group}_validation_predictions.npz"
                with np.load(path, allow_pickle=False) as data:
                    cells[arm] = {key: data[key] for key in
                                  ("pack_row", "pred_kt", "observed_ghi", "clear_sky_ghi", "station", "lead")}
                item = cells[arm]
                if len(item["pack_row"]) == 0 or len(np.unique(item["pack_row"])) != len(item["pack_row"]):
                    raise RuntimeError(f"empty/duplicate validation keys: {station}/{group}/{arm}")
                for label, value in (("split", "validation"),):
                    if any(rows[int(index)][label] != value for index in item["pack_row"]):
                        raise RuntimeError(f"non-validation pack row: {station}/{group}/{arm}")
                if any(int(rows[int(index)]["station_index"]) != int(station_id) or
                       int(rows[int(index)]["lead_index"]) != int(lead)
                       for index, station_id, lead in zip(item["pack_row"], item["station"], item["lead"])):
                    raise RuntimeError(f"row-to-station/lead mismatch: {station}/{group}/{arm}")
                if not np.isfinite(item["pred_kt"]).all() or (item["pred_kt"] < 0).any():
                    raise RuntimeError(f"invalid predicted kt: {station}/{group}/{arm}")
            old, new = cells["original_R"], cells["continued_R"]
            for key in ("pack_row", "observed_ghi", "clear_sky_ghi", "station", "lead"):
                if not np.array_equal(old[key], new[key]):
                    raise RuntimeError(f"old/new R cohort mismatch for {station}/{group}/{key}")
            months = np.array([rows[int(index)]["target_time_bjt"][:7] for index in old["pack_row"]])
            days = np.array([rows[int(index)]["init_time_bjt"][:10] for index in old["pack_row"]])
            entry = {"rows": int(len(old["pack_row"])), "scenarios": {}, "new_minus_old_rmse_wm2": None,
                     "by_month": {}, "by_lead_minutes": {}}
            for arm in ARMS:
                m = metrics(cells[arm]["pred_kt"], old["observed_ghi"], old["clear_sky_ghi"])
                claimed = completions[arm]["runs"][station][group]["ghi_rmse"]
                if abs(m["rmse_wm2"] - claimed) > .001:
                    raise RuntimeError(f"trainer metric mismatch: {station}/{group}/{arm}")
                entry["scenarios"][arm] = m
                saved[(station, group, arm)] = {**cells[arm], "days": days, "months": months}
            entry["new_minus_old_rmse_wm2"] = (entry["scenarios"]["continued_R"]["rmse_wm2"] -
                                                entry["scenarios"]["original_R"]["rmse_wm2"])
            for month in sorted(set(months)):
                select = months == month
                entry["by_month"][month] = {
                    arm: metrics(cells[arm]["pred_kt"][select], old["observed_ghi"][select], old["clear_sky_ghi"][select])
                    for arm in ARMS}
            for lead in range(16):
                select = old["lead"] == lead
                entry["by_lead_minutes"][str((lead + 1) * 15)] = {
                    arm: metrics(cells[arm]["pred_kt"][select], old["observed_ghi"][select], old["clear_sky_ghi"][select])
                    for arm in ARMS}
            report["stations"][station][group] = entry

    # Unaffected groups are a deterministic placebo check: same seed and same
    # inputs should produce identical outputs when history COT is masked.
    for station in STATIONS:
        for group in ("A_ST", "AC_F"):
            old = saved[(station, group, "original_R")]["pred_kt"]
            new = saved[(station, group, "continued_R")]["pred_kt"]
            report["stations"][station][group]["unaffected_prediction_max_abs_diff"] = float(np.max(np.abs(old - new)))

    # Main paper endpoint: average station-specific RMSE equally. Resample
    # initialization dates jointly across stations, retaining all leads/day.
    rng = np.random.default_rng(seed)
    shared_days = sorted(set(saved[("Sili", "A_ST", "original_R")]["days"]) &
                         set(saved[("Zhujia", "A_ST", "original_R")]["days"]))
    if len(shared_days) < 30:
        raise RuntimeError("too few common validation initialization days")
    report["bootstrap"]["days"] = len(shared_days)
    day_lookup = {day: index for index, day in enumerate(shared_days)}
    draws = rng.integers(0, len(shared_days), size=(repetitions, len(shared_days)))
    for group in GROUPS:
        values = {arm: [] for arm in ARMS}
        for arm in ARMS:
            station_rmse = []
            for station in STATIONS:
                item = saved[(station, group, arm)]
                station_rmse.append(metrics(item["pred_kt"], item["observed_ghi"], item["clear_sky_ghi"])["rmse_wm2"])
            values[arm] = float(np.mean(station_rmse))
        day_stats = {}
        for arm in ARMS:
            day_stats[arm] = {}
            for station in STATIONS:
                item = saved[(station, group, arm)]
                err = item["pred_kt"].astype(np.float64) * item["clear_sky_ghi"].astype(np.float64) - item["observed_ghi"].astype(np.float64)
                day_sse = np.zeros(len(shared_days), dtype=np.float64)
                day_n = np.zeros(len(shared_days), dtype=np.float64)
                day_idx = np.fromiter((day_lookup.get(day, -1) for day in item["days"]),
                                      dtype=np.int64, count=len(item["days"]))
                if (day_idx < 0).any():
                    raise RuntimeError("validation rows fall outside shared bootstrap days")
                np.add.at(day_sse, day_idx, err ** 2)
                np.add.at(day_n, day_idx, 1)
                if (day_n == 0).any():
                    raise RuntimeError(f"missing station-day block for {station}/{group}/{arm}")
                day_stats[arm][station] = (day_sse, day_n)
        draw_rmse = {}
        for arm in ARMS:
            station_rmse_draws = []
            for station in STATIONS:
                day_sse, day_n = day_stats[arm][station]
                station_rmse_draws.append(np.sqrt(day_sse[draws].sum(axis=1) / day_n[draws].sum(axis=1)))
            draw_rmse[arm] = np.mean(station_rmse_draws, axis=0)
        deltas = draw_rmse["continued_R"] - draw_rmse["original_R"]
        report["equal_station_primary"][group] = {
            "original_R_mean_station_rmse_wm2": values["original_R"],
            "continued_R_mean_station_rmse_wm2": values["continued_R"],
            "continued_minus_original_wm2": values["continued_R"] - values["original_R"],
            "shared_day_bootstrap_95pct_interval_wm2": np.quantile(deltas, [.025, .975]).tolist(),
            "bootstrap_positive_fraction": float((np.asarray(deltas) > 0).mean()),
        }

    output = run / "independent_retrain_pair_audit.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite independent paired retrain audit")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"], "equal_station_primary": report["equal_station_primary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--original-run", type=Path,
                        help="existing original-R reference run; defaults to <run>/original_R")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.root, args.run, args.repetitions, args.seed, args.original_run)
