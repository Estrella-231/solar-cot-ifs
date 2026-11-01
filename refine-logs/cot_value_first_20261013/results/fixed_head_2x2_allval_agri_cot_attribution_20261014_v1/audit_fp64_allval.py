import csv, json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

root = Path(__file__).parent
rows = list(csv.DictReader((root / "paired_predictions.csv").open(newline="", encoding="utf-8")))
assert len(rows) == 66590, len(rows)
assert len({(r["station"], r["sequence_id"], r["lead_min"], r["target_time_utc"]) for r in rows}) == len(rows)
assert all(r["station"] in {"sili", "zhujia"} for r in rows)
assert all(int(r["lead_min"]) in range(15, 241, 15) for r in rows)
arms = {k: f"pred_{k}_wm2" for k in ("TT", "FT", "TF", "FF")}
y = np.asarray([float(r["ghi_obs_wm2"]) for r in rows], dtype=np.float64)
pred = {k: np.asarray([float(r[col]) for r in rows], dtype=np.float64) for k, col in arms.items()}
assert np.isfinite(y).all() and all(np.isfinite(x).all() for x in pred.values())
stations = np.asarray([r["station"] for r in rows])
dates = np.asarray([
    (datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d")
    for r in rows
])
unique_dates = np.unique(dates)
contrasts = {
    "AGRI_effect_at_same_time_COT": ("FT", "TT"),
    "COT_effect_at_true_AGRI": ("TF", "TT"),
    "COT_effect_at_forecast_AGRI": ("FF", "FT"),
    "AGRI_effect_at_forecast_COT": ("FF", "TF"),
    "joint_forecast_input_effect": ("FF", "TT"),
}
by_station = {}
daily_sse = {s: [] for s in ("sili", "zhujia")}
for s in ("sili", "zhujia"):
    mask = stations == s
    by_station[s] = {"n_instances": int(mask.sum()), "unique_sequences": len({r["sequence_id"] for r in rows if r["station"] == s}),
                     "unique_target_times": len({r["target_time_utc"] for r in rows if r["station"] == s}),
                     "arms": {}}
    for k in arms:
        err = pred[k][mask] - y[mask]
        by_station[s]["arms"][k] = {"RMSE_Wm2": float(np.sqrt(np.mean(err * err)),), "MAE_Wm2": float(np.mean(np.abs(err)))}
    for day in unique_dates:
        m = mask & (dates == day)
        daily_sse[s].append({"n": int(m.sum()), **{k: float(np.sum((pred[k][m] - y[m]) ** 2)) for k in arms}})

point = {name: float(np.mean([by_station[s]["arms"][new]["RMSE_Wm2"] - by_station[s]["arms"][base]["RMSE_Wm2"] for s in ("sili", "zhujia")]))
         for name, (new, base) in contrasts.items()}
def station_rmse(weights, arm, station):
    chunks = daily_sse[station]
    counts = np.asarray([c["n"] for c in chunks], dtype=np.float64)
    sums = np.asarray([c[arm] for c in chunks], dtype=np.float64)
    denom = float(weights @ counts)
    return float(np.sqrt((weights @ sums) / denom)) if denom else np.nan

rng = np.random.default_rng(20261014)
B = 5000
weights = rng.multinomial(len(unique_dates), np.full(len(unique_dates), 1 / len(unique_dates)), size=B)
ci = {}
for name, (new, base) in contrasts.items():
    samples = []
    for w in weights:
        samples.append(float(np.mean([station_rmse(w, new, s) - station_rmse(w, base, s) for s in ("sili", "zhujia")])))
    ci[name] = [float(v) for v in np.quantile(samples, [0.025, 0.975])]

lead_rows = []
for lead in sorted({int(r["lead_min"]) for r in rows}):
    for station in ("sili", "zhujia"):
        mask = (stations == station) & np.asarray([int(r["lead_min"]) == lead for r in rows])
        q = {k: float(np.sqrt(np.mean((pred[k][mask] - y[mask]) ** 2))) for k in arms}
        lead_rows.append({"station": station, "lead_min": lead, "n": int(mask.sum()), **{f"RMSE_{k}_Wm2": v for k, v in q.items()},
                          "delta_FF_minus_TT_Wm2": q["FF"] - q["TT"]})
with (root / "lead_metrics_fp64.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(lead_rows[0])); w.writeheader(); w.writerows(lead_rows)
summary = {
    "state": "INDEPENDENT_FP64_RECOMPUTE_PASS", "scope": "all monthly-split validation forecast instances; fixed seed42 AC head; offline 2x2 input attribution",
    "test_payload_opened": False, "instances": len(rows), "unique_sequences": len({r["sequence_id"] for r in rows}),
    "unique_BJT_target_date_blocks": len(unique_dates), "BJT_dates": list(unique_dates),
    "station_equal_RMSE_Wm2_by_arm": {k: float(np.mean([by_station[s]["arms"][k]["RMSE_Wm2"] for s in ("sili", "zhujia")])) for k in arms},
    "station_equal_delta_RMSE_Wm2_by_contrast": point, "date_block_paired_bootstrap_95CI_Wm2": ci,
    "bootstrap_replicates": B, "bootstrap_unit": "BJT target-date blocks; same resample applied to both stations and all arms",
    "by_station": by_station,
    "interpretation": "TT is an offline same-time oracle reference; FF uses forecast AGRI and its frozen-R COT, but the AC head was trained with actual AGRI and same-time R-COT. These are fixed-head input-shift diagnostics, not operational GHI forecast scores. Overlapping windows remain correlated within dates; date-block intervals address date clustering but do not make overlapping sequences independent.",
}
(root / "fp64_independent_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
