import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RDIR = ROOT / "refine-logs/cot_value_first_20261013"
OUT = RDIR / "results/forecast_cot_val_bank_20261014_v1"
NPZ = OUT / "forecast_cot_val_bank.npz"
PROFILE = json.loads((OUT / "profile.json").read_text(encoding="utf-8"))
Z = np.load(NPZ, allow_pickle=False)
STATIONS = ("sili", "zhujia")
METHODS = ("forecast_simvp_to_R", "persistence_R", "offline_true_future_AGRI_to_R")
LOG_KEYS = ("forecast_log1p_cot", "persistence_log1p_cot", "target_agri_r_log1p_cot")

assert PROFILE["state"] == "COMPLETE_VALIDATION_S_TO_R_COT_BANK"
assert PROFILE["split"] == "val_only" and PROFILE["test_payloads_read"] == 0
assert len(Z["sequence_ids"]) == PROFILE["windows_completed"] == 271
shape = Z[LOG_KEYS[0]].shape
assert shape == (271, 2, 16, 16, 16)
assert all(Z[k].shape == shape for k in (*LOG_KEYS, "cpp_reference_cot", "cpp_valid_mask"))
assert np.isfinite(Z["cpp_reference_cot"][Z["cpp_valid_mask"]]).all()
assert all(np.isfinite(Z[k]).all() for k in LOG_KEYS)
assert int(Z["cpp_valid_mask"].sum()) == PROFILE["cpp_common_support_pixels"] == 550400
archive_sha = hashlib.sha256(NPZ.read_bytes()).hexdigest()
forecast_array_sha = hashlib.sha256(np.ascontiguousarray(Z["forecast_log1p_cot"]).tobytes()).hexdigest()
if forecast_array_sha != PROFILE["forecast_output_sha256"]:
    raise ValueError(f"forecast array SHA mismatch: {forecast_array_sha} != {PROFILE['forecast_output_sha256']}")

mask = Z["cpp_valid_mask"].astype(bool)
ref = Z["cpp_reference_cot"].astype(np.float64)
preds = [np.expm1(Z[k].astype(np.float64)) for k in LOG_KEYS]
target_r = preds[2]
times = Z["target_time_utc"].astype(str)
if times.shape != (shape[0], shape[2]):
    raise ValueError(f"unexpected timestamp shape {times.shape}")
if not np.all(np.isfinite(ref[mask])):
    raise ValueError("nonfinite CPP reference under common support")

def rmse(a, b, m):
    e = a[m] - b[m]
    return float(np.sqrt(np.mean(e * e, dtype=np.float64))) if e.size else None

def mae(a, b, m):
    return float(np.mean(np.abs(a[m] - b[m]), dtype=np.float64)) if m.any() else None

def bias(a, b, m):
    return float(np.mean(a[m] - b[m], dtype=np.float64)) if m.any() else None

lead_rows = []
for si, station in enumerate(STATIONS):
    for li in range(16):
        m = mask[:, si, li]
        row = {"station": station, "lead_min": (li + 1) * 15, "n_pixels": int(m.sum())}
        for name, x in zip(METHODS, preds):
            row[name + "_rmse_cot"] = rmse(x[:, si, li], ref[:, si, li], m)
            row[name + "_mae_cot"] = mae(x[:, si, li], ref[:, si, li], m)
            row[name + "_bias_cot"] = bias(x[:, si, li], ref[:, si, li], m)
            row[name + "_rmse_log1p_cot"] = rmse(Z[LOG_KEYS[METHODS.index(name)]][:, si, li].astype(np.float64), np.log1p(ref[:, si, li]), m)
        # Forecast-to-offline observed-AGRI/R agreement isolates the AGRI forecast-chain perturbation.
        row["forecast_vs_offline_true_agri_R_rmse_cot"] = rmse(preds[0][:, si, li], target_r[:, si, li], m)
        lead_rows.append(row)

station_rows = []
for si, station in enumerate(STATIONS):
    m = mask[:, si]
    row = {"station": station, "n_pixels": int(m.sum()), "windows_with_cpp": int(mask[:,si].any(axis=(1,2,3)).sum())}
    for name, x in zip(METHODS, preds):
        row[name + "_rmse_cot"] = rmse(x[:, si], ref[:, si], m)
        row[name + "_mae_cot"] = mae(x[:, si], ref[:, si], m)
        row[name + "_bias_cot"] = bias(x[:, si], ref[:, si], m)
        row[name + "_rmse_log1p_cot"] = rmse(Z[LOG_KEYS[METHODS.index(name)]][:, si].astype(np.float64), np.log1p(ref[:, si]), m)
    row["forecast_vs_offline_true_agri_R_rmse_cot"] = rmse(preds[0][:,si], target_r[:,si], m)
    station_rows.append(row)

# Paired 2-day BJT block bootstrap for station-equal physical RMSE differences.
date_labels = np.empty((shape[0], 16), dtype=object)
for i in range(shape[0]):
    for li in range(16):
        t = dt.datetime.strptime(times[i, li], "%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=8)
        date_labels[i, li] = t.date().isoformat()
unique_days = sorted(set(date_labels.ravel()))
ord_by_day = {d: dt.date.fromisoformat(d).toordinal() for d in unique_days}
block_of = {d: (ord_by_day[d] - ord_by_day[unique_days[0]]) // 2 for d in unique_days}
blocks = sorted(set(block_of.values()))
block_sse = {m: np.zeros((len(blocks), 2), dtype=np.float64) for m in METHODS}
block_n = np.zeros((len(blocks), 2), dtype=np.int64)
for bi, block in enumerate(blocks):
    for i in range(shape[0]):
        for li in range(16):
            if block_of[date_labels[i, li]] != block:
                continue
            for si in range(2):
                m = mask[i, si, li]
                n = int(m.sum())
                if not n:
                    continue
                block_n[bi, si] += n
                for name, x in zip(METHODS, preds):
                    e = x[i, si, li][m] - ref[i, si, li][m]
                    block_sse[name][bi, si] += np.sum(e * e, dtype=np.float64)

rng = np.random.default_rng(20261015)
B = 2000
comparisons = (("forecast_simvp_to_R", "offline_true_future_AGRI_to_R"),
               ("forecast_simvp_to_R", "persistence_R"))
boot = {f"{a}_minus_{b}": [] for a,b in comparisons}
point = {}
for a,b in comparisons:
    diffs = []
    for si in range(2):
        ra = station_rows[si][a + "_rmse_cot"]
        rb = station_rows[si][b + "_rmse_cot"]
        diffs.append(ra-rb)
    point[f"{a}_minus_{b}"] = float(np.mean(diffs))
for _ in range(B):
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    n = block_n[chosen].sum(axis=0)
    for a,b in comparisons:
        sa = block_sse[a][chosen].sum(axis=0)
        sb = block_sse[b][chosen].sum(axis=0)
        ra = np.sqrt(sa / n)
        rb = np.sqrt(sb / n)
        boot[f"{a}_minus_{b}"].append(float(np.mean(ra-rb)))
cis = {k: [float(x) for x in np.quantile(v, [0.025,0.975])] for k,v in boot.items()}

support_by_station = {s: int(mask[:,i].sum()) for i,s in enumerate(STATIONS)}
result = {"state":"FP64_INDEPENDENT_RECOMPUTE_PASS", "split":"validation_only", "test_payloads_read":0,
    "bank_npz_sha256":archive_sha, "forecast_array_sha256":forecast_array_sha,
    "profile_forecast_array_sha256":PROFILE["forecast_output_sha256"],
    "simvp_checkpoint_sha256":PROFILE["simvp_checkpoint_sha256"], "R_checkpoint_sha256":PROFILE["r_checkpoint_sha256"],
    "sequence_count":shape[0], "station_order":list(STATIONS), "lead_minutes":[15*(i+1) for i in range(16)],
    "cpp_common_support_pixels":int(mask.sum()), "support_pixels_by_station":support_by_station,
    "unique_target_bjt_dates":len(unique_days), "two_day_blocks":len(blocks), "bootstrap_replicates":B,
    "bootstrap_seed":20261015, "aggregate_station_equal_rmse_differences_cot":point,
    "paired_two_day_block_95pct_ci":cis, "station_metrics":station_rows, "lead_metrics":lead_rows,
    "metric_note":"All scores use common CPP-valid support. CPP is a retrieval reference, not independent truth. Offline true-future AGRI→R is not deployable; it is a chain-reference diagnostic. RMSEs are FP64 recomputed from the saved bank arrays."}
(OUT/"forecast_cot_metrics_fp64.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
with (OUT/"forecast_cot_lead_metrics_fp64.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(lead_rows[0]));w.writeheader();w.writerows(lead_rows)

plt.figure(figsize=(9.2,5.4))
colors={"sili":"#2563eb","zhujia":"#ea580c"}
styles={"forecast_simvp_to_R":"-","persistence_R":"--","offline_true_future_AGRI_to_R":":"}
labels={"forecast_simvp_to_R":"SimVP→R","persistence_R":"Persistence→R","offline_true_future_AGRI_to_R":"Observed future AGRI→R (offline ref.)"}
for s in STATIONS:
    for method in METHODS:
        sr=[r for r in lead_rows if r["station"]==s]
        y=[r[method+"_rmse_cot"] for r in sr]
        x=[r["lead_min"] for r in sr]
        plt.plot(x,y,linestyle=styles[method],color=colors[s],marker="o" if method==METHODS[0] else None,
                 linewidth=1.7,label=f"{s.upper()} · {labels[method]}")
plt.xlabel("Forecast lead from history end (min)");plt.ylabel("COT RMSE vs CPP reference")
plt.title("Validation forecast-chain COT error on common CPP support")
plt.grid(alpha=.25);plt.legend(ncol=2,fontsize=8);plt.tight_layout()
fig=OUT/"forecast_cot_rmse_by_lead_fp64.png";plt.savefig(fig,dpi=180);plt.close()
print(json.dumps({"npz_sha256":archive_sha,"forecast_array_sha256":forecast_array_sha,"support":support_by_station,"overall":point,"ci":cis,"dates":len(unique_days),"blocks":len(blocks),"figure":str(fig)},indent=2))
