"""Validation-only analysis of saved B1 COT prediction artifacts.

The script never opens test data or COT source products.  It recomputes all
displayed metrics from the saved paired validation predictions and records
hashes for every input file.
"""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "audits/actual_cot_gain_analysis_20260918/inputs"
OUT = ROOT / "figures/cot_pipeline_20260918"
AUDIT = ROOT / "audits/actual_cot_gain_analysis_20260918"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name):
    path = INPUT / f"{name}_validation_predictions.npz"
    with np.load(path, allow_pickle=False) as item:
        values = {key: item[key].copy() for key in item.files}
    return values, path


def metric(item):
    error = item["pred_kt"].astype(np.float64) * item["clear_sky_ghi"] - item["observed_ghi"]
    by_station = [float(np.sqrt(np.mean(error[item["station"] == site] ** 2))) for site in (0, 1)]
    by_lead = []
    for lead in range(16):
        mask = item["lead"] == lead
        station_values = [np.sqrt(np.mean(error[mask & (item["station"] == site)] ** 2)) for site in (0, 1)]
        by_lead.append(float(np.mean(station_values)))
    return error, {"station_rmse": by_station, "equal_station_rmse": float(np.mean(by_station)), "lead_rmse": by_lead}


def paired(base, candidate):
    for key in ("pack_row", "observed_ghi", "clear_sky_ghi", "station", "lead"):
        if not np.array_equal(base[key], candidate[key]):
            raise RuntimeError("paired prediction key/value mismatch: " + key)
    base_error, base_metric = metric(base)
    candidate_error, candidate_metric = metric(candidate)
    correction = (candidate["pred_kt"].astype(np.float64) - base["pred_kt"].astype(np.float64)) * base["clear_sky_ghi"]
    mse_change = candidate_error**2 - base_error**2
    output = {"baseline": base_metric, "candidate": candidate_metric,
              "delta_equal_station_rmse": candidate_metric["equal_station_rmse"] - base_metric["equal_station_rmse"],
              "prediction_adjustment_Wm2": {"mean": float(correction.mean()), "mean_abs": float(np.abs(correction).mean()),
                                               "p50_abs": float(np.quantile(np.abs(correction), .5)),
                                               "p90_abs": float(np.quantile(np.abs(correction), .9))},
              "sample_fraction_lower_squared_error": float(np.mean(mse_change < 0)),
              "mean_squared_error_change_Wm4": float(mse_change.mean()),
              "by_station": {}, "by_lead": []}
    for site in (0, 1):
        mask = base["station"] == site
        output["by_station"][str(site)] = {"delta_rmse": candidate_metric["station_rmse"][site] - base_metric["station_rmse"][site],
                                            "fraction_lower_squared_error": float(np.mean(mse_change[mask] < 0)),
                                            "mean_adjustment": float(correction[mask].mean()),
                                            "mean_abs_adjustment": float(np.abs(correction[mask]).mean())}
    for lead in range(16):
        mask = base["lead"] == lead
        output["by_lead"].append({"lead_minutes": 15 * (lead + 1),
                                  "delta_equal_station_rmse": candidate_metric["lead_rmse"][lead] - base_metric["lead_rmse"][lead],
                                  "fraction_lower_squared_error": float(np.mean(mse_change[mask] < 0)),
                                  "mean_abs_adjustment": float(np.abs(correction[mask]).mean())})
    return output, correction, mse_change


def main():
    OUT.mkdir(parents=True, exist_ok=True); AUDIT.mkdir(parents=True, exist_ok=True)
    a, a_path = read("A_sp"); stat, stat_path = read("AC_stat_sp"); cmap, map_path = read("AC_map")
    a_station, a_station_path = read("A_station"); ac_station, ac_station_path = read("AC_station")
    stat_report, _, _ = paired(a, stat)
    map_report, correction, mse_change = paired(a, cmap)
    station_report, _, _ = paired(a_station, ac_station)
    b2_path = INPUT / "spatial_b2_complete.json"
    b2 = json.loads(b2_path.read_text(encoding="utf-8"))
    shuffle_metric = b2["runs"]["AC_shuffle"]["station_equal_ghi_rmse"]
    proxy_metric = b2["runs"]["A_proxy"]["station_equal_ghi_rmse"]
    b2_report = {"AC_map": metric(cmap)[1]["equal_station_rmse"], "AC_shuffle": shuffle_metric,
                 "A_proxy": proxy_metric, "shuffle_minus_map": shuffle_metric - metric(cmap)[1]["equal_station_rmse"],
                 "proxy_minus_map": proxy_metric - metric(cmap)[1]["equal_station_rmse"],
                 "boundary": "seed42 10-epoch exploratory B2; AC_shuffle and A_proxy retrained under the B2 contract"}
    source = {path.name: sha(path) for path in (a_path, stat_path, map_path, a_station_path, ac_station_path, b2_path)}
    report = {"state": "VALIDATION_ONLY_RECOMPUTED_FROM_SAVED_PREDICTIONS", "test_used": False,
              "input_sha256": source, "spatial_B1": {"AC_stat_sp_vs_A_sp": stat_report, "AC_map_vs_A_sp": map_report},
              "spatial_B2_controls": b2_report,
              "station_aware_screen": {"AC_station_vs_A_station": station_report,
                  "limitation": "batch16384 changed optimizer-update count; this is a throughput screen, not a representation conclusion"},
              "interpretation_boundary": "one seed, 10 epoch exploratory pilots; no test, no causal or generalization claim"}
    (AUDIT / "actual_cot_gain_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    lead = np.array([15 * (index + 1) for index in range(16)])
    stat_delta = np.array([row["delta_equal_station_rmse"] for row in stat_report["by_lead"]])
    map_delta = np.array([row["delta_equal_station_rmse"] for row in map_report["by_lead"]])
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), layout="constrained")
        groups = ["A_sp", "AC_stat", "AC_map"]
        values = [metric(a)[1]["equal_station_rmse"], metric(stat)[1]["equal_station_rmse"], metric(cmap)[1]["equal_station_rmse"]]
        colors = ["#607d8b", "#9575cd", "#00897b"]
        axes[0, 0].bar(groups, values, color=colors)
        axes[0, 0].set(title="Actual B1 validation endpoint", ylabel="equal-station RMSE (W m$^{-2}$)")
        for i, value in enumerate(values): axes[0, 0].text(i, value + .03, f"{value:.3f}", ha="center", fontsize=9)

        sites = np.arange(2); width = .25
        for offset, item, label, color in ((-width, a, "A_sp", colors[0]), (0, stat, "AC_stat", colors[1]), (width, cmap, "AC_map", colors[2])):
            axes[0, 1].bar(sites + offset, metric(item)[1]["station_rmse"], width, label=label, color=color)
        axes[0, 1].set(title="Actual B1 station split", xticks=sites, xticklabels=["Sili", "Zhujia"], ylabel="RMSE (W m$^{-2}$)")
        axes[0, 1].legend(frameon=False, fontsize=8)

        axes[0, 2].axhline(0, color="#444444", lw=1)
        axes[0, 2].plot(lead, stat_delta, marker="o", color=colors[1], label="AC_stat − A_sp")
        axes[0, 2].plot(lead, map_delta, marker="s", color=colors[2], label="AC_map − A_sp")
        axes[0, 2].set(title="Actual lead-wise RMSE difference", xlabel="forecast lead (min)", ylabel="candidate − A_sp (W m$^{-2}$)")
        axes[0, 2].legend(frameon=False, fontsize=8)

        axes[1, 0].hist(correction, bins=70, color="#00897b", alpha=.85)
        axes[1, 0].axvline(0, color="#222222", lw=1)
        axes[1, 0].set(title="AC_map changes predictions", xlabel="AC_map − A_sp prediction (W m$^{-2}$)", ylabel="validation samples")
        axes[1, 0].text(.02, .94, f"mean |change| = {map_report['prediction_adjustment_Wm2']['mean_abs']:.1f} W m$^{{-2}}$\n"
                                f"90th pct = {map_report['prediction_adjustment_Wm2']['p90_abs']:.1f} W m$^{{-2}}$",
                         transform=axes[1, 0].transAxes, va="top", fontsize=9)

        draw = np.linspace(0, len(correction) - 1, min(6000, len(correction)), dtype=int)
        axes[1, 1].scatter(correction[draw], mse_change[draw], s=3, alpha=.16, color="#5e35b1", rasterized=True)
        axes[1, 1].axhline(0, color="#222222", lw=1); axes[1, 1].axvline(0, color="#222222", lw=1)
        axes[1, 1].set(title="Corrections improve and worsen different samples", xlabel="prediction adjustment (W m$^{-2}$)", ylabel="AC_map MSE − A_sp MSE (W$^2$ m$^{-4}$)")
        axes[1, 1].text(.02, .94, f"lower squared error: {100 * map_report['sample_fraction_lower_squared_error']:.1f}%\n"
                                f"net endpoint ΔRMSE: {map_report['delta_equal_station_rmse']:+.3f} W m$^{{-2}}$",
                         transform=axes[1, 1].transAxes, va="top", fontsize=9)

        labels = ["AC_map", "AC_shuffle", "A_proxy"]
        control_values = [b2_report["AC_map"], b2_report["AC_shuffle"], b2_report["A_proxy"]]
        axes[1, 2].bar(labels, control_values, color=["#00897b", "#f9a825", "#607d8b"])
        axes[1, 2].set(title="Actual B2: spatial-order control", ylabel="equal-station RMSE (W m$^{-2}$)")
        for i, value in enumerate(control_values): axes[1, 2].text(i, value + .02, f"{value:.3f}", ha="center", fontsize=9)
        axes[1, 2].text(.02, .06, f"shuffle − map = {b2_report['shuffle_minus_map']:+.3f} W m$^{{-2}}$\n"
                                "Spatial order has almost no visible effect in this static head.",
                         transform=axes[1, 2].transAxes, fontsize=8.5, color="#9c1c1c")
        fig.suptitle("Actual saved validation outputs: why the current static COT gain is small", fontsize=16, weight="bold")
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig_actual_static_cot_gain_diagnosis_20260918.{ext}", dpi=220 if ext == "png" else None, facecolor="white")
        plt.close(fig)

    lines = ["# 实际保存预测的 COT 小增益诊断", "", "仅使用 B1/站点感知 seed42 validation 预测；未读取测试集。", "",
             "| 比较 | 等权 RMSE 差（候选−基线，W/m²） | 每样本平方误差下降比例 | 平均绝对预测调整（W/m²） |", "|---|---:|---:|---:|"]
    for label, item in (("AC_stat_sp − A_sp", stat_report), ("AC_map − A_sp", map_report), ("AC_station − A_station（仅吞吐筛选）", station_report)):
        lines.append(f"| {label} | {item['delta_equal_station_rmse']:+.4f} | {100*item['sample_fraction_lower_squared_error']:.2f}% | {item['prediction_adjustment_Wm2']['mean_abs']:.2f} |")
    lines += ["", f"B2 空间排列控制：AC_shuffle−AC_map = {b2_report['shuffle_minus_map']:+.4f} W/m²；静态头中空间排列的额外作用接近零。",
              "", "解释边界：这些是同一验证集、单 seed、10 epoch 的已保存预测重算。它们能说明当前静态头的预测调整在样本间相互抵消，不能证明 COT 无效，也不能代替因果时空轨迹实验。"]
    (AUDIT / "actual_cot_gain_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"map_delta_rmse": map_report["delta_equal_station_rmse"],
                      "map_mean_abs_adjustment": map_report["prediction_adjustment_Wm2"]["mean_abs"]}, indent=2))


if __name__ == "__main__":
    main()
