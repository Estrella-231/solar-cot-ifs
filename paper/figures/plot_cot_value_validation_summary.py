#!/usr/bin/env python3
"""Reproducible two-panel summary of synchronous COT value and forecast limits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GHI_JSON = ROOT / "experiments/monthly_R_station16_20261013_v1/results/ghi_heads_three_seed_validation_audit.json"
COT_JSON = ROOT / "refine-logs/cot_value_first_20261013/results/cot_forecast_eval_20261014_v3/cot_forecast_metrics.json"
OUT_PDF = HERE / "fig1_cot_value_and_forecast_limits.pdf"
OUT_PNG = HERE / "fig1_cot_value_and_forecast_limits.png"
OUT_JSON = HERE / "fig1_cot_value_and_forecast_limits_data.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data():
    ghi = json.loads(GHI_JSON.read_text(encoding="utf-8"))
    cot = json.loads(COT_JSON.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in ghi["seeds"]]
    ghi_items = []
    for contrast, label in (("AC_minus_A_wm2", "COT vs AGRI-only"),
                            ("AC_minus_AP_wm2", "COT vs AGRI-capacity control")):
        point = float(np.mean([ghi["per_seed"][str(seed)][contrast] for seed in seeds]))
        interval = ghi["bootstrap"]["results"]["target_bjt_date"][contrast]["percentile_95_ci_wm2"]
        ghi_items.append({"label": label, "point": point,
                          "ci": [float(interval[0]), float(interval[1])]})

    cot_s = next(x for x in cot["macro_summary"]
                 if x["method"] == "S_to_R" and x["scale"] == "local_3x3")
    cot_p = next(x for x in cot["macro_summary"]
                 if x["method"] == "persistence_R" and x["scale"] == "local_3x3")
    cot_ci = next(x for x in cot["paired_date_block_bootstrap"]
                  if x["method_minus_persistence_mean_lead_macro_rmse"] == "S_to_R"
                  and x["scale"] == "local_3x3")
    csi_s = next(x for x in cot["thick_cloud_summary"]
                 if x["method"] == "S_to_R" and x["scale"] == "local_3x3")
    csi_p = next(x for x in cot["thick_cloud_summary"]
                 if x["method"] == "persistence_R" and x["scale"] == "local_3x3")
    data = {
        "scope": "validation only; CPP is a retrieval reference, not independent truth",
        "ghi": {
            "n_rows": int(sum(ghi["validation_rows"].values())),
            "n_target_dates": int(ghi["validation_unique_bjt_dates"]),
            "n_seeds": len(seeds),
            "contrasts": ghi_items,
            "bootstrap": ghi["bootstrap"]["interval"],
        },
        "forecast_cot": {
            "scale": "station-centred 3x3",
            "n_target_dates": int(cot["n_unique_target_dates_bjt"]),
            "s_to_r_rmse": float(cot_s["mean_lead_station_macro_rmse"]),
            "persistence_rmse": float(cot_p["mean_lead_station_macro_rmse"]),
            "delta_rmse": float(cot_ci["point_delta_rmse"]),
            "descriptive_ci95": [float(x) for x in cot_ci["ci95_percentile_descriptive"]],
            "thick_cloud_threshold_cot": float(csi_s["threshold_cot"]),
            "thick_cloud_csi_s_to_r": float(csi_s["csi"]),
            "thick_cloud_csi_persistence": float(csi_p["csi"]),
            "interval_status": cot_ci["status"],
        },
        "sources_sha256": {GHI_JSON.name: sha256(GHI_JSON), COT_JSON.name: sha256(COT_JSON)},
    }
    return data


def main():
    data = load_data()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
    blue, green, orange, gray = "#0072B2", "#009E73", "#D55E00", "#666666"
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    ax.axvline(0, color="black", linewidth=0.9, zorder=1)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
    contrasts = data["ghi"]["contrasts"]
    y_positions = [1, 0]
    markers = ["o", "s"]
    for y, item, marker in zip(y_positions, contrasts, markers):
        left = item["point"] - item["ci"][0]
        right = item["ci"][1] - item["point"]
        ax.errorbar(item["point"], y, xerr=np.array([[left], [right]]), fmt=marker,
                    markersize=5.5, color=blue, ecolor=blue, elinewidth=1.4,
                    capsize=3, zorder=3)
    ax.set_yticks(y_positions, ["AC−A", "AC−AP"])
    ax.set_xlim(-8.2, 1.4)
    ax.set_ylim(-0.8, 1.8)
    ax.set_xlabel(r"Δ GHI RMSE (W m$^{-2}$)")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    ax = axes[1]
    ax.axvline(0, color="black", linewidth=0.9, zorder=1)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
    fc = data["forecast_cot"]
    point, low, high = fc["delta_rmse"], fc["descriptive_ci95"][0], fc["descriptive_ci95"][1]
    ax.errorbar(point, 0, xerr=np.array([[point - low], [high - point]]), fmt="D",
                markersize=5.5, color=orange, ecolor=orange, elinewidth=1.4,
                capsize=3, zorder=3)
    ax.set_yticks([0], ["S→R − Persistence-R"])
    ax.set_xlim(-8.2, 2.0)
    ax.set_ylim(-0.9, 1.2)
    ax.set_xlabel("Δ 3×3 RMSE (dimensionless COT)")
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    fig.subplots_adjust(wspace=0.48, bottom=0.23, top=0.98)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"pdf": str(OUT_PDF), "png": str(OUT_PNG), "data": str(OUT_JSON),
                      "ghi": data["ghi"], "forecast_cot": data["forecast_cot"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
