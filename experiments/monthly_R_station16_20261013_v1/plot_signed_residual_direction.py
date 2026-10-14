#!/usr/bin/env python3
"""Validation figure: overall correction and bright thick-cloud exception."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    metrics = read_csv(a.analysis / "per_seed_metrics.csv")
    sub = read_csv(a.analysis / "signed_direction_subgroups.csv")
    summary = json.loads((a.analysis / "independent_summary.json").read_text())
    palette = {"sili": "#0072B2", "zhujia": "#D55E00"}
    markers = {"sili": "o", "zhujia": "s"}
    seed_jitter = {42: -0.12, 43: 0.0, 44: 0.12}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.4), layout="constrained")
    arms = [("AGRI", "AGRI residual"), ("AGRI_COT", "AGRI + COT residual")]
    for x, (key, label) in enumerate(arms):
        field = f"{key}_minus_A_RMSE_wm2"
        for row in metrics:
            st, seed = row["station"], int(row["seed"])
            ax1.scatter(x + seed_jitter[seed], float(row[field]),
                        color=palette[st], marker=markers[st], s=54,
                        edgecolor="white", linewidth=0.6, zorder=3,
                        label=f"{st.title()} seed {seed}")
        boot_key = "AGRI_residual_minus_A" if key == "AGRI" else "AGRI_COT_residual_minus_A"
        ci = summary["paired_date_block_uncertainty"][boot_key]
        y = ci["left_minus_right_rmse_wm2"]
        lo, hi = ci["date_block_bootstrap_95pct_ci_wm2"]
        ax1.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="D", color="#222222",
                     capsize=4, markersize=7, linewidth=1.5, zorder=4,
                     label="Station-equal mean and 95% date-block CI")
    ax1.axhline(0, color="#333333", linewidth=1)
    ax1.set_xticks(range(2), [v[1] for v in arms])
    ax1.set_ylabel("Validation RMSE change from frozen A (W m$^{-2}$)")
    ax1.set_title("Full validation set\n(n = 2,128 Sili; 2,351 Zhujia)")
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax1.set_xlim(-0.45, 1.45)

    condition = "COT ≥ 3, clear-sky GHI ≥ 100, observed kt ≥ 0.9"
    groups = [("sili", "AGRI"), ("sili", "AGRI_COT"),
              ("zhujia", "AGRI"), ("zhujia", "AGRI_COT")]
    group_x = {g: i for i, g in enumerate(groups)}
    for row in sub:
        if row["subset"] != "COT_ge3_clear_ge100_observed_kt_ge0p9":
            continue
        st, arm, seed = row["station"], row["arm"], int(row["seed"])
        x = group_x[(st, arm)] + seed_jitter[seed] * 0.45
        delta = float(row["candidate_rmse_wm2"]) - float(row["A_rmse_wm2"])
        ax2.scatter(x, delta, color=palette[st], marker=markers[st], s=54,
                    edgecolor="white", linewidth=0.6, zorder=3)
    for i, (st, arm) in enumerate(groups):
        vals = [float(r["candidate_rmse_wm2"]) - float(r["A_rmse_wm2"])
                for r in sub if r["subset"] == "COT_ge3_clear_ge100_observed_kt_ge0p9"
                and r["station"] == st and r["arm"] == arm]
        if vals:
            ax2.scatter(i, np.mean(vals), color=palette[st], marker="D", s=90,
                        edgecolor="#222222", linewidth=0.7, zorder=4)
    ax2.axhline(0, color="#333333", linewidth=1)
    ax2.set_xticks(range(4), ["Sili\nAGRI", "Sili\nAGRI+COT",
                              "Zhujia\nAGRI", "Zhujia\nAGRI+COT"])
    ax2.set_ylabel("Subset RMSE change from frozen A (W m$^{-2}$)")
    ax2.set_title("Bright thick-cloud diagnostic subset\n" + condition)
    ax2.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax2.set_xlim(-0.55, 3.55)
    ax2.text(0.02, 0.02, "Points: seeds 42/43/44; diamonds: seed mean\n"
             "Subset is post hoc validation diagnosis, not an input gate",
             transform=ax2.transAxes, fontsize=8.5, va="bottom",
             bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9})

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=palette["sili"], label="Sili"),
               plt.Line2D([], [], marker="s", linestyle="", color=palette["zhujia"], label="Zhujia"),
               plt.Line2D([], [], marker="D", linestyle="", color="#222222", label="Seed mean / block-CI estimate")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Signed residuals help overall; the bright thick-cloud failure remains", fontsize=14)
    a.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output / "signed_residual_direction_validation.png", dpi=240,
                facecolor="white")
    fig.savefig(a.output / "signed_residual_direction_validation.pdf",
                facecolor="white")
    plt.close(fig)
    (a.output / "figure_alt_text.md").write_text(
        "Two-panel validation figure. Left: AGRI-only and AGRI+COT signed residual heads "
        "reduce station-equal RMSE relative to frozen A; each point is a station/seed and "
        "diamonds show the station-equal estimate with date-block bootstrap intervals. "
        "Right: for validation samples with retrieved center COT at least 3, clear-sky GHI "
        "at least 100 W m-2, and observed kt at least 0.9, AGRI+COT worsens Sili RMSE and "
        "is slightly worse than AGRI-only at Zhujia. The subset uses observed labels only "
        "for post-hoc diagnosis and is not a model input.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
