#!/usr/bin/env python3
"""Plot the historical-COT paired GHI validation result from saved summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_attempt2"
SCORED = RES / "scored_attempt2"


def main() -> None:
    summary = json.loads((SCORED / "global_summary.json").read_text(encoding="utf-8"))
    arms = list(csv.DictReader((SCORED / "arm_metrics.csv").open(newline="", encoding="utf-8")))
    hours = list(csv.DictReader((SCORED / "by_hour.csv").open(newline="", encoding="utf-8")))
    station_rows = []
    for station in ("Niushou", "Wangzhai", "Mizhuang"):
        rows = [r for r in arms if r["station"] == station]
        base = float(np.mean([float(r["rmse_wm2"]) for r in rows if r["arm"] == "FA_HIST"]))
        cot = float(np.mean([float(r["rmse_wm2"]) for r in rows if r["arm"] == "FAC_HIST"]))
        station_rows.append((station, cot - base))

    d = float(summary["paired_delta_rmse_fac_minus_fa_wm2"])
    lo, hi = map(float, summary["paired_bjt_date_block_bootstrap_95pct_delta_ci_wm2"])
    ylabels = ["Overall (date-block 95% CI)"]
    estimates = [d]
    lower = [d - lo]
    upper = [hi - d]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.2, 5.4), layout="constrained",
        gridspec_kw={"width_ratios": [0.88, 1.42]},
    )
    ax0.axvline(0, color="#444444", lw=1.1, ls=(0, (4, 3)))
    ax0.errorbar(estimates, [0], xerr=np.array([lower, upper]), fmt="o",
                 color="#0072B2", ecolor="#0072B2", capsize=5, lw=2.2, ms=8)
    ax0.set_yticks([0], ylabels)
    ax0.set_xlim(-4.8, 1.2)
    ax0.set_ylim(-0.7, 0.7)
    ax0.set_xlabel("ΔRMSE: history-COT input − zero-COT control (W m⁻²)")
    ax0.set_title("Overall paired validation result")
    ax0.text(0.03, 0.95,
             f"Point: {d:.2f} W m⁻²\n95% date-block CI: [{lo:.2f}, {hi:.2f}]\n"
             f"RMSE: {summary['equal_station_equal_seed_mean_rmse_wm2']['FA_HIST']:.2f} → "
             f"{summary['equal_station_equal_seed_mean_rmse_wm2']['FAC_HIST']:.2f} W m⁻²\n"
             f"Relative point improvement: {summary['relative_rmse_improvement_pct']:.2f}%",
             transform=ax0.transAxes, va="top", ha="left", fontsize=9.5,
             bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95})
    ax0.text(0.5, -0.23, "Negative values favor the history-COT input.",
             transform=ax0.transAxes, ha="center", va="top", fontsize=9)

    cats = [(s, v, "o") for s, v in station_rows]
    cats += [(r["hour_bin"], float(r["delta_fac_minus_fa_wm2"]), "s") for r in hours]
    # Keep site results above hourly results, with a visible separator.
    labels = [x[0] for x in cats]
    vals = [x[1] for x in cats]
    yy = np.arange(len(cats))[::-1]
    colors = ["#D55E00"] * len(station_rows) + ["#009E73"] * len(hours)
    markers = [x[2] for x in cats]
    ax1.axvline(0, color="#444444", lw=1.1, ls=(0, (4, 3)))
    for y, value, color, marker in zip(yy, vals, colors, markers):
        ax1.scatter(value, y, color=color, marker=marker, s=58, zorder=3)
        ax1.text(value + 0.12, y, f"{value:.2f}", va="center", ha="left", fontsize=9)
    ax1.axhline(len(station_rows) - 0.5, color="#999999", lw=0.8, ls=":")
    ax1.set_yticks(yy, labels)
    ax1.set_xlim(-3.3, 0.7)
    ax1.set_ylim(-0.7, len(cats) - 0.3)
    ax1.set_xlabel("ΔRMSE (W m⁻²)")
    ax1.set_title("Point estimates by station and forecast hour\n(no subgroup intervals)")

    fig.suptitle("Historical COT input: Hubei station-GHI validation", fontsize=14, fontweight="semibold")
    for ax in (ax0, ax1):
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
    fig.savefig(RES / "history_cot_ghi_validation_attempt2.png", dpi=300, facecolor="white")
    fig.savefig(RES / "history_cot_ghi_validation_attempt2.pdf", facecolor="white")
    plt.close(fig)
    print(f"wrote {RES / 'history_cot_ghi_validation_attempt2.png'}")
    print(f"wrote {RES / 'history_cot_ghi_validation_attempt2.pdf'}")


if __name__ == "__main__":
    main()
