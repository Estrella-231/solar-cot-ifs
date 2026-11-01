#!/usr/bin/env python3
"""Publication-style diagnostic plot for the frozen CPP-reference score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.metrics.read_text(encoding="utf-8"))
    assert d["state"] == "COMPLETE_DIRECT_COT_CPP_REFERENCE_VALIDATION_SCORE"
    overall = d["overall"]["station_3x3"]
    names = ["Last-frame", "SimVP→R", "COT-only", "AGRI+COT"]
    arms = ["cot_last", "cot_simvp_r", "cot_p_c", "cot_p_xc"]
    labels = ["Last frame", "SimVP→R", "COT-only P", "AGRI+COT P"]
    colors = ["#969696", "#4C78A8", "#F2A541", "#008C7A"]
    rmse = [overall[k]["rmse"] for k in arms]
    csi = [overall[k]["thick_csi"] for k in arms]
    primary = d["primary_pairwise_bootstrap"]
    contrasts = ["cot_p_xc-cot_p_c", "cot_p_xc-cot_simvp_r"]
    c_labels = ["AGRI+COT P − COT-only P", "AGRI+COT P − SimVP→R"]
    deltas = [primary[k]["station_equal_3x3_rmse_delta"] for k in contrasts]
    cis = [primary[k]["station_equal_3x3_rmse_delta_ci95"] for k in contrasts]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.dpi": 150, "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, vals, title, ylabel, fmt in [
        (axes[0], rmse, "Station-neighbourhood COT RMSE", "RMSE (COT units)", "{:.2f}"),
        (axes[1], csi, "Thick-cloud CSI (COT ≥ 30)", "CSI", "{:.3f}"),
    ]:
        bars = ax.bar(np.arange(4), vals, color=colors, width=0.72)
        ax.set_xticks(np.arange(4), labels, rotation=22, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    y = np.arange(len(deltas))[::-1]
    for yi, delta, ci, color in zip(y, deltas, cis, ["#D17C00", "#008C7A"]):
        ax.errorbar(delta, yi, xerr=[[delta-ci[0]], [ci[1]-delta]], fmt="o", color=color,
                    capsize=4, linewidth=1.7, markersize=6)
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_yticks(y, c_labels)
    ax.set_title("Paired 95% date-block intervals")
    ax.set_xlabel("ΔRMSE (COT units); negative favours AGRI+COT P")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(-2.8, 1.4)
    fig.suptitle("Direct future-COT forecasts: COT skill does not yet imply a GHI gain", y=1.04, fontsize=14)
    fig.text(0.5, -0.015,
             "CPP is a retrieval reference, not independent truth. Outer-validation development score; 1,362 station-target keys, 23/26 date blocks. Test/buffer closed.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, bbox_inches="tight")
    fig.savefig(a.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
