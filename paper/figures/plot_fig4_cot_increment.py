"""Figure 4: station RMSE pairs; paired interval only where verified."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
STATIONS = pd.read_csv(HERE / "fig4_station_increment_data.csv")
GAINS = pd.read_csv(HERE / "fig4_regional_paired_gain.csv")
OUT = HERE / "fig4_cot_incremental_value"


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.35, 1]},
        layout="constrained",
    )
    colors = {"A": "#586f7c", "AC": "#008b79"}
    station_order = [
        ("Hunan", "Sili"), ("Hunan", "Zhujia"),
        ("Hubei", "Niushou"), ("Hubei", "Wangzhai"), ("Hubei", "Mizhuang"),
    ]
    y_positions = [4, 3, 1, 0, -1]
    for y, (region, station) in zip(y_positions, station_order):
        row = STATIONS[(STATIONS.region == region) & (STATIONS.station == station)].iloc[0]
        a, ac = float(row.agri_rmse_wm2), float(row.agri_cot_rmse_wm2)
        ax0.plot([a, ac], [y, y], color="#aab2b5", linewidth=1.1, zorder=1)
        ax0.scatter(a, y, s=31, color=colors["A"], marker="o", edgecolor="white",
                    linewidth=0.5, zorder=3, label="AGRI baseline" if y == 4 else None)
        ax0.scatter(ac, y, s=34, color=colors["AC"], marker="D", edgecolor="white",
                    linewidth=0.5, zorder=3, label="AGRI + COT" if y == 4 else None)
        ax0.text(max(a, ac) + 1.7, y, f"n={int(row.validation_examples)}",
                 va="center", ha="left", fontsize=7, color="#40484c")
    ax0.set_yticks(y_positions, [station for _, station in station_order])
    ax0.set_xlim(50, 151)
    ax0.set_ylim(-1.55, 4.65)
    ax0.set_xlabel("Station validation RMSE (W m$^{-2}$)")
    ax0.set_title("(a) Three-seed mean by station", loc="left")
    ax0.grid(axis="x", color="#d9dddf", linewidth=0.6)
    ax0.set_axisbelow(True)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.axhline(2, color="#b8bec1", linewidth=0.8)
    ax0.text(0.02, 0.95, "HUNAN", transform=ax0.transAxes, fontsize=7,
             fontweight="bold", color="#53636b")
    ax0.text(0.02, 0.37, "HUBEI", transform=ax0.transAxes, fontsize=7,
             fontweight="bold", color="#53636b")
    ax0.legend(loc="lower right", frameon=False, handletextpad=0.5)

    for y, region in zip([1, 0], ["Hunan", "Hubei"]):
        row = GAINS[GAINS.region == region].iloc[0]
        x = float(row.rmse_gain_wm2)
        lo, hi = row.ci_low_wm2, row.ci_high_wm2
        if pd.notna(lo) and pd.notna(hi):
            lo, hi = float(lo), float(hi)
            ax1.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="o", markersize=5,
                         color="#008b79", ecolor="#008b79", capsize=3, linewidth=1.5)
            label = f"{x:.2f} [{lo:.2f}, {hi:.2f}]"
            label_x = hi + 0.25
        else:
            ax1.scatter(x, y, s=30, color="#008b79", edgecolor="white", zorder=3)
            label = f"{x:.2f} (point estimate)"
            label_x = x + 0.25
        ax1.text(label_x, y, label, va="center", ha="left", fontsize=7)
        ax1.text(0.03, y - 0.19, f"n={int(row.validation_examples):,}",
                 transform=ax1.get_yaxis_transform(), va="top", ha="left",
                 fontsize=7, color="#596267")
    ax1.axvline(0, color="#363d40", linewidth=0.9)
    ax1.set_yticks([1, 0], ["Hunan", "Hubei"])
    ax1.set_xlim(0, 8.3)
    ax1.set_ylim(-0.42, 1.42)
    ax1.set_xlabel("RMSE reduction: baseline − COT (W m$^{-2}$)")
    ax1.set_title("(b) Region-equal RMSE reduction", loc="left")
    ax1.grid(axis="x", color="#d9dddf", linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
