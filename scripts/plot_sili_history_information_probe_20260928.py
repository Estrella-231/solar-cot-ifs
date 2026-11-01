"""Plot the exploratory recent-two vs full-eight history information probe."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.summary.read_text())
    leads = np.arange(15, 241, 15)
    colors = {"recent2": "#0072B2", "full8": "#D55E00"}
    labels = {"recent2": "Recent two frames", "full8": "All eight history frames"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), layout="constrained")
    for arm in ("recent2", "full8"):
        values = d["results"][arm]["lead"]
        y = [values[str(k)]["thick_rmse"] for k in range(1, 17)]
        axes[0].plot(leads, y, color=colors[arm], marker="o" if arm == "recent2" else "s",
                     markersize=4, linewidth=1.8, label=labels[arm])
    axes[0].set_title("16×16 local-patch lattice")
    axes[0].set_xlabel("Forecast lead (min)")
    axes[0].set_ylabel("COT RMSE where CPP COT ≥ 30")
    axes[0].grid(True, color="#D9D9D9", linewidth=.7, alpha=.8)
    axes[0].set_xlim(15, 240)
    axes[0].legend(frameon=False, loc="upper left")

    station_scores = [d["results"][arm]["station_pixel"]["thick_rmse"] for arm in ("recent2", "full8")]
    bars = axes[1].bar([0, 1], station_scores, width=.58,
                       color=[colors["recent2"], colors["full8"]], edgecolor="white")
    axes[1].set_xticks([0, 1], ["Recent two", "All eight"])
    axes[1].set_ylim(0, 36)
    axes[1].set_title("Exact station pixel, all 16 leads pooled")
    axes[1].set_ylabel("COT RMSE where CPP COT ≥ 30")
    axes[1].grid(True, axis="y", color="#D9D9D9", linewidth=.7, alpha=.8)
    axes[1].set_axisbelow(True)
    n = d["results"]["recent2"]["station_pixel"]["bins"]["30-60"]["n"] + d["results"]["recent2"]["station_pixel"]["bins"]["60-100"]["n"]
    for bar, val in zip(bars, station_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + .45, f"{val:.2f}\n(n={n})",
                     ha="center", va="bottom", fontsize=9)
    fig.suptitle("Incremental signal from older COT history\n(validation-only linear probe)", fontsize=13)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(a.output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
