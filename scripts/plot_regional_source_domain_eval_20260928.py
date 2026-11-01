"""Plot frozen-R source-domain diagnostics from saved validation metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.metrics.read_text(encoding="utf-8"))
    leads = np.asarray([int(k) * 15 for k in d["per_lead"]])
    sizes = (64, 128, 256)
    colors = {64: "#4477AA", 128: "#EE9944", 256: "#228833"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    ax = axes[0]
    for size in sizes:
        vals = [d["per_lead"][str(i)]["coverage"][str(size)]["rmse_cot"] for i in range(1, 17)]
        ax.plot(leads, vals, marker="o", ms=3.5, lw=1.8, color=colors[size], label=f"{size}×{size}")
    ax.set(title="COT RMSE on each domain's available support", xlabel="Forecast lead (min)", ylabel="COT RMSE")
    ax.legend(frameon=False)
    ax.grid(alpha=.25)

    ax = axes[1]
    for size in sizes:
        vals = [d["per_lead"][str(i)]["coverage"][str(size)]["thick_rmse_cot"] for i in range(1, 17)]
        ax.plot(leads, vals, marker="o", ms=3.5, lw=1.8, color=colors[size], label=f"{size}×{size}")
    ax.set(title="Thick-cloud RMSE (CPP COT ≥ 30)", xlabel="Forecast lead (min)", ylabel="COT RMSE")
    ax.grid(alpha=.25)

    ax = axes[2]
    last = d["per_lead"]["16"]
    labels = ["64×64\ncovered", "64→128\nnew band", "128→256\nnew band"]
    entries = [last["coverage"]["64"], last["newly_added_sources_64_to_128"],
               last["newly_added_sources_128_to_256"]]
    vals = [e["rmse_cot"] for e in entries]
    ns = [e.get("n", e.get("n_valid_station_pixels")) for e in entries]
    bcolors = [colors[64], colors[128], colors[256]]
    bars = ax.bar(np.arange(3), vals, color=bcolors, width=.68)
    for bar, n, value in zip(bars, ns, vals):
        ax.text(bar.get_x() + bar.get_width()/2, value + .35, f"RMSE {value:.1f}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylim(0, max(vals) * 1.34)
    ax.set(title="+240 min: coverage vs newly added pixels", ylabel="COT RMSE")
    ax.grid(axis="y", alpha=.25)
    ax.text(.02, .98, "New bands are different pixels;\nnot a paired model comparison.", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color="#444444")

    fig.suptitle("No-training upstream source-domain diagnostic · frozen R256 + same motion", fontsize=14)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
