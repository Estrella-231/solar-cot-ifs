"""Plot paired lead-wise COT errors from the fixed local-64 validation audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.summary.read_text())
    leads = np.array([x["lead_min"] for x in d["per_lead"]])
    names = ["simvp", "persistence", "transport", "learned"]
    labels = {"simvp": "SimVP→R", "persistence": "Persistence",
              "transport": "Transport / step 0", "learned": "Learned residual"}
    colors = {"simvp": "#607d8b", "persistence": "#e69f00",
              "transport": "#009e73", "learned": "#8e63b6"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for name in names:
        axes[0].plot(leads, [x["paired_common_support"][name]["rmse"] for x in d["per_lead"]],
                     marker="o", ms=3.5, lw=1.5, color=colors[name], label=labels[name])
        axes[1].plot(leads, [x["paired_common_support"][name]["thick_rmse"] for x in d["per_lead"]],
                     marker="o", ms=3.5, lw=1.5, color=colors[name], label=labels[name])
    axes[0].set_ylabel("COT RMSE (all COT regimes)")
    axes[1].set_ylabel("COT RMSE (CPP COT ≥ 30)")
    axes[0].legend(ncol=2, frameon=False)
    axes[1].legend(ncol=2, frameon=False)
    axes[2].plot(leads, [x["transport_support_fraction"] for x in d["per_lead"]],
                 color="#333333", marker="o", ms=3.5, lw=1.6)
    axes[2].set_ylabel("Shared support / CPP-valid pixels")
    axes[2].set_ylim(0, 1.03)
    axes[2].set_xlabel("Forecast lead from initialization (min)")
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.set_xlim(10, 245)
    fig.suptitle("Sili local 64×64 — paired validation audit; CPP is retrieval reference",
                 fontsize=13)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output.with_suffix(".png"), dpi=220)
    fig.savefig(a.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
