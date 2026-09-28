"""Render the validation-only, capacity-matched COT history-control audit."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5"
OUT = RESULT / "history_control_validation.png"
PDF = RESULT / "history_control_validation.pdf"


def main():
    data = json.loads((RESULT / "training_complete.json").read_text(encoding="utf-8"))
    transport = data["full_history"]["selected_lead_metrics"]
    repeat = data["repeat_last_control"]["best_learned_lead_metrics"]
    full = data["full_history"]["best_learned_lead_metrics"]
    leads = np.arange(1, 17) * 15
    colors = {"transport": "#555B66", "repeat": "#D98500", "full": "#007C91"}
    labels = {"transport": "Step-0 pure transport (selected)",
              "repeat": "Best learned: repeated-last history",
              "full": "Best learned: full 8-frame history"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), layout="constrained")
    for ax, scope, title in zip(axes[:2], ("patch", "station"),
                                ("64×64 patch", "Station pixel")):
        for key, series in (("transport", transport), ("repeat", repeat), ("full", full)):
            vals = [series[scope][str(k)]["learned"]["thick_rmse"] for k in range(1, 17)]
            ax.plot(leads, vals, marker="o", markersize=3.7, linewidth=1.8,
                    color=colors[key], label=labels[key])
        ax.set_title(f"CPP COT ≥ 30 RMSE — {title}")
        ax.set_xlabel("Forecast lead (min)")
        ax.set_ylabel("COT RMSE")
        ax.set_xticks([15, 60, 120, 180, 240])
        ax.grid(True, alpha=.24)

    ax = axes[2]
    bins = ("0–5", "5–10", "10–30", "30–60", "60–100")
    methods = (("transport", data["full_history"]["selected_metrics"]["patch"]["transport"]),
               ("repeat", data["repeat_last_control"]["best_learned_metrics"]["patch"]["learned"]),
               ("full", data["full_history"]["best_learned_metrics"]["patch"]["learned"]))
    x = np.arange(len(bins)); width = .24
    for j, (key, metric) in enumerate(methods):
        vals = [metric["bins"][b]["rmse"] for b in ("0_5", "5_10", "10_30", "30_60", "60_100")]
        ax.bar(x + (j - 1) * width, vals, width, color=colors[key], label=labels[key])
    ax.set_title("Patch RMSE by reference COT bin")
    ax.set_xlabel("CPP reference COT")
    ax.set_ylabel("COT RMSE")
    ax.set_xticks(x, bins)
    ax.grid(axis="y", alpha=.24)
    ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Capacity-matched COT history control — validation only", fontsize=14)
    fig.savefig(OUT, dpi=220)
    fig.savefig(PDF)
    print(OUT)
    print(PDF)


if __name__ == "__main__":
    main()
