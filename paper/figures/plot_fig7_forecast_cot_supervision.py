from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "fig7_forecast_cot_metrics_20261017.csv"
OUT = HERE

df = pd.read_csv(DATA)
df = df[df["scale"] == "local_3x3"].copy()
methods = [
    ("original_simvp", "Original SimVP", "#0072B2", "-", "o"),
    ("mse_finetune", "MSE fine-tuning", "#D55E00", "--", "s"),
    ("temporal_delta", "Temporal-change loss", "#009E73", "-.", "^"),
    ("spatial_gradient", "Spatial-gradient loss", "#CC79A7", ":", "D"),
    ("persistence_r", "Last-frame persistence", "#000000", (0, (5, 2)), "X"),
]
station_labels = {"sili": "Sili", "zhujia": "Zhujia"}

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
fig, axs = plt.subplots(2, 2, figsize=(180/25.4, 132/25.4), layout="constrained")
for col, station in enumerate(("sili", "zhujia")):
    ax_rmse = axs[0, col]
    ax_csi = axs[1, col]
    sub = df[df["station"] == station]
    for code, label, color, ls, marker in methods:
        m = sub[sub["method"] == code].sort_values("lead_min")
        ax_rmse.plot(m["lead_min"], m["rmse"], label=label, color=color, linestyle=ls,
                     marker=marker, markersize=3.3, linewidth=1.25)
        ax_csi.plot(m["lead_min"], m["csi"], label=label, color=color, linestyle=ls,
                    marker=marker, markersize=3.3, linewidth=1.25)
    ax_rmse.set_title(f"({chr(97+col)}) {station_labels[station]}")
    ax_rmse.set_ylabel("COT RMSE (dimensionless)")
    ax_csi.set_title(f"({chr(99+col)}) {station_labels[station]}")
    ax_csi.set_ylabel("Thick-cloud CSI")
    for ax in (ax_rmse, ax_csi):
        ax.set_xlabel("Forecast lead (min)")
        ax.set_xlim(10, 245)
        ax.set_xticks([15, 60, 120, 180, 240])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    ax_rmse.set_ylim(8, 27)
    ax_csi.set_ylim(0.15, 0.66)
    ax_csi.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False,
           columnspacing=1.3, handlelength=2.5)
fig.supxlabel("Forecast lead time from the last observed frame", fontsize=8.5)
fig.savefig(OUT / "fig7_forecast_cot_supervision.pdf", facecolor="white")
fig.savefig(OUT / "fig7_forecast_cot_supervision.png", dpi=400, facecolor="white")
fig.savefig(OUT / "fig7_forecast_cot_supervision.svg", facecolor="white")
plt.close(fig)
manifest = {
    "source_csv": DATA.name,
    "source_csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    "support": "CPP-valid pixels only; common station x target-time x pixel masks; local 3x3 neighbourhood",
    "aggregation": "pooled valid-pixel COT RMSE and CSI by station and +15 to +240 min lead",
    "methods": [x[0] for x in methods],
    "uncertainty_displayed": False,
    "uncertainty_note": "The sparse CPP reference supports date-block intervals only for the pre-specified aggregate comparisons; per-lead lines are point estimates.",
    "split": "validation only; 3140 windows; 0 test payload reads",
    "formats": ["PDF", "PNG", "SVG"],
    "dimensions_mm": [180, 132],
    "matplotlib": mpl.__version__,
}
(OUT / "fig7_forecast_cot_supervision_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n",encoding="utf-8")
print(json.dumps(manifest,indent=2))
