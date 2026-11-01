"""Plot published Cloud-FNO values beside local old-validation COT RMSE.

Local curves use the existing 128-sequence, Sili station-pixel paired audit.
No predictions are rerun and no test data are read.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE_DIR = PROJECT / "results/paired_forecast_oracle_cot_rmse_20261003_v2"
OUT = HERE / "predicted_image_cot_rmse_20261016_v3"
HOURS = (1, 2, 3, 4)
LEADS = (60, 120, 180, 240)


def main():
    source_csv = SOURCE_DIR / "paired_cot_rmse.csv"
    source_report = SOURCE_DIR / "REPORT.md"
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {int(row["lead_minutes"]): row for row in csv.DictReader(stream)}
    if not all(lead in rows for lead in LEADS):
        raise RuntimeError("Missing one or more hourly lead rows in the source table")

    forecast = np.array([float(rows[lead]["station_forecast_rmse"]) for lead in LEADS])
    observed = np.array([float(rows[lead]["station_oracle_rmse"]) for lead in LEADS])
    counts = np.array([int(rows[lead]["station_n"]) for lead in LEADS])
    cloud_fno = np.array([6.75, 8.09, 8.93, 9.61])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.29, wspace=0.04)
    ax = axes[0]
    ax.plot(HOURS, cloud_fno, color="#8f63bd", linestyle="--", marker="D", linewidth=2,
            markersize=6, label="Cloud-FNO (paper Fig. 2)")
    for x, y in zip(HOURS, cloud_fno):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set_title("Literature reference")
    ax.set_xlabel("Forecast lead (hours)")
    ax.set_ylabel("COT RMSE (dimensionless)")
    ax.set_xticks(HOURS)
    ax.set_ylim(0, 14)
    ax.set_xlim(0.8, 4.2)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(HOURS, forecast, color="#0072B2", marker="o", linewidth=2,
            label="R(SimVP forecast AGRI)")
    ax.plot(HOURS, observed, color="#009E73", marker="s", linewidth=2,
            label="R(observed future AGRI; offline)")
    for x, y, n in zip(HOURS, forecast, counts):
        ax.annotate(f"{y:.2f}\nn={n}", (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for x, y in zip(HOURS, observed):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -17), textcoords="offset points",
                    ha="center", va="top", fontsize=8)
    ax.set_title("Hunan Sili station pixel · old validation set")
    ax.set_xlabel("Forecast lead (hours)")
    ax.set_xticks(HOURS)
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(0, 14)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("COT retrieval RMSE against CPP reference", fontsize=14)
    fig.text(
        0.5, 0.075,
        "Local: 128 selected sequences from old Sili validation; CPP-supported station n varies by lead. "
        "Green = offline R(observed future AGRI). CPP is a retrieval reference; Cloud-FNO uses a different dataset/protocol and is not directly comparable.",
        ha="center", va="top", fontsize=8, wrap=True,
    )
    OUT.mkdir(parents=True, exist_ok=False)
    fig.savefig(OUT / "predicted_image_cot_rmse_1to4h.png", dpi=300)
    fig.savefig(OUT / "predicted_image_cot_rmse_1to4h.pdf")
    plt.close(fig)

    records = []
    for i, hour in enumerate(HOURS):
        records.append({
            "lead_hour": hour,
            "lead_minutes": LEADS[i],
            "station_pixel_valid_n": int(counts[i]),
            "forecast_agri_to_R_cot_rmse": float(forecast[i]),
            "observed_future_agri_to_R_cot_rmse_offline": float(observed[i]),
            "cloud_fno_paper_fig2_cot_rmse_different_protocol": float(cloud_fno[i]),
        })
    (OUT / "values.json").write_text(json.dumps({
        "source_csv": str(source_csv),
        "source_report": str(source_report),
        "scope": "old validation paired audit; Sili station pixel; shared CPP reference and valid mask",
        "records": records,
        "limitations": [
            "valid station sample count varies by lead",
            "future observed AGRI retrieval is offline only",
            "CPP is a retrieval reference, not independent physical truth",
            "Cloud-FNO values are from a different dataset and protocol",
            "no confidence intervals are shown; this plot reproduces point estimates only",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
