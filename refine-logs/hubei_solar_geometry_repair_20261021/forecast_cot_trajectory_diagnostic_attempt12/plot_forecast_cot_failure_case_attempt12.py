#!/usr/bin/env python3
"""Plot one validation-only AGRI/COT/GHI forecast-chain case with full provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CASE = HERE / "case_202509271330_sample77.npz"
PAIR = ROOT / "refine-logs/hubei_solar_geometry_repair_20261021/fixed_head_cot_substitution_attempt12/paired_predictions_common_validation.csv"
INIT = "202509271330"
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
COLORS = {"Niushou": "#D55E00", "Wangzhai": "#0072B2", "Mizhuang": "#009E73"}
MARKERS = {"Niushou": "*", "Wangzhai": "^", "Mizhuang": "o"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    z = np.load(CASE, allow_pickle=False)
    df = pd.read_csv(PAIR, dtype={"station": str, "init_time_bjt": str, "target_time_bjt": str, "target_date_bjt": str})
    d = df[df.init_time_bjt == INIT].copy()
    if set(d.station.unique()) != set(STATIONS) or set(d.source_role.unique()) != {"validation"}:
        raise ValueError("Selected case must be a three-station validation-only event")
    d["lead_h"] = d.lead_minutes / 60.0
    assert d.groupby(["station", "lead_minutes"]).size().max() == 1
    lead_min = np.asarray(z["lead_minutes"], dtype=int)
    target_times = z["target_time_bjt"].astype(str).tolist()
    assert lead_min.tolist() == [60, 120, 180, 240]
    assert len(set(target_times)) == 4

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(17, 13), layout="constrained")
    gs = fig.add_gridspec(7, 4, height_ratios=[1, 1, 1, 1, 0.10, 0.88, 0.88])
    image_axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(4)])
    cot_axes = [fig.add_subplot(gs[5, c]) for c in range(3)]
    ghi_axes = [fig.add_subplot(gs[6, c]) for c in range(3)]
    legend_ax = fig.add_subplot(gs[5:, 3])
    legend_ax.axis("off")

    row_titles = [
        "Observed future AGRI C02 reflectance",
        "SimVP forecast AGRI C02 reflectance",
        "Offline observed-future AGRI→R COT",
        "Forecast AGRI→R COT",
    ]
    c02_norm = mpl.colors.Normalize(vmin=0.0, vmax=0.8)
    cot_norm = mpl.colors.Normalize(vmin=0.0, vmax=65.0)
    cot_cmap = mpl.colormaps["magma"].with_extremes(bad="#b8b8b8")
    c02_cmap = mpl.colormaps["gray"]
    mappable_c02 = None
    mappable_cot = None
    for r in range(4):
        for c, lead in enumerate(lead_min):
            ax = image_axes[r, c]
            ax.set_aspect("equal")
            ax.set_xlim(-0.5, 15.5); ax.set_ylim(15.5, -0.5)
            ax.set_xticks([0, 4, 8, 12, 15]); ax.set_yticks([0, 4, 8, 12, 15])
            ax.grid(color="white", alpha=0.23, linewidth=0.35)
            if r < 2:
                field = z["observed_c02" if r == 0 else "predicted_c02"][c]
                mappable_c02 = ax.imshow(field, cmap=c02_cmap, norm=c02_norm, interpolation="nearest")
            else:
                key = "observed_future_agri_to_r_cot" if r == 2 else "forecast_cot"
                field = z[key][c].astype(float).copy()
                if r == 2:
                    field[~z["observed_future_cot_valid_mask"][c]] = np.nan
                mappable_cot = ax.imshow(field, cmap=cot_cmap, norm=cot_norm, interpolation="nearest")
            for st in STATIONS:
                rr, cc = CENTERS[st]
                ax.scatter(cc, rr, marker=MARKERS[st], s=45 if MARKERS[st] != "*" else 72,
                           facecolor=COLORS[st], edgecolor="white", linewidth=0.55, zorder=4)
            if r == 0:
                ax.set_title(f"+{lead} min · {target_times[c]}")
            if c == 0:
                ax.set_ylabel(row_titles[r] + "\nrow")
            else:
                ax.set_yticklabels([])
            if r == 3:
                ax.set_xlabel("column")
            else:
                ax.set_xticklabels([])
    cb1 = fig.colorbar(mappable_c02, ax=image_axes[:2, :].ravel().tolist(), shrink=0.78, pad=0.01)
    cb1.set_label("AGRI C02 visible reflectance (preprocessed physical channel units)")
    cb2 = fig.colorbar(mappable_cot, ax=image_axes[2:, :].ravel().tolist(), shrink=0.78, pad=0.01)
    cb2.set_label("COT (retrieval units)")

    for st, ax in zip(STATIONS, cot_axes):
        q = d[d.station == st].sort_values("lead_minutes")
        ax.plot(q.lead_h, q.forecast_cot_station, color="#0072B2", marker="o", ms=3.4,
                lw=1.7, label="Forecast AGRI→R")
        ax.plot(q.lead_h, q.offline_future_cot_station, color="#D55E00", marker="s", ms=3.2,
                lw=1.5, ls="--", label="Observed future AGRI→R (offline)")
        ax.plot(q.lead_h, q.history_cot_station, color="#777777", lw=1.1, ls=":",
                label="History COT held constant")
        ax.set_title(f"{st}: station-pixel COT")
        ax.set_xlabel("Lead from init (h)")
        ax.set_ylabel("COT")
        ax.set_xlim(0.2, 4.05); ax.set_xticks([0.25, 1, 2, 3, 4])
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)

    for st, ax in zip(STATIONS, ghi_axes):
        q = d[d.station == st].sort_values("lead_minutes")
        actual = q.truth.to_numpy(float)
        pforecast = q.forecast_cot.to_numpy(float)
        poracle = q.offline_future_agri.to_numpy(float)
        rmse_f = float(np.sqrt(np.mean((pforecast - actual) ** 2)))
        rmse_o = float(np.sqrt(np.mean((poracle - actual) ** 2)))
        x = q.lead_h.to_numpy(float)
        ax.plot(x, actual, color="#111111", marker="o", ms=3.2, lw=1.7, label="Observed 15-min GHI")
        ax.plot(x, pforecast, color="#0072B2", marker="o", ms=3.0, lw=1.5, label="Frozen FAC head + forecast COT")
        ax.plot(x, poracle, color="#D55E00", marker="s", ms=3.0, lw=1.4, ls="--",
                label="Same head + offline future AGRI→R COT")
        ax.set_title(f"{st}: GHI on exact available validation leads\nRMSE forecast-COT {rmse_f:.1f}; offline-reference {rmse_o:.1f} W m⁻²")
        ax.set_xlabel("Lead from init (h)"); ax.set_ylabel("15-min mean GHI (W m⁻²)")
        ax.set_xlim(0.2, 4.05); ax.set_xticks([0.25, 1, 2, 3, 4]); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    # A compact fourth column avoids hiding missing-data and scope boundaries.
    handles, labels = cot_axes[0].get_legend_handles_labels()
    h2, l2 = ghi_axes[0].get_legend_handles_labels()
    legend_ax.legend(handles + h2, labels + l2, loc="upper left", frameon=False, ncol=1)
    legend_ax.text(0.02, 0.45,
        "Illustrative validation event\n\n"
        "Init: 2025-09-27 13:30 BJT\n"
        "Patch: 16×16; station markers shown\n"
        "C02 maps use a fixed 0–0.8 scale\n"
        "COT maps use a fixed 0–65 scale\n\n"
        "Offline future AGRI→R is an attribution reference, not a deployable input or independent COT truth.\n\n"
        "GHI curves reuse one frozen FAC checkpoint; only COT input changes. Missing validation leads remain gaps.\n\n"
        "Selected for a large local COT trajectory mismatch; this is not a population summary.",
        va="top", ha="left", fontsize=9, wrap=True)
    legend_ax.set_title("Legend and scope", loc="left")

    fig.suptitle("Forecast-COT failure diagnosis on a held-out validation event\n"
                 "Observed AGRI vs SimVP AGRI → frozen R, COT trajectory, and fixed-head GHI response")
    out_png = HERE / "validation_event_20250927_1330_attempt12.png"
    out_pdf = HERE / "validation_event_20250927_1330_attempt12.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    trajectories = d.sort_values(["station", "lead_minutes"])
    trajectories.to_csv(HERE / "validation_event_20250927_1330_trajectories.csv", index=False)
    summary = {
        "state": "VALIDATION_ONLY_ILLUSTRATIVE_CASE_FIGURE",
        "init_time_bjt": INIT,
        "sample_index": 77,
        "selection_note": "diagnostic case selected for pronounced station COT trajectory mismatch; not representative population estimate",
        "source_split": "validation",
        "stations": {},
        "physical_image": "AGRI C02 visible reflectance map in preprocessed physical channel units; observed and forecast shown with the same fixed 0-0.8 scale",
        "offline_cot_reference": "corrected frozen R applied to observed future AGRI at target time; not an independent COT truth and unavailable at forecast issuance",
        "ghi_scope": "same frozen FAC head/checkpoint, only COT input substituted; exact common validation keys per station; missing leads are not interpolated",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "source_sha256": {"case_arrays": sha(CASE), "paired_prediction_csv": sha(PAIR)},
        "outputs": {"png": str(out_png), "pdf": str(out_pdf), "trajectory_csv": str(HERE / "validation_event_20250927_1330_trajectories.csv")},
    }
    for st in STATIONS:
        q = d[d.station == st]
        f = q.forecast_cot.to_numpy(float) - q.truth.to_numpy(float)
        o = q.offline_future_agri.to_numpy(float) - q.truth.to_numpy(float)
        summary["stations"][st] = {
            "n_common_leads": int(len(q)),
            "forecast_cot_input_ghi_rmse_wm2": float(np.sqrt(np.mean(f*f))),
            "offline_future_agri_to_r_input_ghi_rmse_wm2": float(np.sqrt(np.mean(o*o))),
            "station_cot_forecast_vs_offline_future_rmse": float(np.sqrt(np.mean((q.forecast_cot_station-q.offline_future_cot_station)**2))),
            "forecast_cot_at_120min": (float(q.loc[q.lead_minutes == 120, "forecast_cot_station"].iloc[0]) if (q.lead_minutes == 120).any() else None),
            "offline_future_cot_at_120min": (float(q.loc[q.lead_minutes == 120, "offline_future_cot_station"].iloc[0]) if (q.lead_minutes == 120).any() else None),
            "forecast_cot_at_240min": (float(q.loc[q.lead_minutes == 240, "forecast_cot_station"].iloc[0]) if (q.lead_minutes == 240).any() else None),
            "offline_future_cot_at_240min": (float(q.loc[q.lead_minutes == 240, "offline_future_cot_station"].iloc[0]) if (q.lead_minutes == 240).any() else None),
        }
    (HERE / "validation_event_20250927_1330_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

