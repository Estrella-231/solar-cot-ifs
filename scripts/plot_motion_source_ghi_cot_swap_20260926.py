"""Plot same-frozen-head GHI/kt validation input-swap diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"SimVP": "#0072B2", "Motion+Source": "#009E73"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True, help="stable daylight audit JSON")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_STABLE_DAYLIGHT_RESCORE" or report.get("test_used") is not False:
        raise RuntimeError("stable-daylight GHI rescore is incomplete or test-contaminated")
    if a.output_dir.exists():
        raise RuntimeError("refuse to overwrite GHI diagnostic figures")
    a.output_dir.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5), sharex="col")
    station_names = ("Sili", "Zhujia")
    metrics = (("ghi_Wm2", "GHI RMSE (W m$^{-2}$)"), ("kt", "$k_t$ RMSE"))
    for ri, station in enumerate(station_names):
        for ci, (metric_key, ylabel) in enumerate(metrics):
            ax = axes[ri, ci]
            for arm, color in COLORS.items():
                lead = np.array(sorted(map(int, report["stations"][station]["per_lead"])))
                rmse = np.array([report["stations"][station]["per_lead"][str(k)][arm][metric_key]["rmse"]
                                 for k in lead])
                ax.plot(lead, rmse, marker="o", markersize=3.5, linewidth=2,
                        color=color, label=arm)
            ax.axvspan(180, 240, color="#e8edf2", alpha=.8, zorder=0)
            ax.set_title(f"{station} — {ylabel}")
            ax.set_ylabel(ylabel)
            ax.set_xlim(15, 240)
            ax.set_xticks([15, 60, 120, 180, 210, 240])
            ax.grid(True, color="#d8dee5", linewidth=.7, alpha=.8)
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 1:
                ax.set_xlabel("Forecast lead from history end (min)")
        boot = report["stations"][station]["far_date_bootstrap"]["GHI_Wm2"]
        ci = boot["delta_motion_minus_simvp_rmse_95pct"]
        far0 = report["stations"][station]["far_180_to_240min"]["SimVP"]["ghi_Wm2"]["rmse"]
        far1 = report["stations"][station]["far_180_to_240min"]["Motion+Source"]["ghi_Wm2"]["rmse"]
        axes[ri, 0].text(.02, .97,
            f"+180…240: ΔRMSE={far1-far0:+.2f} W/m²\n95% date-bootstrap CI [{ci[0]:+.2f}, {ci[1]:+.2f}]\nP(Motion lower)={boot['probability_motion_rmse_lower']:.2f}",
            transform=axes[ri, 0].transAxes, ha="left", va="top", fontsize=8.7,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "alpha": .88, "edgecolor": "#cccccc"})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle("Validation-only future-COT input swap through one frozen AC_ST station operator\n"
                 "clear-sky GHI ≥ 100 W/m²; head trained on SimVP COT; Motion+Source is a post-hoc input-swap diagnostic",
                 y=.99, fontsize=12.5)
    fig.tight_layout(rect=(0, .06, 1, .91), h_pad=1.3, w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(a.output_dir / f"motion_source_fixed_ghi_head.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(a.output_dir.resolve())


if __name__ == "__main__":
    main()
