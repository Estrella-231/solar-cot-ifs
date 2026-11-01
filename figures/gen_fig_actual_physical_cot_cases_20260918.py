"""Render real predicted AGRI/COT case studies extracted from frozen validation cache."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "audits/physical_cot_cases_20260918_v4"
OUT = ROOT / "figures/cot_pipeline_20260918"


def false_rgb(agri):
    """C03/C02/C01 false colour with one robust stretch shared by the four frames."""
    rgb = np.moveaxis(agri[:, [2, 1, 0]], 1, -1).astype(np.float64)
    low = np.quantile(rgb, .02, axis=(0, 1, 2), keepdims=True)
    high = np.quantile(rgb, .98, axis=(0, 1, 2), keepdims=True)
    return np.clip((rgb - low) / (high - low), 0, 1)


def text_panel(ax, title, record):
    ax.axis("off")
    colour = "#00796b" if title == "COT correction reduces error" else "#b71c1c"
    observed, a, ac = (record["observed_GHI_Wm2"], record["A_sp_GHI_Wm2"], record["AC_map_GHI_Wm2"])
    ax.text(.02, .93, title, fontsize=13, weight="bold", color=colour, va="top")
    ax.text(.02, .78, f"Station: {record['station']}\nTarget lead: +{record['lead_minutes']} min\n"
                        f"station COT at [8,8]: {record['target_predicted_COT_station_center_log1p']:.2f}", fontsize=10, va="top")
    ax.text(.02, .52, f"Observed GHI: {observed:.1f} W m$^{{-2}}$\n"
                        f"AGRI-only: {a:.1f}\n"
                        f"AGRI + static COT: {ac:.1f}\n"
                        f"COT correction: {record['COT_prediction_adjustment_Wm2']:+.1f}", fontsize=10, va="top")
    ax.text(.02, .21, "Same type of static COT dimming can\nmove one case toward the observed GHI\nand another away from it. The static map\ndoes not encode the missing cloud evolution.",
            fontsize=9.5, va="top")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT / "actual_predicted_agri_cot_cases.npz", allow_pickle=False) as item:
        data = {key: item[key].copy() for key in item.files}
    record = json.loads((INPUT / "selection.json").read_text(encoding="utf-8"))["selection"]
    cmin = min(float(data["improves_cot"].min()), float(data["worsens_cot"].min()))
    cmax = max(float(data["improves_cot"].max()), float(data["worsens_cot"].max()))
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(15.5, 10.2), layout="constrained")
        grid = fig.add_gridspec(4, 5, width_ratios=(1, 1, 1, 1, 1.48))
        image = None
        for block, outcome in enumerate(("improves", "worsens")):
            rgb = false_rgb(data[outcome + "_agri"])
            cot = data[outcome + "_cot"]
            frame = data[outcome + "_frame_lead"]
            base_row = 2 * block
            for column in range(4):
                ax_rgb = fig.add_subplot(grid[base_row, column])
                ax_rgb.imshow(rgb[column], interpolation="nearest")
                ax_rgb.scatter(8, 8, marker="*", s=105, facecolor="white", edgecolor="black", linewidth=1.1, zorder=3)
                ax_rgb.set(xticks=[], yticks=[], title=f"+{15 * (int(frame[column]) + 1)} min")
                if column == 0:
                    ax_rgb.set_ylabel("Predicted AGRI\nfalse-colour", fontsize=10)
                ax_cot = fig.add_subplot(grid[base_row + 1, column])
                image = ax_cot.imshow(cot[column], cmap="magma", vmin=cmin, vmax=cmax, interpolation="nearest")
                ax_cot.contour(cot[column], colors="white", linewidths=.55, levels=4)
                ax_cot.scatter(8, 8, marker="*", s=105, facecolor="white", edgecolor="black", linewidth=1.1, zorder=3)
                ax_cot.set(xticks=[], yticks=[])
                if column == 0:
                    ax_cot.set_ylabel("Predicted\nlog1p(COT)", fontsize=10)
            panel = fig.add_subplot(grid[base_row:base_row + 2, 4])
            text_panel(panel, "COT correction reduces error" if outcome == "improves" else "COT correction increases error", record[outcome])
        colourbar = fig.colorbar(image, ax=[axis for axis in fig.axes if axis.has_data()], shrink=.72, location="right")
        colourbar.set_label("frozen R predicted log1p(COT)")
        fig.suptitle("Real frozen-cache validation sequences: static COT can apply the same optical correction with opposite GHI outcome",
                     fontsize=15, weight="bold")
        fig.text(.01, .005, "All maps are actual predicted AGRI/COT from saved validation rows. AGRI display uses a per-sequence robust false-colour stretch; it does not alter model inputs. Cases are predeclared median-magnitude outcomes among high station-centre COT, top-decile COT-adjustment rows.", fontsize=8.5, color="#555555")
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"fig_actual_physical_cot_cases_20260918.{extension}", dpi=240 if extension == "png" else None, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
