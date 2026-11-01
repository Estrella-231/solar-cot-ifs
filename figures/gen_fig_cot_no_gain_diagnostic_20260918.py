"""Illustrative mechanism figure; it intentionally contains no experiment data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "cot_pipeline_20260918"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260918)


def cloud(center, width=2.4):
    yy, xx = np.mgrid[:16, :16]
    value = 8 * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * width**2))
    return value + .08 * rng.normal(size=(16, 16))


def label(ax, text):
    ax.text(.02, .03, text, transform=ax.transAxes, fontsize=8.5, color="#555555",
            va="bottom", ha="left", style="italic")


def main():
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(14.6, 7.1), layout="constrained")
        grid = fig.add_gridspec(2, 4, width_ratios=(1.1, .15, 1.1, 1.1), height_ratios=(1, 1))

        # A: deterministic representation, not an independent observation.
        ax = fig.add_subplot(grid[0, 0])
        agri_proxy = np.clip(1 - cloud((10.5, 6.5)) / 12, 0, 1)
        ax.imshow(agri_proxy, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.scatter(8, 8, marker="*", s=170, facecolor="white", edgecolor="black", linewidth=1.3, zorder=3)
        ax.set(title="Predicted AGRI at target time", xticks=[], yticks=[])
        label(ax, "same SimVP state already enters the GHI head")
        arrow = fig.add_subplot(grid[0, 1]); arrow.axis("off")
        arrow.annotate("", xy=(.92, .5), xytext=(.08, .5), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#444444"})
        arrow.text(.5, .62, "frozen R", ha="center", fontsize=9)
        ax = fig.add_subplot(grid[0, 2])
        c = ax.imshow(cloud((10.5, 6.5)), cmap="magma", vmin=0, vmax=8, interpolation="nearest")
        ax.scatter(8, 8, marker="*", s=170, facecolor="white", edgecolor="black", linewidth=1.3, zorder=3)
        ax.set(title="Predicted COT at the same target time", xticks=[], yticks=[])
        label(ax, "COT = R(predicted AGRI), not an extra observation")
        fig.colorbar(c, ax=ax, shrink=.78, label="illustrative log1p(COT)")

        ax = fig.add_subplot(grid[0, 3]); ax.axis("off")
        ax.text(.03, .82, "A target-time COT map can help", fontsize=13, weight="bold", color="#00796b")
        ax.text(.03, .61, "when R makes cloud optical thickness easier\nfor a finite GHI head to use.", fontsize=11)
        ax.text(.03, .35, "It has no guaranteed new information:\nCOT is a deterministic re-expression\nof the same predicted AGRI field.", fontsize=11)
        ax.text(.03, .09, "Thus an AC_F−A_ST gain is an empirical\nrepresentation result, not automatic.", fontsize=10, color="#6a1b9a")

        # B: target frame versus interval label.
        ax = fig.add_subplot(grid[1, :2])
        minute = np.linspace(0, 15, 200)
        cot_station = 2.0 + 5.6 / (1 + np.exp(-(minute - 9) / 1.3))
        ghi = 760 - 70 * cot_station + 22 * np.sin(minute / 2)
        ax.plot(minute, cot_station, color="#8e24aa", lw=2.4, label="station COT trajectory (illustrative)")
        ax.set(xlim=(0, 15), xlabel="minutes inside target GHI window", ylabel="COT proxy", title="A single COT frame is an endpoint snapshot")
        ax.axvspan(0, 15, color="#ffcc80", alpha=.20, label="label window: mean GHI")
        ax.axvline(15, color="#212121", ls="--", lw=1.5)
        ax.annotate("target COT map\nused at +15 min", xy=(15, cot_station[-1]), xytext=(10.1, 7.1), arrowprops={"arrowstyle": "->"}, fontsize=9)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        twin = ax.twinx(); twin.plot(minute, ghi, color="#f57c00", lw=1.7, ls="--", label="instantaneous GHI proxy")
        twin.set_ylabel("GHI proxy")
        label(ax, "label is interval mean; target frame alone may miss within-window cloud motion")

        # C: forecast error is converted and can move cloud relative to station.
        ax = fig.add_subplot(grid[1, 2])
        true = cloud((8.0, 7.5), 2.0); predicted = cloud((11.5, 7.5), 2.0)
        delta = predicted - true
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-7, vmax=7, interpolation="nearest")
        ax.scatter(8, 8, marker="*", s=170, facecolor="white", edgecolor="black", linewidth=1.3)
        ax.set(title="Forecast-position error becomes COT-position error", xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, shrink=.78, label="illustrative predicted − true COT")
        label(ax, "R cannot restore cloud location lost by SimVP")

        # D: direct operational diagnosis.
        ax = fig.add_subplot(grid[1, 3]); ax.axis("off")
        ax.text(.03, .89, "What the paired ablation resolves", fontsize=12.5, weight="bold")
        entries = [
            ("AC_F − A_ST", "Does the same-time predicted COT map help?"),
            ("AC_ST − AC_F", "Does history + causal prefix add trajectory value?"),
            ("COT retrieval audit", "Is R accurate enough on forecast AGRI?"),
            ("per-lead / cloud regime", "Is any gain confined to moving thick cloud?"),
        ]
        y = .72
        for title, body in entries:
            ax.text(.04, y, title, fontsize=10.5, weight="bold", color="#1565c0")
            ax.text(.04, y - .09, body, fontsize=9.5)
            y -= .21
        fig.suptitle("Why a predicted COT map at every future time may still show no GHI gain\n"
                     "Illustrative mechanisms and planned tests; this figure contains no experiment values.",
                     fontsize=15, weight="bold")
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"fig_predicted_cot_no_gain_diagnostic_20260918.{extension}", dpi=220 if extension == "png" else None,
                        facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
