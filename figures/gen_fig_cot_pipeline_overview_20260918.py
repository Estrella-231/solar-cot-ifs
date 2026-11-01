"""Draw the implemented deployable pathway, using only code-contract facts."""
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

from cot_pipeline_visual_style import PALETTE, save, style


def box(ax, x, y, w, h, label, colour, detail):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                linewidth=1.2, edgecolor=colour, facecolor="#FFFFFF"))
    ax.text(x + w / 2, y + h * .62, label, ha="center", va="center", weight="bold",
            color=colour, fontsize=10)
    ax.text(x + w / 2, y + h * .30, detail, ha="center", va="center", color=PALETTE["ink"],
            fontsize=8, linespacing=1.3)


def arrow(ax, x0, x1, y=.53, label=None):
    ax.annotate("", (x1, y), (x0, y), arrowprops={"arrowstyle": "->", "lw": 1.3,
                                                    "color": PALETTE["ink"]})
    if label:
        ax.text((x0 + x1) / 2, y + .075, label, ha="center", fontsize=7.5,
                color=PALETTE["muted"])


def main():
    style()
    fig, ax = plt.subplots(figsize=(11.2, 3.1))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, .02, .34, .15, .38, "History AGRI", "#0072B2", "8 frames\n13 channels × 16×16")
    box(ax, .23, .34, .15, .38, "Frozen SimVP", "#0072B2", "extrapolates AGRI\n+15 … +240 min")
    box(ax, .44, .34, .15, .38, "Frozen COT R", "#D55E00", "predicted AGRI →\n1 COT map, 16×16")
    box(ax, .65, .34, .15, .38, "COT-gated head", "#009E73", "aligned COT condition\nfor AGRI features")
    box(ax, .86, .34, .12, .38, "GHI", "#009E73", "station kt\n→ GHI")
    arrow(ax, .17, .23, label="future AGRI")
    arrow(ax, .38, .44, label="13 × 16×16")
    arrow(ax, .59, .65, label="COT map")
    arrow(ax, .80, .86, label="lead + station")
    ax.text(.5, .12, "Implemented AC_gate route: only the frozen predicted-AGRI domain is used at inference;"
            " observed GHI and future reference CPP are not inputs.", ha="center", fontsize=8.2,
            color=PALETTE["ink"])
    save(fig, "fig_cot_pipeline_overview_20260918")


if __name__ == "__main__":
    main()
