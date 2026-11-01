"""Draw the paper's COT estimation and forecast-input workflow."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none"})

fig, ax = plt.subplots(figsize=(12.0, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis("off")

INK = "#252a30"
BLUE = "#eaf1f7"
TEAL = "#e8f3f0"
GOLD = "#f7f1e5"
WHITE = "#ffffff"


def box(x, y, w, h, title, detail="", face=WHITE, edge=INK, lw=1.1, dashed=False):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.10",
                           linewidth=lw, edgecolor=edge, facecolor=face,
                           linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=INK,
            fontsize=9.2, fontweight="bold", wrap=True)
    if detail:
        ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", color=INK,
                fontsize=7.5, linespacing=1.25, wrap=True)


def arrow(x1, y1, x2, y2, label=None, style="-"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                          linewidth=1.05, color=INK, linestyle=style,
                          connectionstyle="arc3,rad=0")
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + .12, label, ha="center", va="bottom",
                fontsize=7.1, color=INK, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})


ax.text(.20, 5.18, "A  Synchronous irradiance estimation", ha="left", va="center",
        fontsize=11, fontweight="bold", color=INK)
ax.plot([.2, 11.8], [4.98, 4.98], color=INK, lw=.75)

# Synchronous path: branch from aligned observed imagery into COT retrieval and GHI head.
box(.25, 3.75, 1.8, .85, "Observed AGRI", "13 channels + geometry", BLUE)
box(2.55, 3.75, 2.1, .85, "Frozen COT retriever R", "16×16 patch → COT field", TEAL)
box(5.15, 3.75, 1.8, .85, "COT station crop", "central 3×3 pixels", GOLD)
box(7.45, 3.75, 1.75, .85, "Matched GHI head", "AGRI + geometry + COT")
box(9.75, 3.75, 1.9, .85, "15-min GHI", "kt prediction × clear-sky GHI", BLUE)
arrow(2.05, 4.17, 2.55, 4.17)
arrow(4.65, 4.17, 5.15, 4.17)
arrow(6.95, 4.17, 7.45, 4.17)
arrow(9.20, 4.17, 9.75, 4.17)
box(2.55, 2.72, 2.1, .58, "CPP supervision", "valid retrieval pixels only", WHITE, edge="#6f7882", lw=.9)
arrow(3.60, 3.30, 3.60, 3.75, style="--")
ax.text(5.15, 3.00, "GHI branch also receives the observed AGRI and solar geometry directly.",
        ha="left", va="center", fontsize=8, color=INK)

ax.text(.20, 2.22, "B  Forecast-input application", ha="left", va="center",
        fontsize=11, fontweight="bold", color=INK)
ax.plot([.2, 11.8], [2.02, 2.02], color=INK, lw=.75)

box(.25, .72, 2.0, .85, "Observed history", "8 frames · regional 256×256", BLUE)
box(2.75, .72, 1.8, .85, "Fixed SimVP", "8 observed → 16 future AGRI")
box(5.05, .72, 1.9, .85, "Forecast AGRI", "+15 to +240 min", BLUE)
box(7.45, .72, 1.8, .85, "Frozen retriever R", "each forecast frame → COT", TEAL)
box(9.75, .72, 1.9, .85, "Forecast evaluation", "COT skill and matched GHI skill", GOLD, dashed=True)
arrow(2.25, 1.15, 2.75, 1.15)
arrow(4.55, 1.15, 5.05, 1.15)
arrow(6.95, 1.15, 7.45, 1.15)
arrow(9.25, 1.15, 9.75, 1.15)

ax.text(.28, .32,
        "Forecast COT is derived from predicted AGRI; it is distinct from COT retrieved from observed future imagery.",
        ha="left", va="center", fontsize=8, color=INK)

fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.05)
fig.savefig(OUT / "fig2_cot_estimation_forecast_workflow.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig2_cot_estimation_forecast_workflow.png", dpi=240, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig2_cot_estimation_forecast_workflow.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
