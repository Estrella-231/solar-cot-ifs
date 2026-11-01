"""Shared style and deterministic conceptual 16x16 fields for COT figures.

These fields are illustrative only.  They are not satellite observations,
retrieval outputs, or experimental results.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "cot_pipeline_20260918"

PALETTE = {
    "agri": "#0072B2",
    "cot": "#D55E00",
    "fused": "#009E73",
    "aux": "#7A5195",
    "ink": "#202124",
    "muted": "#6B7280",
}


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def fields():
    """A reproducible 16x16 cloud pattern and aligned false-colour proxy."""
    y, x = np.mgrid[:16, :16]
    cot = (
        1.0
        + 7.0 * np.exp(-((x - 10.5) ** 2 / 18 + (y - 5.0) ** 2 / 12))
        + 3.2 * np.exp(-((x - 5.0) ** 2 / 9 + (y - 11.5) ** 2 / 18))
        + 0.9 * (x / 15)
    )
    cot = np.clip(cot, 0, 12)
    # False colour is deliberately a visual proxy, not an FY-4B RGB product.
    cloud = np.clip(cot / 12, 0, 1)
    rgb = np.dstack((0.15 + 0.70 * cloud, 0.32 + 0.53 * cloud, 0.52 + 0.42 * cloud))
    return cot, np.clip(rgb, 0, 1)


def setup_output():
    OUT.mkdir(parents=True, exist_ok=True)


def stamp(ax):
    ax.text(0.01, 0.01, "Illustrative synthetic field", transform=ax.transAxes,
            fontsize=7, color=PALETTE["muted"], va="bottom")


def save(fig, stem):
    setup_output()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
