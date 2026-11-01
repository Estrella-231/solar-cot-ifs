"""Build the manuscript locator and observed-date-support figure.

All dates and coordinates are copied from the audited current-use inventory and
the station/grid source scripts. Date lines show first-to-last support only;
they do not imply continuous daily availability.
"""
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


OUT = Path(__file__).resolve().parent

STATIONS = {
    "Hunan": [
        ("Sili", 112.347916667, 25.9478611111, "#0072B2", "o"),
        ("Zhujia", 112.45, 29.566666667, "#D55E00", "s"),
    ],
    "Hubei": [
        ("Niushou", 112.0400, 32.0800, "#009E73", "o"),
        ("Wangzhai", 112.1200, 32.0400, "#CC79A7", "s"),
        ("Mizhuang", 112.2000, 32.1200, "#E69F00", "^"),
    ],
}

SUPPORT = [
    ("Hunan", "AGRI / GHI samples", "2024-04-02", "2026-06-30", "#4C78A8", "-"),
    ("Hunan", "CPP-supervised retrieval", "2024-04-02", "2026-03-02", "#F58518", "--"),
    ("Hubei", "AGRI / GHI samples", "2025-08-01", "2026-07-14", "#4C78A8", "-"),
    ("Hubei", "CPP-supervised retrieval", "2025-08-01", "2026-03-02", "#F58518", "--"),
]


def plot_region(ax, name, panel, label_offsets):
    points = STATIONS[name]
    lat0 = sum(p[2] for p in points) / len(points)
    lon0 = sum(p[1] for p in points) / len(points)
    xy = []
    for label, lon, lat, color, marker in points:
        x = (lon - lon0) * 111.32 * __import__("math").cos(__import__("math").radians(lat0))
        y = (lat - lat0) * 110.57
        xy.append((label, x, y, color, marker))
    for (label, x, y, color, marker), (dx, dy) in zip(xy, label_offsets):
        ax.scatter(x, y, s=44, c=color, marker=marker, edgecolor="white", linewidth=0.7,
                   zorder=3, label=label)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha="left" if dx >= 0 else "right", va="center")
    ax.set_title(f"({panel}) {name}", loc="left", fontweight="bold", fontsize=10)
    ax.set_xlabel("East–west distance (km)", fontsize=8)
    ax.set_ylabel("North–south distance (km)", fontsize=8)
    spread = max(max(abs(x), abs(y)) for _, x, y, _, _ in xy)
    radius = max(spread * 1.45, 14)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius * 0.72, radius * 0.72)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=7, length=2)
    ax.grid(color="#D9DEE5", linewidth=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#7B8794")
        spine.set_linewidth(0.6)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelcolor": "#263238",
        "text.color": "#263238",
        "axes.titlecolor": "#17252A",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.15, 3.7))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 0.9), hspace=0.6, wspace=0.38,
                          left=0.15, right=0.98, top=0.95, bottom=0.20)
    ax_hn = fig.add_subplot(gs[0, 0])
    ax_hb = fig.add_subplot(gs[0, 1])
    plot_region(ax_hn, "Hunan", "a", [(7, 8), (7, -9)])
    plot_region(ax_hb, "Hubei", "b", [(6, -11), (5, -2), (6, 8)])

    ax = fig.add_subplot(gs[1, :])
    origin = datetime(2024, 1, 1)
    for y, row in enumerate(SUPPORT):
        region, label, start, end, color, style = row
        yy = 4 - y
        x0 = (datetime.fromisoformat(start) - origin).days
        x1 = (datetime.fromisoformat(end) - origin).days
        ax.plot([x0, x1], [yy, yy], color=color, linestyle=style, linewidth=3.2,
                solid_capstyle="round")
        ax.scatter([x0, x1], [yy, yy], color=color, s=12, zorder=3)
        ax.text(x0 + 10, yy + 0.17, start[:7], ha="left", va="bottom", fontsize=6.5, color=color)
        ax.text(x1 - 10, yy + 0.17, end[:7], ha="right", va="bottom", fontsize=6.5, color=color)
    ticks = [datetime(y, 1, 1) for y in (2024, 2025, 2026)]
    ax.set_xticks([(t - origin).days for t in ticks], [str(t.year) for t in ticks])
    ax.set_xlim(0, (datetime(2026, 9, 1) - origin).days)
    ax.set_ylim(0.4, 4.6)
    ax.set_yticks([4, 3, 2, 1], ["Hunan · AGRI/GHI", "Hunan · CPP retrieval", "Hubei · AGRI/GHI", "Hubei · CPP retrieval"])
    ax.tick_params(axis="y", labelsize=7, length=0, pad=5)
    ax.grid(axis="x", color="#E0E4E8", linewidth=0.55)
    ax.set_xlabel("Date span of available records (BJT)", fontsize=8, labelpad=2)
    ax.set_title("(c) Observed support by region and source", loc="left", fontsize=9, fontweight="bold")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#7B8794")
    fig.savefig(OUT / "fig1_study_regions_coverage.pdf", facecolor="white")
    fig.savefig(OUT / "fig1_study_regions_coverage.png", dpi=320, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
