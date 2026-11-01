from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "day_assignment.csv"
OUT = HERE / "monthly_split_timeline_zh"


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assignment = {date.fromisoformat(r["date_BJT"]): r["split"] for r in rows}
    start, end = min(assignment), max(assignment)
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)

    colors = {"train": "#D9DEE3", "val": "#D55E00", "test": "#0072B2", "buffer": "#E6B84A"}
    plt.rcParams.update({
        "font.family": next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
                              if n in {x.name for x in font_manager.fontManager.ttflist}), "DejaVu Sans"),
        "axes.unicode_minus": False,
        "font.size": 11,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(14, 3.4), layout="constrained")
    # The training line is the background; the other dates are individually marked.
    ax.hlines(0, mdates.date2num(start), mdates.date2num(end), color=colors["train"],
              linewidth=5, label="训练", zorder=1)
    markers = {"val": ("o", "验证"), "test": ("^", "测试"), "buffer": ("|", "缓冲")}
    sizes = {"val": 46, "test": 46, "buffer": 115}
    for split, (marker, label) in markers.items():
        xs = [mdates.date2num(dt) for dt in dates if assignment.get(dt) == split]
        ax.scatter(xs, [0] * len(xs), s=sizes[split], marker=marker, linewidths=1.2,
                   color=colors[split], label=label, zorder=3)
    ax.set_xlim(mdates.date2num(start) - 10, mdates.date2num(end) + 10)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", color="#E8EAED", linewidth=0.7)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#70777D")
    ax.set_title("湖南新月度划分的日期覆盖", fontsize=18, fontweight="bold", pad=15)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    counts = Counter(assignment.values())
    val_dates = [d for d, s in assignment.items() if s == "val"]
    caption = (
        f"全范围 {start:%Y-%m-%d} 至 {end:%Y-%m-%d}；验证日期 {len(val_dates)} 天，分散在各月份（每个合格月份2天），"
        f"验证覆盖 {min(val_dates):%Y-%m-%d} 至 {max(val_dates):%Y-%m-%d}。\n"
        f"训练 {counts['train']} 天 · 验证 {counts['val']} 天 · 测试 {counts['test']} 天 · 缓冲 {counts['buffer']} 天。"
        "图示为日期划分，不是模型精度；新测试载荷未读取。"
    )
    fig.text(0.5, -0.07, caption, ha="center", va="top", fontsize=10, color="#333333", linespacing=1.5)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT.with_suffix('.png')}")
    print(f"Saved {OUT.with_suffix('.pdf')}")
    print("counts", dict(counts), "val_range", min(val_dates), max(val_dates), "val_days", len(val_dates))


if __name__ == "__main__":
    main()
