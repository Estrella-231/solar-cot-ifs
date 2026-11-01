"""Two-station no-COT vs forecast-COT GHI comparison from paired_metrics.json."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "paired_metrics.json"
OUT = HERE / "two_station_no_cot_vs_forecast_cot_zh"


def chinese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available:
            return name
    return "DejaVu Sans"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    stations = [("Sili", "四里"), ("Zhujia", "竺家")]
    arms = ["A0", "AI"]
    labels = ["无 COT", "+预测 COT"]
    values = {
        arm: [data["arms"][arm]["stations"][key]["rmse_wm2"] for key, _ in stations]
        for arm in arms
    }

    plt.rcParams.update({
        "font.family": chinese_font(),
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 18,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(9.2, 6.4), layout="constrained")
    x = [0, 1]
    width = 0.34
    colors = ["#667F8C", "#D55E00"]
    bars_by_arm = []
    for i, arm in enumerate(arms):
        positions = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(positions, values[arm], width=width, color=colors[i],
                      edgecolor="white", linewidth=1.1, label=labels[i])
        bars_by_arm.append(bars)
        for j, bar in enumerate(bars):
            val = values[arm][j]
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.8, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=11)

    for j in range(len(stations)):
        delta = values["AI"][j] - values["A0"][j]
        pct = 100 * delta / values["A0"][j]
        top = max(values["A0"][j], values["AI"][j])
        ax.text(x[j], top + 11, f"Δ {delta:+.2f} ({pct:+.2f}%)",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax.set_xticks(x, [name for _, name in stations])
    ax.set_ylabel("GHI RMSE（W/m²）")
    ax.set_ylim(0, max(max(v) for v in values.values()) * 1.22)
    ax.set_title("加入预测 COT 后，四里和竺家 GHI RMSE 均未下降", pad=18, fontweight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    note = (
        "同一验证集、相同 AGRI 预测与 GHI 评价行；99,849 条日间验证样本，单 seed，未使用 test。\n"
        "“+预测 COT”对应 AI：在 8 帧历史 COT 上加入 16 帧 SimVP 预测 AGRI→R 的 COT；并非纯预测 COT 单独输入。"
    )
    fig.text(0.5, -0.018, note, ha="center", va="top", fontsize=9.5,
             color="#333333", linespacing=1.45)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT.with_suffix('.png')}")
    print(f"Saved: {OUT.with_suffix('.pdf')}")
    for key, name in stations:
        a, b = (data["arms"][arm]["stations"][key]["rmse_wm2"] for arm in arms)
        print(f"{name}: no COT={a:.6f}; +forecast COT={b:.6f}; delta={b-a:+.6f} W/m2")


if __name__ == "__main__":
    main()
