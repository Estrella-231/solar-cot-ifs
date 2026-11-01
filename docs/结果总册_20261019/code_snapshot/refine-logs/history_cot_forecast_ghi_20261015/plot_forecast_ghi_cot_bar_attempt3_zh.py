from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "independent_score_attempt3.json"
OUT = HERE / "forecast_ghi_cot_bar_attempt3_zh"


def chinese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available:
            return name
    return "DejaVu Sans"


def main():
    d = json.loads(SOURCE.read_text(encoding="utf-8"))
    tr = d["results"]["transport"]["0"]
    hy = d["results"]["hybrid"]["0"]
    # Keep the no-COT reference and the requested generic +COT result.
    # The retained +COT value is the original hybrid-correction arm.
    labels = ["无 COT", "+COT"]
    values = [tr["A_rmse"], hy["COT_rmse"]]
    deltas = [0.0, hy["delta_rmse"]]
    # The stored field is reduction_pct, whose sign is opposite to RMSE worsening.
    pct = [0.0, 100.0 * deltas[1] / values[0]]

    plt.rcParams.update({
        "font.family": chinese_font(),
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.labelsize": 13,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout="constrained")
    colors = ["#667F8C", "#D55E00"]
    bars = ax.bar(range(2), values, color=colors, width=0.66, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("GHI RMSE（W/m²）")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title("加入预测 COT 后，四里 GHI 预报 RMSE 未下降", pad=18, fontweight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for i, (bar, val, delta, change) in enumerate(zip(bars, values, deltas, pct)):
        text = f"{val:.2f}"
        if i:
            text += f"\nΔ {delta:+.2f}（RMSE +{change:.2f}%）"
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.2, text,
                ha="center", va="bottom", fontsize=11, fontweight="bold" if i else "normal")

    ci_h = hy["date_bootstrap_95ci_delta"]
    note = (
        f"留出计分：{d['score_init_date_count']} 个初始化日期、{d['scoring_rows']} 条配对样本；3 个 seed。\n"
        f"日期块差值 95% CI：+COT−无 COT [{ci_h[0]:+.2f}, {ci_h[1]:+.2f}] W/m²。\n"
        "旧合同、四里单站探索性验证；+COT对应原混合修正组，含历史 COT；两组共用 SimVP 预测 AGRI；test 未使用。"
    )
    fig.text(0.5, -0.025, note, ha="center", va="top", fontsize=9.5, color="#333333", linespacing=1.5)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT.with_suffix('.png')}")
    print(f"Saved: {OUT.with_suffix('.pdf')}")
    print(f"No COT {values[0]:.2f}; +COT {values[1]:.2f} W/m2")


if __name__ == "__main__":
    main()
