from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FORECAST_COT_FIXED_HEAD_ORACLE_AGRI_DIAGNOSTIC.md"
AUDIT = HERE / "fp64_independent_audit.json"
OUT = HERE / "forecast_cot_only_ghi_bar_zh"


def chinese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available:
            return name
    return "DejaVu Sans"


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    # Values come directly from the independent FP64 audit.
    station_rows = [audit["by_station"][s] for s in ("sili", "zhujia")]
    stations = ["四里", "竺家"]
    ref = [r["RMSE_same_time_R_COT"] for r in station_rows]
    forecast = [r["RMSE_forecast_COT_oracle_AGRI"] for r in station_rows]
    delta = [r["delta_RMSE"] for r in station_rows]
    overall_ci = audit["station_equal_delta_RMSE_date_block_bootstrap_95ci"]
    overall_delta = audit["station_equal_delta_RMSE"]

    plt.rcParams.update({
        "font.family": chinese_font(),
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(9.5, 6.1), layout="constrained")
    x = [0, 1]
    width = 0.31
    b0 = ax.bar([i - width / 2 for i in x], ref, width, label="真实未来 AGRI → R 的 COT", color="#667F8C")
    b1 = ax.bar([i + width / 2 for i in x], forecast, width, label="SimVP 预测 AGRI → R 的 COT", color="#008C83")
    ax.set_xticks(x, stations)
    ax.set_ylabel("GHI RMSE（W/m²）")
    ax.set_ylim(0, 108)
    ax.set_title("将 SimVP 预测云图反演的 COT 输入 GHI 头，误差未下降", pad=16, fontweight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for bars in (b0, b1):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.4,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=11)
    for i, d in enumerate(delta):
        ax.text(i, max(ref[i], forecast[i]) + 10.5, f"Δ {d:+.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222")
    fig.text(
        0.5, -0.025,
        f"同一冻结 AC 头；真实目标 AGRI、太阳几何、GHI 标签与权重均固定，只替换 COT 输入。"
        f"两站等权差值 {overall_delta:+.2f} W/m²；日期块 95% CI [{overall_ci[0]:+.2f}, {overall_ci[1]:+.2f}]，跨 0。\n"
        "离线归因：预报时真实目标 AGRI 不可用；验证窗口受 CPP 覆盖筛选（8 个目标日期），不能作为端到端业务预报分数。",
        ha="center", va="top", fontsize=9.5, color="#333333", linespacing=1.5,
    )
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT.with_suffix('.png')}")
    print(f"Saved: {OUT.with_suffix('.pdf')}")
    print(f"Sili: {ref[0]:.2f} -> {forecast[0]:.2f}; Zhujia: {ref[1]:.2f} -> {forecast[1]:.2f}")


if __name__ == "__main__":
    main()
