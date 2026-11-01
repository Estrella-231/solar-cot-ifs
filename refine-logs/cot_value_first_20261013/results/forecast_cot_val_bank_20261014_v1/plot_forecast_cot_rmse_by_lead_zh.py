from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
CSV = HERE / "forecast_cot_lead_metrics_fp64.csv"
OUT = HERE / "forecast_cot_rmse_by_lead_zh_20261016"


def choose_chinese_font():
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def main():
    df = pd.read_csv(CSV)
    required = {
        "station", "lead_min", "n_pixels", "forecast_simvp_to_R_rmse_cot",
        "persistence_R_rmse_cot", "offline_true_future_AGRI_to_R_rmse_cot",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing metric values found; refusing to draw a connected line across gaps")

    plt.rcParams.update({
        "font.family": choose_chinese_font(),
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharey=True, layout="constrained")
    styles = [
        ("forecast_simvp_to_R_rmse_cot", "SimVP预测AGRI → R", "#0072B2", "-", "o"),
        ("offline_true_future_AGRI_to_R_rmse_cot", "真实未来AGRI → R（离线参照）", "#D55E00", ":", "s"),
        ("persistence_R_rmse_cot", "末帧持续性AGRI → R", "#666666", "--", "^"),
    ]
    station_titles = {"sili": "四里", "zhujia": "竺家"}
    for ax, station in zip(axes, ("sili", "zhujia")):
        sub = df.loc[df.station.eq(station)].sort_values("lead_min")
        if len(sub) != 16 or sub.lead_min.tolist() != list(range(15, 241, 15)):
            raise ValueError(f"Unexpected lead coverage for {station}: {sub.lead_min.tolist()}")
        for col, label, color, ls, marker in styles:
            ax.plot(sub.lead_min, sub[col], label=label, color=color, linestyle=ls,
                    marker=marker, markersize=4.2, linewidth=2.0)
        ax.set_title(station_titles[station])
        ax.set_xlabel("距历史末帧的预报时效（分钟）")
        ax.set_xticks(range(15, 241, 30))
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("COT RMSE（物理单位）")
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle("湖南未来 COT 逐时效误差：预测云图与真实未来云图对照", fontsize=16, fontweight="bold")
    fig.text(
        0.5, -0.025,
        "16×16站点裁块；共同CPP有效像元上计算。真实未来AGRI→R为离线参照，预报时不可用。"
        "验证集仅271个窗口、8个CPP支持日期，窗口重叠；CPP是参考产品，不是独立真值。",
        ha="center", va="top", fontsize=9, color="#333333",
    )
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    df.to_csv(OUT.with_name(OUT.name + "_source_metrics.csv"), index=False)
    print(f"Saved: {OUT.with_suffix('.png')}")
    print(f"Saved: {OUT.with_suffix('.pdf')}")
    print(f"Source: {CSV}")


if __name__ == "__main__":
    main()
