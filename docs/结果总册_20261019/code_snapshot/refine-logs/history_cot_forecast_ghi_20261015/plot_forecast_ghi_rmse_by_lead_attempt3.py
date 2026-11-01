from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "independent_score_attempt3.json"
OUT = HERE / "forecast_ghi_rmse_by_lead_attempt3_zh"


def chinese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available:
            return name
    return "DejaVu Sans"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for lead in range(15, 241, 15):
        key = str(lead)
        t = report["results"]["transport"][key]
        h = report["results"]["hybrid"][key]
        a_seeds = t["per_seed_rmse"]["A"]
        t_seeds = t["per_seed_rmse"]["transport"]
        h_seeds = h["per_seed_rmse"]["hybrid"]
        if not (len(a_seeds) == len(t_seeds) == len(h_seeds) == 3):
            raise ValueError(f"Expected 3 paired seeds at lead {lead}")
        rows.append({
            "lead_min": lead,
            "n_rows": t["n"],
            "A_no_COT_mean_seed_RMSE_Wm2": sum(a_seeds) / 3,
            "transport_COT_mean_seed_RMSE_Wm2": sum(t_seeds) / 3,
            "hybrid_COT_mean_seed_RMSE_Wm2": sum(h_seeds) / 3,
            "A_seed_RMSE": ";".join(f"{x:.6f}" for x in a_seeds),
            "transport_seed_RMSE": ";".join(f"{x:.6f}" for x in t_seeds),
            "hybrid_seed_RMSE": ";".join(f"{x:.6f}" for x in h_seeds),
        })
    table = pd.DataFrame(rows)

    plt.rcParams.update({
        "font.family": chinese_font(),
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    series = [
        ("A_no_COT_mean_seed_RMSE_Wm2", "无 COT（A）", "#666666", "--", "o"),
        ("transport_COT_mean_seed_RMSE_Wm2", "输运预测 COT", "#0072B2", "-", "s"),
        ("hybrid_COT_mean_seed_RMSE_Wm2", "混合修正预测 COT", "#D55E00", "-.", "^"),
    ]
    for col, label, color, ls, marker in series:
        ax.plot(table.lead_min, table[col], label=label, color=color, linestyle=ls,
                marker=marker, linewidth=2.2, markersize=5)
    ax.set_title("未来 COT 输入对 GHI 预报 RMSE 的影响（四里验证）", fontweight="bold")
    ax.set_xlabel("距历史末帧的预报时效（分钟）")
    ax.set_ylabel("GHI RMSE（W/m²）")
    ax.set_xticks(range(15, 241, 15))
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.5, -0.03,
        "旧合同四里站探索性验证；共48个留出初始化日期、4472条配对样本。"
        "曲线为3个seed逐lead RMSE的算术均值；每组共用SimVP预测AGRI。"
        "日期块总体95% CI：输运COT−A [+2.55,+15.49]，混合COT−A [+2.35,+14.76] W/m²；test未使用。",
        ha="center", va="top", fontsize=9, color="#333333",
    )
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    table.to_csv(OUT.with_name(OUT.name + "_data.csv"), index=False)
    print(f"Saved {OUT.with_suffix('.png')}")
    print(f"Saved {OUT.with_suffix('.pdf')}")
    print(f"Saved {OUT.with_name(OUT.name + '_data.csv')}")


if __name__ == "__main__":
    main()
