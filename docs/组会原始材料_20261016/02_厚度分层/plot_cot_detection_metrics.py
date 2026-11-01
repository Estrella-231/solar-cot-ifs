from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / "refine-logs" / "cot_value_first_20261013" / "cot_cloud_detection_monthly_R_seed42_validation.json"
OUT_STEM = HERE / "COT云与厚云检出指标_R验证"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    groups = [
        ("总体\n1,398样本", data["overall"]),
        ("四里\n700样本", data["by_station"]["sili"]),
        ("竺家\n698样本", data["by_station"]["zhujia"]),
    ]
    definitions = [
        ("cloud", "云代理", "COT ≥ 0.1"),
        ("thick_cloud", "厚云", "COT ≥ 30"),
    ]
    metrics = [
        ("pod_recall", "POD（命中率）"),
        ("false_alarm_ratio", "FAR（虚警比）"),
        ("csi", "CSI（综合检出评分）"),
    ]

    records: list[dict[str, object]] = []
    for group_name, group in groups:
        for key, condition_name, threshold in definitions:
            scores = group[key]
            for metric_key, metric_name in metrics:
                records.append(
                    {
                        "group": group_name,
                        "condition": condition_name,
                        "threshold": threshold,
                        "metric": metric_name,
                        "score": scores[metric_key],
                        "n_valid_pixels": scores["valid_pixels"],
                        "n_reference_positive": scores["reference_positive"],
                    }
                )
    pd.DataFrame(records).to_csv(HERE / "COT云与厚云检出指标_R验证.csv", index=False, encoding="utf-8-sig")

    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update(
        {
            "font.family": font.get_name(),
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {"云代理": "#0072B2", "厚云": "#D55E00"}
    x = np.arange(len(groups), dtype=float)
    width = 0.31
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.65), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.78, wspace=0.23)

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for j, (condition_key, condition_name, _threshold) in enumerate(definitions):
            values = [group[condition_key][metric_key] for _, group in groups]
            xpos = x + (j - 0.5) * width
            bars = ax.bar(
                xpos,
                values,
                width=width,
                label=condition_name,
                color=colors[condition_name],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#263746",
                )
        ax.set_title(metric_title, pad=9, fontsize=12)
        ax.set_xticks(x, [name for name, _ in groups])
        ax.set_ylim(0, 1.14)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#d8e0e6", linewidth=0.7, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#55636f")
        ax.spines["bottom"].set_color("#55636f")
        ax.tick_params(axis="both", length=3.5, width=0.8, color="#55636f")

    axes[0].set_ylabel("指标值（0–1）")
    fig.suptitle("COT反演的云与厚云检出表现", fontsize=15, y=0.96, fontweight="semibold")
    fig.text(
        0.5,
        0.875,
        "验证集 R 输出对照 CPP 参考（356,169个共同有效像元）｜云代理：COT≥0.1；厚云：COT≥30｜固定阈值，未在验证集调阈值",
        ha="center",
        va="center",
        fontsize=9.7,
        color="#394956",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.835), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.055,
        "POD越高越好；FAR越低越好；CSI越高越好。总体为像元合并统计，分站分别统计；CPP是检索参考而非独立真值，test载荷未读取。",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#5b6570",
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_STEM.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
