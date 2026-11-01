#!/usr/bin/env python3
"""Render transparent, validation-only paired RMSE contrasts from saved JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CPP_GROUPS = [
    ("all_validation", "All validation"),
    ("CPP_nearzero_0_0.1", "CPP COT 0–0.1"),
    ("CPP_thin_0.1_5", "CPP COT 0.1–5"),
    ("CPP_moderate_5_30", "CPP COT 5–30"),
    ("CPP_thick_30_100", "CPP COT 30–100"),
]
CLP_GROUPS = [
    ("CLP_clear_0", "CPP class 0: clear"),
    ("CLP_water_1", "CPP class 1: water cloud"),
    ("CLP_ice_2", "CPP class 2: ice cloud"),
]
CONTRASTS = [
    ("M11_minus_M10", "Joint COT + cloud score − COT"),
    ("M11_minus_M01", "Joint COT + cloud score − cloud score"),
]
COLORS = {"M11_minus_M10": "#0072B2", "M11_minus_M01": "#D55E00"}


def write_plot(summary: dict, groups: list[tuple[str, str]], output: Path, title: str, note: str) -> None:
    lookup = {g["group"]: g for g in summary["groups"]}
    selected = [(key, label, lookup[key]) for key, label in groups if key in lookup]
    if len(selected) != len(groups):
        missing = [key for key, _ in groups if key not in lookup]
        raise RuntimeError(f"missing required groups: {missing}")

    fig = plt.figure(figsize=(12.4, max(4.8, 0.72 * len(selected) + 2.25)), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.20])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1], sharey=None)]
    axes[1].sharey(axes[0])
    offsets = {"M11_minus_M10": -0.13, "M11_minus_M01": 0.13}
    for ax, (contrast_key, contrast_label) in zip(axes, CONTRASTS):
        for i, (_, label, group) in enumerate(selected):
            c = group["contrasts"][contrast_key]
            point = float(c["RMSE_delta_Wm2"])
            low, high = map(float, c["date_block_CI_delta_Wm2"])
            y = i + offsets[contrast_key]
            ax.errorbar(point, y,
                        xerr=np.asarray([[point - low], [high - point]]),
                        fmt="o", markersize=5.5, capsize=3.0, elinewidth=1.5,
                        color=COLORS[contrast_key], markeredgecolor="white", markeredgewidth=0.5,
                        zorder=3)
        ax.axvline(0, color="#333333", linewidth=1.1, linestyle="--", zorder=1)
        ax.grid(axis="x", color="#D8DDE3", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Δ RMSE (W m$^{-2}$); negative = lower RMSE", fontsize=9.5)
        ax.set_title(contrast_label, fontsize=10.5, pad=9)
        ax.tick_params(axis="x", labelsize=8.5)

    axes[0].set_yticks(np.arange(len(selected)))
    ylabels = []
    for _, label, group in selected:
        ylabels.append(f"{label}\nN={group['n']:,}; shared-support blocks={group['common_two_day_blocks']}")
    axes[0].set_yticklabels(ylabels, fontsize=8.7)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=13, fontweight="semibold")
    footer = fig.add_subplot(grid[1, :])
    footer.axis("off")
    footer.text(0.5, 0.48, note, ha="center", va="center", fontsize=7.8, color="#3F4750", wrap=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("state") != "COMPLETE_VALIDATION_CPP_CLP_STRATA" or summary.get("test_payloads_read") != 0:
        raise RuntimeError("validation/test provenance gate failed")
    output_dir = args.output_dir
    note = ("Points are paired candidate-minus-baseline RMSE; whiskers are pointwise 95% percentile CIs from 2,000 "+
            "shared two-day BJT date-block bootstrap replicates sampled from all 26 frozen blocks (seed 20261020); "+
            "draws with zero stratum support for any station/seed are rejected. No multiplicity adjustment. "+
            "CPP COT/CLP are co-produced retrieval references, not independent truth; CPP-matched strata cover "+
            "1,265/4,226 validation rows (29.9%). Same-time validation only; not an independent test.")
    cpp_note = note + " COT bins use the station-centre pixel [8,8] and fixed edges [0,0.1), [0.1,5), [5,30), [30,100]."
    clp_note = note + " CLP classes 0/1/2 denote CPP clear/water-cloud/ice-cloud classes at [8,8], not official CLM."
    write_plot(summary, CPP_GROUPS, output_dir / "paired_delta_by_cpp_cot",
               "Four-arm GHI comparison by CPP-reference COT", cpp_note)
    write_plot(summary, CLP_GROUPS, output_dir / "paired_delta_by_clp_class",
               "Four-arm GHI comparison by CPP CLP class", clp_note)

    manifest = {
        "state": "FIGURES_RENDERED",
        "source_summary": str(args.summary),
        "outputs": [
            "paired_delta_by_cpp_cot.png", "paired_delta_by_cpp_cot.pdf",
            "paired_delta_by_clp_class.png", "paired_delta_by_clp_class.pdf",
        ],
        "software": {"matplotlib": plt.matplotlib.__version__, "numpy": np.__version__},
        "rendering": {"raster_dpi": 300, "background": "opaque white", "format": ["PNG", "PDF"]},
        "statistic": "equal-station/equal-seed candidate-minus-baseline RMSE; paired two-day BJT date-block bootstrap CI",
        "bootstrap": summary["bootstrap"],
        "test_payloads_read": 0,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "alt_text.txt").write_text(
        "Two paired forest plots show GHI RMSE differences for the joint COT-plus-cloud-score input versus COT-only and cloud-score-only. Negative values favor the joint input. The all-validation point estimates are slightly negative but their 95% date-block intervals cross zero. The CPP COT and CLP subset plots show heterogeneous cloud-regime effects, with far fewer matched samples than the full validation set. These are exploratory same-time validation results, not independent-test claims.",
        encoding="utf-8")


if __name__ == "__main__":
    main()
