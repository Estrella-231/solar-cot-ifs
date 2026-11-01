"""Render validation-only COT routing diagnostics and false-cloud case maps."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diagnostic_dir", type=Path)
    args = ap.parse_args()
    root = args.diagnostic_dir
    summary = json.loads((root / "diagnostic_summary.json").read_text())
    data = np.load(root / "false_cloud_cases.npz")
    target = data["cpp_cot"][:, 0]
    prob = data["cloud_probability"][:, 0]
    low = data["low_expert_cot"][:, 0]
    cloud = data["cloud_expert_cot"][:, 0]
    final = data["model_route_cot"][:, 0]
    oracle = data["oracle_route_cot"][:, 0]
    raw = data["raw_agri"]
    meta = summary["cases"]
    n = len(meta)
    cot_cmap = "magma"
    cot_max = 40.0
    fig, axes = plt.subplots(n, 7, figsize=(20, max(4.2, 3.2 * n)), layout="constrained")
    if n == 1:
        axes = axes[None, :]
    titles = ["AGRI pseudo-RGB\nR=C03, G=C02, B=C01", "CPP reference COT", "P(cloud)",
              "Low-COT expert\nscale 0–0.1", "Cloud expert", "Model routed COT", "CPP-oracle routed COT"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=10)
    for i in range(n):
        row = meta[i]
        rgb = np.stack([raw[i, 2], raw[i, 1], raw[i, 0]], axis=-1)
        for k, ch in enumerate((2, 1, 0)):
            lo, hi = summary["rgb_channel_limits_1_99pct"][str(ch)]
            rgb[..., k] = np.clip((rgb[..., k] - lo) / max(hi - lo, 1e-6), 0, 1)
        y, x = int(row["selected_pixel_row"]), int(row["selected_pixel_col"])
        axes[i, 0].imshow(rgb, interpolation="nearest")
        axes[i, 0].plot(x, y, marker="x", color="white", markeredgewidth=2, markersize=8)
        images = [
            axes[i, 1].imshow(target[i], cmap=cot_cmap, vmin=0, vmax=cot_max, interpolation="nearest"),
            axes[i, 2].imshow(prob[i], cmap="viridis", vmin=0, vmax=1, interpolation="nearest"),
            axes[i, 3].imshow(low[i], cmap="viridis", vmin=0, vmax=0.1, interpolation="nearest"),
            axes[i, 4].imshow(cloud[i], cmap=cot_cmap, vmin=0, vmax=cot_max, interpolation="nearest"),
            axes[i, 5].imshow(final[i], cmap=cot_cmap, vmin=0, vmax=cot_max, interpolation="nearest"),
            axes[i, 6].imshow(oracle[i], cmap=cot_cmap, vmin=0, vmax=cot_max, interpolation="nearest"),
        ]
        for ax in axes[i]:
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0, 5, 10, 15])
            ax.set_xlabel("column")
        axes[i, 0].set_ylabel(f"{row['station_id']}\n{row['timestamp_bjt']}\nrow")
        axes[i, 1].plot(x, y, marker="x", color="cyan", markeredgewidth=2, markersize=8)
        axes[i, 2].plot(x, y, marker="x", color="white", markeredgewidth=2, markersize=8)
        axes[i, 5].plot(x, y, marker="x", color="cyan", markeredgewidth=2, markersize=8)
        axes[i, 6].plot(x, y, marker="x", color="cyan", markeredgewidth=2, markersize=8)
        axes[i, 5].set_title(f"Model routed COT\nFP pixel={row['selected_false_cloud_prediction_cot']:.1f}", fontsize=9)
        axes[i, 6].set_title(f"CPP-oracle routed COT\nCPP pixel={row['selected_CPP_reference_cot']:.3f}", fontsize=9)
    fig.colorbar(images[0], ax=axes[:, [1, 4, 5, 6]].ravel().tolist(),
                 label="COT (retrieval units; display clipped at 40)", shrink=0.78)
    fig.colorbar(images[2], ax=axes[:, 3].ravel().tolist(), label="Low expert COT (display scale 0–0.1)", shrink=0.78)
    fig.colorbar(images[1], ax=axes[:, 2].ravel().tolist(), label="Cloud-class probability", shrink=0.78)
    fig.suptitle("Validation cases: CPP near-zero pixels routed to the cloud expert\n"
                 "AGRI pseudo-RGB uses one fixed validation-wide 1st–99th percentile stretch per band",
                 fontsize=14)
    fig.savefig(root / "false_cloud_case_montage.png", dpi=180)
    plt.close(fig)

    route = summary["route_comparison"]
    names = ["Frozen R", "Model gate", "CPP oracle gate"]
    keys = ["frozen_R", "model_classification_route", "CPP_reference_class_route_oracle_diagnostic"]
    focus = ["all", "cloud_cot_ge0p1", "thick_cot_ge30", "station_5x5"]
    labels = ["All pixels", "CPP COT ≥ 0.1", "CPP COT ≥ 30", "Station 5×5"]
    near_vals = [route[k]["near_zero_cot_lt0p1"]["rmse"] for k in keys]
    other_vals = np.asarray([[route[k][g]["rmse"] for g in focus] for k in keys])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    xx = np.arange(len(focus))
    width = 0.24
    for j, (name, vals) in enumerate(zip(names, other_vals)):
        ax[0].bar(xx + (j - 1) * width, vals, width, label=name)
    ax[0].set_xticks(xx, labels, rotation=12, ha="right")
    ax[0].set_ylabel("COT RMSE")
    ax[0].set_title("Other validation strata")
    ax[0].legend(frameon=False)
    ax[1].bar(np.arange(3), near_vals, color=["#547c8a", "#cc7a00", "#238b75"])
    ax[1].set_xticks(np.arange(3), names, rotation=12, ha="right")
    ax[1].set_ylabel("COT RMSE")
    ax[1].set_title("Near-zero CPP pixels (COT < 0.1)")
    for j, value in enumerate(near_vals):
        ax[1].text(j, value, f"{value:.3f}", ha="center", va="bottom")
    fig.suptitle("Oracle routing isolates gate errors; CPP reference is a retrieval label, not independent truth")
    fig.savefig(root / "routing_error_decomposition.png", dpi=180)
    plt.close(fig)

    context = summary["route_comparison"]["near_zero_context"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")
    edge_names = ["Within 1 pixel\nof CPP cloud", "Farther from\nCPP cloud"]
    edge_values = [context["within_one_pixel_of_CPP_cloud_fraction_false_cloud_rate"],
                   context["farther_than_one_pixel_from_CPP_cloud_false_cloud_rate"]]
    ax[0].bar(edge_names, np.array(edge_values) * 100, color=["#cc6677", "#4477aa"])
    ax[0].set_ylabel("False-cloud rate among near-zero pixels (%)")
    ax[0].set_title("By CPP cloud-edge proximity")
    sun_names = ["cosSOZ < 0.2\n(lower sun)", "cosSOZ ≥ 0.2\n(higher sun)"]
    sun_values = [context["low_sun_cosSOZ_lt0p2_false_cloud_fraction"],
                  context["higher_sun_cosSOZ_ge0p2_false_cloud_fraction"]]
    ax[1].bar(sun_names, np.array(sun_values) * 100, color=["#aa4499", "#228833"])
    ax[1].set_ylabel("False-cloud rate among near-zero pixels (%)")
    ax[1].set_title("By cosSOZ")
    for a in ax:
        for bar in a.patches:
            a.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.1f}%",
                   ha="center", va="bottom")
    fig.suptitle("Classification errors concentrate near CPP cloud boundaries; sun-angle rates are similar")
    fig.savefig(root / "error_context_rates.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
