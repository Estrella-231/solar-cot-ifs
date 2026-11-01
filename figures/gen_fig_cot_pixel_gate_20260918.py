"""Explain the pointwise COT gate and station-aware readout without result claims."""
import matplotlib.pyplot as plt
import numpy as np

from cot_pipeline_visual_style import PALETTE, fields, save, stamp, style


def main():
    style()
    cot, _ = fields()
    # Solely a visual proxy for a learnt per-pixel scaling term, bounded like tanh(gamma).
    gamma_proxy = 0.55 * np.tanh((cot - cot.mean()) / (cot.std() + 1e-6))
    modulation = 1 + .1 * gamma_proxy
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.15), layout="constrained")
    im0 = axes[0].imshow(cot, cmap="magma", vmin=0, vmax=12, interpolation="nearest")
    axes[0].set_title("COT(r, c)", fontsize=9); stamp(axes[0])
    axes[1].imshow(gamma_proxy, cmap="coolwarm", vmin=-.55, vmax=.55, interpolation="nearest")
    axes[1].set_title("illustrative γ(r, c)", fontsize=9); stamp(axes[1])
    im2 = axes[2].imshow(modulation, cmap="viridis", vmin=.945, vmax=1.055, interpolation="nearest")
    axes[2].set_title("1 + 0.1 tanh(γ)", fontsize=9); stamp(axes[2])
    for ax in axes[:3]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(8, 8, marker="*", s=85, color="white", edgecolor="black")
    fig.colorbar(im0, ax=axes[0], fraction=.048, pad=.03, label="COT")
    fig.colorbar(im2, ax=axes[2], fraction=.048, pad=.03, label="scale")
    axes[3].axis("off")
    axes[3].text(.02, .87, "Implemented operation", fontsize=10, weight="bold", color=PALETTE["fused"])
    axes[3].text(.02, .66, "AGRI features: 13 → 16 → 32\n"
                         "COT gate: 1 → 64, split into γ and β\n"
                         "fused(r,c) = SiLU[agri(r,c) ×\n"
                         "  (1 + 0.1 tanh γ(r,c)) + β(r,c)]\n\n"
                         "Readout concatenates:\n"
                         "• station pixel (8,8)\n"
                         "• 5×5 neighbourhood mean\n"
                         "• full-patch mean\n"
                         "• four COT statistics, geometry,\n"
                         "  lead embedding, station embedding",
                  fontsize=8.4, va="top", color=PALETTE["ink"], linespacing=1.35)
    save(fig, "fig_cot_pixel_gate_20260918")


if __name__ == "__main__":
    main()
