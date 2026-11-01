"""Show the 16x16 aligned COT/AGRI station-patch representation conceptually."""
import matplotlib.pyplot as plt
import numpy as np

from cot_pipeline_visual_style import PALETTE, fields, save, stamp, style


def grid(ax):
    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.35, alpha=.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([0, 8, 15]); ax.set_yticks([0, 8, 15])
    ax.set_xlabel("column"); ax.set_ylabel("row")


def main():
    style()
    cot, rgb = fields()
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.35), layout="constrained")
    axes[0].imshow(rgb, origin="upper", interpolation="nearest")
    axes[0].set_title("Predicted AGRI false-colour proxy", fontsize=9)
    grid(axes[0]); stamp(axes[0])
    im = axes[1].imshow(cot, origin="upper", cmap="magma", vmin=0, vmax=12,
                        interpolation="nearest")
    axes[1].set_title("Predicted COT map", fontsize=9)
    grid(axes[1]); stamp(axes[1])
    fig.colorbar(im, ax=axes[1], fraction=.048, pad=.04, label="illustrative COT")
    axes[2].imshow(rgb, origin="upper", interpolation="nearest", alpha=.72)
    axes[2].contour(cot, levels=[3, 6, 9], colors=["#FDE725", "#35B779", "#440154"], linewidths=1.4)
    axes[2].scatter(8, 8, marker="*", s=135, color="white", edgecolor="black", zorder=5,
                    label="station pixel (8, 8)")
    axes[2].add_patch(plt.Rectangle((5.5, 5.5), 5, 5, fill=False, ec="white", lw=1.4,
                                    ls="--", label="5×5 neighbourhood"))
    axes[2].set_title("Pixelwise alignment used by AC_gate", fontsize=9)
    grid(axes[2]); axes[2].legend(loc="lower left", fontsize=7, frameon=True)
    stamp(axes[2])
    save(fig, "fig_cot_patch_alignment_20260918")


if __name__ == "__main__":
    main()
