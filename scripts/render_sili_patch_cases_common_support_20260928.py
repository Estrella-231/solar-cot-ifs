"""Render Sili validation COT cases on the exact shared support mask."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.audit / "case_maps.npz")
    cases = json.loads((a.audit / "case_manifest.json").read_text())
    support = z["common_support"].astype(bool)
    fields = [z["truth"], z["simvp"], z["persistence"], z["transport"], z["learned"]]
    names = ["CPP retrieval reference", "SimVP→R", "Persistence",
             "Pure transport / step 0", "Learned residual"]

    fig, axes = plt.subplots(len(cases), 6, figsize=(16, 2.9 * len(cases)),
                             constrained_layout=True)
    cmap_cot = plt.get_cmap("magma").copy()
    cmap_cot.set_bad("#d9dde3")
    cmap_diff = plt.get_cmap("RdBu_r").copy()
    cmap_diff.set_bad("#d9dde3")
    cot_im = diff_im = None
    for j, name in enumerate(names):
        axes[0, j].set_title(name, fontsize=10)
    axes[0, 5].set_title("Learned − CPP", fontsize=10)
    for i, case in enumerate(cases):
        common = support[i]
        for j, raw in enumerate(fields):
            view = np.ma.masked_where(~common, raw[i])
            im = axes[i, j].imshow(view, origin="upper", cmap=cmap_cot, vmin=0,
                                   vmax=100, interpolation="nearest")
            if cot_im is None:
                cot_im = im
            axes[i, j].scatter([32], [32], marker="+", color="cyan", s=45,
                               linewidths=1.2)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        diff = np.ma.masked_where(~common, z["learned"][i] - z["truth"][i])
        im = axes[i, 5].imshow(diff, origin="upper", cmap=cmap_diff,
                               norm=TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50),
                               interpolation="nearest")
        if diff_im is None:
            diff_im = im
        axes[i, 5].scatter([32], [32], marker="+", color="cyan", s=45,
                           linewidths=1.2)
        axes[i, 5].set_xticks([])
        axes[i, 5].set_yticks([])
        axes[i, 0].set_ylabel(
            f"{case['kind']}\ninit {case['init_day_BJT']}\n+{case['lead_min']} min",
            fontsize=8)
    fig.colorbar(cot_im, ax=axes[:, :5].ravel().tolist(), shrink=.75,
                 label="COT", location="bottom", pad=.02)
    fig.colorbar(diff_im, ax=axes[:, 5].ravel().tolist(), shrink=.75,
                 label="Learned − CPP COT", location="bottom", pad=.02)
    fig.suptitle("Sili local 64×64 validation cases — CPP retrieval reference; "
                 "gray = outside shared valid/support mask; cyan = station",
                 fontsize=12)
    a.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output / "case_maps_common_support.png", dpi=220)
    fig.savefig(a.output / "case_maps_common_support.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
