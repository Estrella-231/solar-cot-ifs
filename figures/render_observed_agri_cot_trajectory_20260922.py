"""Validation-only: frozen COT retrieval from all 16 observed future AGRI frames."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from render_station_neighborhood_cot_trajectory_20260922 import (
    LEADS, CENTER, extract_observed, read_manifest, retrieve, sha,
)

RGB = (1, 2, 0)  # R=C02, G=C03, B=C01


def rgb_stack(agri):
    native = np.moveaxis(agri[:, RGB], 1, -1).astype(np.float64)
    low = np.nanquantile(native, .01, axis=(0, 1, 2), keepdims=True)
    high = np.nanquantile(native, .99, axis=(0, 1, 2), keepdims=True)
    rgb = np.clip((native - low) / np.maximum(high - low, 1e-12), 0, 1) ** (1 / 2.2)
    rgb[~np.isfinite(rgb).all(axis=-1)] = .5
    return rgb, low.ravel(), high.ravel()


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite")
    sys.path.insert(0, str(root.parent / "HunanSimVPDDP_20260907" / "scripts"))
    row = read_manifest(root.parent / "HunanSimVPFull_20260906/manifests_v1/hunan_8to16.csv")
    raw = extract_observed(root, row["data_relpaths"].split("|"))
    cot_log, r_sha = retrieve(root, raw)
    rgb, low, high = rgb_stack(raw[:, :13])
    cot_limit = float(np.nanmax(cot_log))

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    figure, axes = plt.subplots(4, 8, figsize=(14.4, 7.3), layout="constrained")
    image = None
    for index, lead in enumerate(LEADS):
        row_index, pair_column = divmod(index, 4)
        left = pair_column * 2
        ax_rgb, ax_cot = axes[row_index, left:left + 2]
        ax_rgb.imshow(rgb[index], interpolation="nearest")
        image = ax_cot.imshow(cot_log[index], cmap="magma", vmin=0, vmax=cot_limit, interpolation="nearest")
        for ax in (ax_rgb, ax_cot):
            ax.scatter(CENTER[1], CENTER[0], marker="*", s=45, facecolor="white", edgecolor="black", linewidth=.7, zorder=3)
            ax.set(xticks=[], yticks=[])
        ax_rgb.set_title(f"+{lead} min: observed AGRI", fontsize=7)
        ax_cot.set_title("R(observed AGRI)", fontsize=7)
    figure.colorbar(image, ax=axes, shrink=.78, label="log1p(COT)")
    figure.suptitle("Frozen COT retrieval from observed future AGRI: 16-lead validation trajectory, Zhujia station", fontsize=13)
    figure.text(.5, .006, "RGB: R=C02, G=C03, B=C01; one shared 1st-99th percentile stretch. Star = station pixel [8,8]. Future observations are diagnostic-only.", ha="center", fontsize=7.5, color="#444444")
    output.mkdir(parents=True)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"observed_agri_cot_trajectory.{extension}", dpi=300)
    plt.close(figure)
    payload = {
        "state": "COMPLETE_VALIDATION_16_LEAD_OBSERVED_AGRI_COT_RETRIEVAL",
        "test_used": False,
        "selection": "same predeclared temporal-median validation sequence as prior figures; no AGRI/COT/GHI/error selection",
        "sequence": "hunan_202508012015",
        "station": "zhujia",
        "station_pixel": list(CENTER),
        "init_time_bjt": "2025-08-02T06:00:00+08:00",
        "leads_minutes": LEADS.tolist(),
        "input_contract": "actual target AGRI plus target-time cosSOZ/cosRAA/day_mask, passed through frozen R",
        "scope": "future observed AGRI is used only for validation error-propagation diagnosis and never enters deployment or GHI-head training",
        "r_checkpoint_sha256": r_sha,
        "rgb_contract": {"red": "C02", "green": "C03", "blue": "C01", "gamma": 2.2, "shared_percentile_stretch": [1, 99], "low": low.tolist(), "high": high.tolist()},
        "output_files": {path.name: sha(path) for path in output.iterdir() if path.suffix in (".png", ".pdf")},
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
