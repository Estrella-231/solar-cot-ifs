"""Observed-AGRI-only version of the station-neighbourhood COT summary."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_station_neighborhood_cot_trajectory_20260922 import (
    CENTER, LEADS, WINDOW, extract_observed, local_metrics, read_manifest, retrieve, sha,
)


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite")
    sys.path.insert(0, str(root.parent / "HunanSimVPDDP_20260907" / "scripts"))
    row = read_manifest(root.parent / "HunanSimVPFull_20260906/manifests_v1/hunan_8to16.csv")
    raw = extract_observed(root, row["data_relpaths"].split("|"))
    log_cot, r_sha = retrieve(root, raw)
    result = local_metrics(log_cot)
    cot_limit = float(np.nanmax(result["local_maps"]))

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16.2, 6.2), layout="constrained")
    grids = fig.add_gridspec(4, 16, height_ratios=(1, .18, 1.2, 1.2))
    image = None
    for frame in range(16):
        ax = fig.add_subplot(grids[0, frame])
        image = ax.imshow(result["local_maps"][frame], cmap="magma", vmin=0, vmax=cot_limit, interpolation="nearest")
        ax.scatter(WINDOW, WINDOW, marker="*", s=32, facecolor="white", edgecolor="black", linewidth=.6, zorder=3)
        ax.scatter(result["peak_col"][frame] - CENTER[1] + WINDOW, result["peak_row"][frame] - CENTER[0] + WINDOW,
                   marker="o", s=23, facecolor="none", edgecolor="#31a354", linewidth=1.0, zorder=4)
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"+{LEADS[frame]}", fontsize=7)
        if frame == 0:
            ax.set_ylabel("R(observed AGRI)\n5x5 COT", fontsize=8)
    cax = fig.add_subplot(grids[1, :])
    cax.axis("off")
    fig.colorbar(image, ax=cax, orientation="horizontal", fraction=.9, pad=.18,
                 label="physical COT; star = station, green circle = local peak")

    ax_center = fig.add_subplot(grids[2, :8])
    ax_peak = fig.add_subplot(grids[2, 8:])
    ax_track = fig.add_subplot(grids[3, :])
    ax_center.plot(LEADS, result["center_cot"], color="#54278f", marker="o", ms=3)
    ax_peak.plot(LEADS, result["peak_cot"], color="#54278f", marker="o", ms=3)
    offsets_col = result["peak_col"] - CENTER[1]
    offsets_row = result["peak_row"] - CENTER[0]
    ax_track.plot(offsets_col, offsets_row, color="#54278f", marker="o", ms=4)
    for index, (x, y) in enumerate(zip(offsets_col, offsets_row), 1):
        ax_track.annotate(str(index), (x, y), xytext=(2, 2), textcoords="offset points", color="#54278f", fontsize=6)
    ax_center.set(xlabel="Forecast lead from history end (min)", ylabel="COT at station pixel", title="Station-overhead cloud thickness")
    ax_peak.set(xlabel="Forecast lead from history end (min)", ylabel="Maximum COT in 5x5 neighbourhood", title="Nearest local thick-cloud signal")
    ax_track.set(xlabel="Peak column offset from station pixel", ylabel="Peak row offset from station pixel",
                 title="Trajectory of the local thickest-cloud location; labels = lead index 1..16",
                 xlim=(-WINDOW-.3, WINDOW+.3), ylim=(WINDOW+.3, -WINDOW-.3),
                 xticks=range(-WINDOW, WINDOW+1), yticks=range(-WINDOW, WINDOW+1))
    for ax in (ax_center, ax_peak):
        ax.set(xticks=[15, 60, 120, 180, 240])
        ax.grid(alpha=.25)
    ax_track.grid(alpha=.25)
    fig.suptitle("Observed-future-AGRI COT trajectory around Zhujia station (init 2025-08-02 06:00 BJT)", fontsize=13)
    output.mkdir(parents=True)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"observed_station_cot_summary.{ext}", dpi=300)
    plt.close(fig)
    payload = {
        "state": "COMPLETE_VALIDATION_OBSERVED_STATION_COT_SUMMARY",
        "test_used": False,
        "selection": "same predeclared temporal-median validation sequence; no AGRI/COT/GHI/error selection",
        "station": "zhujia", "station_pixel": list(CENTER), "neighborhood": "5x5 centered at station",
        "sequence": "hunan_202508012015", "init_time_bjt": "2025-08-02T06:00:00+08:00",
        "leads_minutes": LEADS.tolist(), "r_checkpoint_sha256": r_sha,
        "input_contract": "actual future AGRI plus target-time geometry through frozen R; validation diagnostic only",
        "metrics": {key: value.tolist() for key, value in result.items() if key != "local_maps"},
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
