"""Validation-only 16-lead station-neighbourhood COT trajectory comparison."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

STATION = "zhujia"
STATION_INDEX = 1
SEQUENCE = "hunan_202508012015"
CACHE_INDEX = 2082
BOUNDS = (65, 81, 142, 158)
CENTER = (8, 8)
WINDOW = 2
LEADS = np.arange(15, 241, 15)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["seq_id"] == SEQUENCE]
    if len(rows) != 1 or rows[0]["split"] != "val":
        raise RuntimeError("selected validation sequence is not unique")
    return rows[0]


def retrieve(root, raw):
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root.parent / "HunanSimVPDDP_20260907" / "scripts"))
    from train_cot_repaired import COTUNet
    cfg = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    norm = json.loads((root / "data/cot_repaired_pack_20260907_v1/norm.json").read_text())
    model = COTUNet().float().eval().requires_grad_(False)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu")["model"], strict=True)
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    x = (raw - mean) / std
    x[~np.isfinite(x)] = 0
    with torch.no_grad():
        result = model(torch.from_numpy(np.ascontiguousarray(x))).numpy()[:, 0]
    if not np.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("invalid frozen-R output")
    return result, cfg["checkpoint_sha256"]


def extract_observed(root, rels):
    from hunan_data import load_frame
    source = Path("/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed")
    r1, r2, c1, c2 = BOUNDS
    pieces = []
    for rel in rels[8:]:
        agri, _, geometry = load_frame(source, rel)
        pieces.append(np.concatenate((agri[:, r1:r2, c1:c2], geometry[:, r1:r2, c1:c2]), axis=0))
    result = np.asarray(pieces, dtype=np.float32)
    if result.shape != (16, 16, 16, 16) or not np.isfinite(result[:, 13:]).all():
        raise RuntimeError("observed target patch contract failed")
    return result


def local_metrics(log_maps):
    physical = np.expm1(log_maps)
    r, c = CENTER
    local = physical[:, r-WINDOW:r+WINDOW+1, c-WINDOW:c+WINDOW+1]
    flat = local.reshape(len(local), -1)
    peak_flat = flat.argmax(1)
    peak_r, peak_c = np.divmod(peak_flat, local.shape[-1])
    return {
        "center_cot": physical[:, r, c],
        "peak_cot": flat.max(1),
        "peak_row": peak_r + r - WINDOW,
        "peak_col": peak_c + c - WINDOW,
        "local_maps": local,
    }


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite")
    sys.path.insert(0, str(root.parent / "HunanSimVPDDP_20260907" / "scripts"))
    row = read_manifest(root.parent / "HunanSimVPFull_20260906/manifests_v1/hunan_8to16.csv")
    observed_raw = extract_observed(root, row["data_relpaths"].split("|"))
    observed_log, r_sha = retrieve(root, observed_raw)
    predicted_log = np.asarray(np.load(root / "data/forecast_cot_trajectory_trainval_20260918_v1/val_forecast_cot_log1p.npy", mmap_mode="r")[CACHE_INDEX, STATION_INDEX, :, 0], dtype=np.float32)
    if predicted_log.shape != (16, 16, 16):
        raise RuntimeError("forecast COT trajectory contract failed")
    actual = local_metrics(observed_log)
    forecast = local_metrics(predicted_log)
    cot_limit = float(np.nanmax((actual["local_maps"], forecast["local_maps"])))

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16.2, 7.5), layout="constrained")
    grids = fig.add_gridspec(5, 16, height_ratios=(1, 1, .17, 1.2, 1.2))
    cot_image = None
    for frame in range(16):
        for row_index, (name, result) in enumerate((("Observed", actual), ("SimVP", forecast))):
            ax = fig.add_subplot(grids[row_index, frame])
            cot_image = ax.imshow(result["local_maps"][frame], cmap="magma", vmin=0, vmax=cot_limit, interpolation="nearest")
            ax.scatter(WINDOW, WINDOW, marker="*", s=32, facecolor="white", edgecolor="black", linewidth=.6, zorder=3)
            ax.scatter(result["peak_col"][frame] - CENTER[1] + WINDOW, result["peak_row"][frame] - CENTER[0] + WINDOW, marker="o", s=23, facecolor="none", edgecolor="#31a354", linewidth=1.0, zorder=4)
            ax.set(xticks=[], yticks=[])
            if row_index == 0:
                ax.set_title(f"+{LEADS[frame]}", fontsize=7)
            if frame == 0:
                ax.set_ylabel(name + "\n5x5 COT", fontsize=8)
    color_ax = fig.add_subplot(grids[2, :])
    color_ax.axis("off")
    fig.colorbar(cot_image, ax=color_ax, orientation="horizontal", fraction=.9, pad=.18, label="physical COT; star = station, green circle = local peak")

    ax_center = fig.add_subplot(grids[3, :8])
    ax_peak = fig.add_subplot(grids[3, 8:])
    ax_track = fig.add_subplot(grids[4, :])
    for result, label, color, marker in ((actual, "R(observed AGRI)", "#54278f", "o"), (forecast, "R(SimVP AGRI)", "#008c7a", "s")):
        ax_center.plot(LEADS, result["center_cot"], color=color, marker=marker, ms=3, label=label)
        ax_peak.plot(LEADS, result["peak_cot"], color=color, marker=marker, ms=3, label=label)
        ax_track.plot(result["peak_col"] - CENTER[1], result["peak_row"] - CENTER[0], color=color, marker=marker, ms=4, label=label)
        for index in range(16):
            ax_track.annotate(str(index + 1), (result["peak_col"][index] - CENTER[1], result["peak_row"][index] - CENTER[0]), xytext=(2, 2), textcoords="offset points", color=color, fontsize=6)
    ax_center.set(xlabel="Forecast lead from history end (min)", ylabel="COT at station pixel", title="Station-overhead cloud thickness")
    ax_peak.set(xlabel="Forecast lead from history end (min)", ylabel="Maximum COT in 5x5 neighbourhood", title="Nearest local thick-cloud signal")
    ax_track.set(xlabel="Peak column offset from station pixel", ylabel="Peak row offset from station pixel", title="Trajectory of the local thickest-cloud location; labels = lead index 1..16")
    ax_track.set(xlim=(-WINDOW-.3, WINDOW+.3), ylim=(WINDOW+.3, -WINDOW-.3), xticks=range(-WINDOW, WINDOW+1), yticks=range(-WINDOW, WINDOW+1))
    for ax in (ax_center, ax_peak):
        ax.set(xticks=[15, 60, 120, 180, 240])
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=7)
    ax_track.grid(alpha=.25)
    ax_track.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Validation 16-lead COT trajectory around Zhujia station (init 2025-08-02 06:00 BJT)", fontsize=13)
    output.mkdir(parents=True)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"station_neighborhood_cot_trajectory.{ext}", dpi=300)
    plt.close(fig)

    payload = {"state": "COMPLETE_VALIDATION_16_LEAD_STATION_NEIGHBORHOOD_COT_DIAGNOSTIC", "test_used": False,
        "selection": "same predeclared temporal-median validation sequence as the preceding physical trajectory visual; no COT/GHI/error selection",
        "station": STATION, "station_pixel": list(CENTER), "neighborhood": "5x5 centered at station", "sequence": SEQUENCE, "cache_index": CACHE_INDEX,
        "init_time_bjt": "2025-08-02T06:00:00+08:00", "leads_minutes": LEADS.tolist(), "r_checkpoint_sha256": r_sha,
        "observed_contract": "each actual future AGRI frame plus target-time geometry runs through frozen R; validation diagnosis only",
        "forecast_contract": "cached frozen R(SimVP forecast AGRI), same target leads and station patch",
        "metrics": {"observed": {k: v.tolist() for k, v in actual.items() if k != "local_maps"}, "forecast": {k: v.tolist() for k, v in forecast.items() if k != "local_maps"}},
        "output_files": {p.name: sha(p) for p in output.iterdir() if p.suffix in (".png", ".pdf")}}
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
