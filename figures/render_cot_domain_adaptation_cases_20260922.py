"""Render a preselected validation trajectory before/after forecast-domain COT adaptation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RGB = (1, 2, 0)  # R=C02, G=C03, B=C01


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite a figure directory")
    prior = json.loads((root / "figures/cpp_vs_forecast_cot_trajectory_20260922/manifest.json").read_text())
    if prior["test_used"] is not False or prior["selection"].find("temporal median") < 0:
        raise RuntimeError("prior case selection changed")
    sequence, station = int(prior["sequence"]), int(prior["station_index"])
    leads = [lead // 15 - 1 for lead in prior["leads_minutes"]]
    refs = prior["cpp_pack_rows"]
    pack = root / "data/forecast_domain_cot_full16_20260922/val"
    keys = np.load(pack / "keys.npy", mmap_mode="r")
    indices = []
    for lead, ref in zip(leads, refs):
        positions = np.flatnonzero((keys[:, 0] == sequence) & (keys[:, 1] == station) &
                                   (keys[:, 2] == lead) & (keys[:, 3] == ref))
        if len(positions) != 1:
            raise RuntimeError(f"preselected row missing/duplicated: lead {lead}")
        indices.append(int(positions[0]))
    original = root / "data/cot_repaired_pack_20260907_v1"
    actual = np.asarray(np.load(original / "x_raw.npy", mmap_mode="r")[refs, :13], np.float32)
    cpp = np.asarray(np.load(original / "target.npy", mmap_mode="r")[refs, 0], np.float64) * 100
    mask = np.asarray(np.load(original / "mask.npy", mmap_mode="r")[refs, 0], bool)
    forecast = np.asarray(np.load(pack / "forecast_x.npy", mmap_mode="r")[indices, :13], np.float32)
    run = root / "runs/forecast_domain_cot_full16_seed42_20260922"
    predictions = {}
    for arm in ("real_train_control", "forecast_train_adapt"):
        with np.load(run / arm / "validation_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["keys"][indices], keys[indices]):
                raise RuntimeError("saved forecast case keys do not match pack")
            if not np.array_equal(saved["reference_cot"][indices, 0], cpp.astype(np.float32)):
                raise RuntimeError("saved CPP reference differs")
            if not np.array_equal(saved["mask"][indices, 0], mask):
                raise RuntimeError("saved CPP mask differs")
            predictions[arm] = np.expm1(saved["pred_log1p_cot"][indices, 0].astype(np.float64))
    native = np.moveaxis(np.concatenate((actual, forecast))[:, RGB], 1, -1).astype(np.float64)
    low = np.nanquantile(native, .01, axis=(0, 1, 2), keepdims=True)
    high = np.nanquantile(native, .99, axis=(0, 1, 2), keepdims=True)
    rgb = np.clip((native - low) / np.maximum(high - low, 1e-12), 0, 1) ** (1 / 2.2)
    rgb[~np.isfinite(rgb).all(axis=-1)] = .5
    old = np.where(mask, predictions["real_train_control"], np.nan)
    new = np.where(mask, predictions["forecast_train_adapt"], np.nan)
    cpp = np.where(mask, cpp, np.nan)
    delta = new - old
    cot_max = float(np.nanmax([cpp, old, new]))
    delta_max = float(np.nanmax(np.abs(delta)))
    fig, axes = plt.subplots(4, 6, figsize=(14.2, 8.7), layout="constrained")
    titles = ("Observed future AGRI\nfalse colour", "SimVP forecast AGRI\nfalse colour",
              "CPP COT reference", "Original R\non forecast AGRI",
              "Adapted R\non forecast AGRI", "Adapted − original\nforecast COT")
    for i in range(4):
        fields = (rgb[i], rgb[i + 4], cpp[i], old[i], new[i], delta[i])
        for j, (ax, field) in enumerate(zip(axes[i], fields)):
            if j < 2:
                ax.imshow(field, interpolation="nearest")
            elif j == 5:
                change_image = ax.imshow(field, cmap="RdBu_r", vmin=-delta_max, vmax=delta_max,
                                         interpolation="nearest")
            else:
                cot_image = ax.imshow(field, cmap="magma", vmin=0, vmax=cot_max, interpolation="nearest")
            ax.scatter(8, 8, marker="*", s=65, facecolor="white", edgecolor="black", linewidth=.8)
            ax.set(xticks=[], yticks=[])
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
        axes[i, 0].set_ylabel(f"+{prior['leads_minutes'][i]} min", fontsize=9)
    fig.colorbar(cot_image, ax=axes[:, 2:5], shrink=.8, label="COT")
    fig.colorbar(change_image, ax=axes[:, 5], shrink=.8, label="COT change")
    fig.suptitle(f"Forecast-domain COT adaptation: {prior['station']}, init {prior['init_time_bjt']}\n"
                 "Time-selected validation case; RGB stretch shared across 8 AGRI maps; "
                 "star marks station [8,8]; CPP is retrieval reference", fontsize=11)
    output.mkdir(parents=True)
    fig.savefig(output / "cot_domain_adaptation_case.png", dpi=220)
    plt.close(fig)
    metrics = {}
    for arm, array in predictions.items():
        err = (array - cpp)[mask]
        metrics[arm] = {"case_valid_pixels": int(len(err)), "case_rmse": float(np.sqrt(np.mean(err ** 2)))}
    report = {"state": "COMPLETE_COT_DOMAIN_ADAPTATION_VALIDATION_CASE", "test_used": False,
              "selection": prior["selection"], "station": prior["station"], "sequence": sequence,
              "leads_minutes": prior["leads_minutes"], "cpp_pack_rows": refs,
              "validation_pack_positions": indices, "case_metrics": metrics,
              "figure_sha256": digest(output / "cot_domain_adaptation_case.png")}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
