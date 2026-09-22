"""Show real validation AGRI inputs and original/continued COT retrievals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite figure directory")
    pack = root / "data/cot_repaired_pack_20260907_v1"
    run = root / "runs/cot_real_agri_highcloud_pilot_seed42_20260922"
    with (pack / "rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    val_indices = np.array([int(row["index"]) for row in rows if row["split"] == "validation"], np.int64)
    val_rows = [rows[int(index)] for index in val_indices]
    target = np.asarray(np.load(pack / "target.npy", mmap_mode="r")[val_indices, 0], np.float32) * np.float32(100)
    mask = np.asarray(np.load(pack / "mask.npy", mmap_mode="r")[val_indices, 0], bool)
    cases = []
    conditions = (("clear center", lambda x: x < .1),
                  ("medium center", lambda x: (x >= 5) & (x < 15)),
                  ("thick center", lambda x: x >= 30))
    for label, condition in conditions:
        valid = np.array([row["station_id"] == "sili" for row in val_rows]) & mask[:, 8, 8] & condition(target[:, 8, 8])
        choices = np.flatnonzero(valid)
        if not len(choices):
            raise RuntimeError(f"no validation case in {label}")
        # Dataset rows are UTC time sorted. The median case is chosen without
        # looking at RGB, model predictions, or errors.
        cases.append((label, int(choices[len(choices) // 2]), int(len(choices))))
    positions = [case[1] for case in cases]
    images = np.asarray(np.load(pack / "x_raw.npy", mmap_mode="r")[val_indices[positions], :13], np.float32)
    native = np.moveaxis(images[:, [1, 2, 0]], 1, -1).astype(np.float64)
    low = np.nanquantile(native, .01, axis=(0, 1, 2), keepdims=True)
    high = np.nanquantile(native, .99, axis=(0, 1, 2), keepdims=True)
    rgb = np.clip((native - low) / np.maximum(high - low, 1e-12), 0, 1) ** (1 / 2.2)
    rgb[~np.isfinite(rgb).all(axis=-1)] = .5
    predictions = {}
    sources = {"old": root / "runs/cot_repaired_seed42_v1/validation_predictions.npz",
               "continued": run / "log_control/validation_predictions.npz"}
    for name, path in sources.items():
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["row_indices"], val_indices):
                raise RuntimeError(f"{name} saved validation ordering differs")
            if not np.array_equal(saved["reference_cot"][positions, 0], target[positions]):
                raise RuntimeError(f"{name} CPP reference differs")
            if not np.array_equal(saved["mask"][positions, 0], mask[positions]):
                raise RuntimeError(f"{name} CPP mask differs")
            predictions[name] = np.expm1(saved["pred_log1p_cot"][positions, 0].astype(np.float64))
    cpp = np.where(mask[positions], target[positions], np.nan)
    old = np.where(mask[positions], predictions["old"], np.nan)
    new = np.where(mask[positions], predictions["continued"], np.nan)
    old_error = old - cpp
    new_error = new - cpp
    cot_max = float(np.nanmax(np.stack((cpp, old, new))))
    error_max = float(np.nanmax(np.abs(np.stack((old_error, new_error)))))
    fig, axes = plt.subplots(3, 6, figsize=(13, 6.5), layout="constrained")
    headings = ("Real AGRI false colour", "CPP COT reference", "Frozen R COT",
                "Continued R COT", "Frozen R − CPP", "Continued R − CPP")
    for i in range(3):
        fields = (rgb[i], cpp[i], old[i], new[i], old_error[i], new_error[i])
        for j, (axis, field) in enumerate(zip(axes[i], fields)):
            if j == 0:
                axis.imshow(field, interpolation="nearest")
            elif j >= 4:
                error_image = axis.imshow(field, cmap="RdBu_r", vmin=-error_max, vmax=error_max,
                                          interpolation="nearest")
            else:
                cot_image = axis.imshow(field, cmap="magma", vmin=0, vmax=cot_max, interpolation="nearest")
            axis.scatter(8, 8, marker="*", s=50, facecolor="white", edgecolor="black", linewidth=.7)
            axis.set(xticks=[], yticks=[])
            if i == 0:
                axis.set_title(headings[j], fontsize=8)
        axes[i, 0].set_ylabel(cases[i][0], fontsize=8)
    fig.colorbar(cot_image, ax=axes[:, 1:4], shrink=.83, label="COT")
    fig.colorbar(error_image, ax=axes[:, 4:6], shrink=.83, label="COT error")
    fig.suptitle("Real AGRI → COT: time-median validation examples per station-center CPP bin\n"
                 "Sili; RGB R=C02 G=C03 B=C01 (shared stretch); star=station [8,8]; CPP is retrieval reference",
                 fontsize=10)
    output.mkdir(parents=True)
    fig.savefig(output / "real_agri_cot_retrieval_cases.png", dpi=220)
    plt.close(fig)
    report = {"state": "COMPLETE_REAL_AGRI_COT_RETRIEVAL_CASES", "test_used": False,
              "selection": "Sili validation only; for each predeclared station-center CPP COT bin, select temporal median without examining predictions or errors",
              "cases": [{"label": label, "candidates": count, "validation_position": position,
                         "pack_row": int(val_indices[position]), "timestamp_bjt": val_rows[position]["timestamp_bjt"],
                         "center_cpp_cot": float(target[position, 8, 8])}
                        for label, position, count in cases],
              "figure_sha256": sha(output / "real_agri_cot_retrieval_cases.png")}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
