"""Render frozen COT retrieval fidelity on actual future AGRI only.

This is a validation-only diagnostic: it compares R(observed target AGRI)
with the aligned CPP COT reference on the same deterministic trajectory used
by the forecast-COT visual. It does not enter GHI training or inference.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from render_cpp_vs_forecast_cot_trajectory_20260922 import LEAD_INDEXES, rgb_stack, select_trajectory, sha


def retrieve(root, agri_and_geometry):
    sys.path.insert(0, str(root / "scripts"))
    from train_cot_repaired import COTUNet
    norm = json.loads((root / "data/cot_repaired_pack_20260907_v1/norm.json").read_text())
    config = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    checkpoint = torch.load(config["checkpoint"], map_location="cpu")
    model = COTUNet().float().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    normalized = (agri_and_geometry.astype(np.float32) - mean) / std
    normalized[~np.isfinite(normalized)] = 0
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.ascontiguousarray(normalized))).numpy()[:, 0]
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError("frozen COT retrieval returned invalid log1p COT")
    return prediction


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite a trajectory figure directory")
    config = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    ((station, sequence), selected, cpp_indexes), candidate_count = select_trajectory(root)
    row_metadata = [item[1] for item in selected]
    actual_full = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/x_raw.npy", mmap_mode="r")[cpp_indexes], dtype=np.float32)
    actual = actual_full[:, :13]
    cpp = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/target.npy", mmap_mode="r")[cpp_indexes, 0], dtype=np.float64) * 100.0
    mask = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/mask.npy", mmap_mode="r")[cpp_indexes, 0], dtype=bool)
    retrieved_log = retrieve(root, actual_full)
    retrieved = np.expm1(retrieved_log)
    actual_rgb, _, rgb_low, rgb_high = rgb_stack(actual, actual)
    cpp_masked = cpp.copy(); cpp_masked[~mask] = np.nan
    retrieved_masked = retrieved.copy(); retrieved_masked[~mask] = np.nan
    difference = retrieved - cpp; difference[~mask] = np.nan
    cot_limit = float(max(np.nanmax(cpp_masked), np.nanmax(retrieved_masked)))
    diff_limit = float(np.nanmax(np.abs(difference)))

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(4, 4, figsize=(9.0, 8.6), layout="constrained")
    titles = ("Observed future AGRI\nfalse-colour", "CPP COT reference",
              "Frozen R(observed AGRI)", "R(observed) − CPP COT")
    cot_image = diff_image = None
    frame_metrics = []
    for frame, lead in enumerate(LEAD_INDEXES):
        fields = (actual_rgb[frame], cpp_masked[frame], retrieved_masked[frame], difference[frame])
        valid = mask[frame]
        error = difference[frame][valid]
        frame_metrics.append({"lead_minutes": int((lead + 1) * 15), "valid_pixels": int(valid.sum()),
                              "cpp_mean": float(cpp[frame][valid].mean()), "cpp_max": float(cpp[frame][valid].max()),
                              "retrieved_mean": float(retrieved[frame][valid].mean()), "retrieved_max": float(retrieved[frame][valid].max()),
                              "cot_mae": float(np.abs(error).mean()), "cot_rmse": float(np.sqrt(np.mean(error ** 2))),
                              "cot_bias": float(error.mean())})
        for column, (axis, field) in enumerate(zip(axes[frame], fields)):
            if column == 0:
                axis.imshow(field, interpolation="nearest")
            elif column == 3:
                diff_image = axis.imshow(field, cmap="RdBu_r", vmin=-diff_limit, vmax=diff_limit, interpolation="nearest")
            else:
                cot_image = axis.imshow(field, cmap="magma", vmin=0, vmax=cot_limit, interpolation="nearest")
            axis.scatter(8, 8, marker="*", s=65, facecolor="white", edgecolor="black", linewidth=.8, zorder=3)
            axis.set(xticks=[], yticks=[])
            if frame == 0:
                axis.set_title(titles[column], fontsize=9)
        axes[frame, 0].set_ylabel(f"+{(lead + 1) * 15} min", fontsize=9)
    figure.colorbar(cot_image, ax=axes[:, (1, 2)], shrink=.85, label="COT")
    figure.colorbar(diff_image, ax=axes[:, 3], shrink=.85, label="COT difference")
    station_name = ("Sili", "Zhujia")[station]
    figure.suptitle(f"Frozen COT retriever on observed future AGRI: {station_name}, init {row_metadata[0]['init_time_bjt']}", fontsize=11)
    figure.text(.5, .006,
                "RGB: R=C02, G=C03, B=C01; shared 1st–99th percentile stretch. Stars mark patch [8,8]. "
                "CPP is a retrieval reference, not independent physical truth.",
                ha="center", fontsize=7.5, color="#444444")
    output.mkdir(parents=True)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"cpp_vs_real_agri_cot_trajectory.{extension}", dpi=300)
    plt.close(figure)
    manifest = {
        "state": "COMPLETE_VALIDATION_CPP_VS_REAL_AGRI_COT_TRAJECTORY_VISUAL", "test_used": False,
        "selection": "same temporal-median daytime validation trajectory as forecast-COT figure; no COT/GHI/error selection",
        "candidate_count": candidate_count, "station": station_name, "station_index": station, "sequence": sequence,
        "cpp_pack_rows": cpp_indexes, "leads_minutes": [item["lead_minutes"] for item in frame_metrics],
        "init_time_bjt": row_metadata[0]["init_time_bjt"], "target_time_bjt": [item["target_time_bjt"] for item in row_metadata],
        "frame_metrics": frame_metrics,
        "rgb_contract": {"red": "C02", "green": "C03", "blue": "C01", "gamma": 2.2,
                         "shared_percentile_stretch": [1, 99], "low": rgb_low.tolist(), "high": rgb_high.tolist()},
        "cot_contract": "physical CPP COT reference versus expm1(frozen R(observed target AGRI)); CPP is not independent physical truth",
        "difference_contract": "R(observed target AGRI) minus CPP COT; symmetric display range spans selected valid values without clipping",
        "inputs": {"r_checkpoint_sha256": config["checkpoint_sha256"], "cpp_rows_csv": sha(root / "data/cot_repaired_pack_20260907_v1/rows.csv")},
        "output_files": {path.name: sha(path) for path in output.iterdir() if path.suffix in (".png", ".pdf")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
