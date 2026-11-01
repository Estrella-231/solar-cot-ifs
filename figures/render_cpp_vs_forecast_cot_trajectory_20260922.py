"""Render a deterministic real-CPP versus forecast-COT validation trajectory.

Four hourly spaced future targets share one Hunan validation initialization.
The COT prediction is frozen R(SimVP forecast AGRI); the physical target is
the saved CPP COT reference. CPP is explicitly a retrieval reference rather
than independent physical truth.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEAD_INDEXES = (3, 7, 11, 15)  # canonical +60, +120, +180, +240 min
RGB_CHANNELS = (1, 2, 0)  # R=C02, G=C03, B=C01


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_utc(value):
    """Rows use both ISO spellings for the same UTC instant: Z and +00:00."""
    return value.replace("Z", "+00:00")


def select_trajectory(root):
    base = root / "data/head_pack_trainval_20260908_v1"
    rows = read_csv(base / "rows.csv")
    sequence = np.load(base / "sequence.npy", mmap_mode="r")
    station = np.load(base / "station.npy", mmap_mode="r")
    lead = np.load(base / "lead.npy", mmap_mode="r")
    split = np.load(base / "split.npy", mmap_mode="r")
    cpp_rows = read_csv(root / "data/cot_repaired_pack_20260907_v1/rows.csv")
    # The COT retriever pack calls this split "validation" while the station
    # head pack uses integer split=1. They refer to the same held-out period.
    cpp_index = {(item["station_id"], canonical_utc(item["timestamp_utc"])): int(item["index"])
                 for item in cpp_rows if item["split"] == "validation" and item["status"] == "ok"}
    groups = {}
    for index, item in enumerate(rows):
        if int(split[index]) != 1 or item["day_mask_sza_lt_90"] != "True":
            continue
        key = (int(station[index]), int(sequence[index]))
        groups.setdefault(key, {})[int(lead[index])] = (index, item)
    candidates = []
    for key, by_lead in groups.items():
        if not all(item in by_lead for item in LEAD_INDEXES):
            continue
        selected = [by_lead[item] for item in LEAD_INDEXES]
        cpp = [cpp_index.get((entry[1]["station"], canonical_utc(entry[1]["target_time_utc"]))) for entry in selected]
        if any(item is None for item in cpp):
            continue
        candidates.append((key, selected, cpp))
    if not candidates:
        raise RuntimeError("no daytime validation trajectory has four hourly CPP references")
    # Temporal median is fixed before examining AGRI, COT, GHI, or model error.
    candidates.sort(key=lambda item: (item[1][0][1]["init_time_utc"], item[0]))
    return candidates[len(candidates) // 2], len(candidates)


def load_forecast_agri(root, sequence, station, leads):
    base = root / "data/head_pack_trainval_20260908_v1"
    specs = json.loads((base / "source_shards.json").read_text())
    cache = root / "data/frozen_forecast_trainval_20260907_v1/val"
    selected = None
    for spec in (item for item in specs if item["split"] == "val"):
        with np.load(cache / spec["name"], allow_pickle=False) as shard:
            locations = np.flatnonzero(shard["indices"] == sequence)
            if len(locations):
                if len(locations) != 1:
                    raise RuntimeError("duplicate sequence in frozen cache")
                selected = shard["predicted_agri_physical"][int(locations[0]), station, list(leads)].copy()
                break
    if selected is None:
        raise RuntimeError("selected sequence absent from frozen validation cache")
    return selected


def rgb_stack(actual, forecast):
    native = np.moveaxis(np.concatenate((actual, forecast), axis=0)[:, RGB_CHANNELS], 1, -1).astype(np.float64)
    low = np.nanquantile(native, .01, axis=(0, 1, 2), keepdims=True)
    high = np.nanquantile(native, .99, axis=(0, 1, 2), keepdims=True)
    rgb = np.clip((native - low) / np.maximum(high - low, 1e-12), 0, 1) ** (1 / 2.2)
    rgb[~np.isfinite(rgb).all(axis=-1)] = .5
    return rgb[:len(actual)], rgb[len(actual):], low.reshape(-1), high.reshape(-1)


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite a trajectory figure directory")
    ((station, sequence), selected, cpp_indexes), candidate_count = select_trajectory(root)
    row_indexes = np.array([item[0] for item in selected], dtype=np.int64)
    row_metadata = [item[1] for item in selected]
    actual = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/x_raw.npy", mmap_mode="r")[cpp_indexes, :13])
    cpp = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/target.npy", mmap_mode="r")[cpp_indexes, 0], dtype=np.float64) * 100.0
    mask = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/mask.npy", mmap_mode="r")[cpp_indexes, 0], dtype=bool)
    forecast = load_forecast_agri(root, sequence, station, LEAD_INDEXES)
    forecast_log = np.load(root / "data/forecast_cot_trajectory_trainval_20260918_v1/val_forecast_cot_log1p.npy", mmap_mode="r")[sequence, station, list(LEAD_INDEXES), 0].astype(np.float64)
    forecast_cot = np.expm1(forecast_log)
    actual_rgb, forecast_rgb, rgb_low, rgb_high = rgb_stack(actual, forecast)
    cpp_masked = cpp.copy(); cpp_masked[~mask] = np.nan
    forecast_masked = forecast_cot.copy(); forecast_masked[~mask] = np.nan
    diff = forecast_cot - cpp; diff[~mask] = np.nan
    cot_limit = float(max(np.nanmax(cpp_masked), np.nanmax(forecast_masked)))
    diff_limit = float(np.nanmax(np.abs(diff)))

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(4, 5, figsize=(11.1, 8.6), layout="constrained")
    titles = ("Observed future AGRI\nfalse-colour", "SimVP forecast AGRI\nfalse-colour",
              "CPP COT reference", "Forecast COT\nR(SimVP AGRI)", "Forecast − CPP COT")
    cot_image = diff_image = None
    for frame, lead in enumerate(LEAD_INDEXES):
        fields = (actual_rgb[frame], forecast_rgb[frame], cpp_masked[frame], forecast_masked[frame], diff[frame])
        for column, (axis, field) in enumerate(zip(axes[frame], fields)):
            if column < 2:
                axis.imshow(field, interpolation="nearest")
            elif column == 4:
                diff_image = axis.imshow(field, cmap="RdBu_r", vmin=-diff_limit, vmax=diff_limit, interpolation="nearest")
            else:
                cot_image = axis.imshow(field, cmap="magma", vmin=0, vmax=cot_limit, interpolation="nearest")
            axis.scatter(8, 8, marker="*", s=65, facecolor="white", edgecolor="black", linewidth=.8, zorder=3)
            axis.set(xticks=[], yticks=[])
            if frame == 0:
                axis.set_title(titles[column], fontsize=9)
        axes[frame, 0].set_ylabel(f"+{(lead + 1) * 15} min", fontsize=9)
    figure.colorbar(cot_image, ax=axes[:, (2, 3)], shrink=.85, label="COT")
    figure.colorbar(diff_image, ax=axes[:, 4], shrink=.85, label="COT difference")
    station_name = ("Sili", "Zhujia")[station]
    figure.suptitle(f"Validation COT trajectory: {station_name}, init {row_metadata[0]['init_time_bjt']}", fontsize=12)
    figure.text(.5, .006,
                "RGB: R=C02, G=C03, B=C01; one 1st–99th percentile stretch shared by all eight AGRI maps. "
                "Stars mark patch [8,8]. CPP is a retrieval reference, not independent physical truth.",
                ha="center", fontsize=7.5, color="#444444")
    output.mkdir(parents=True)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"cpp_vs_forecast_cot_trajectory.{extension}", dpi=300)
    plt.close(figure)
    manifest = {
        "state": "COMPLETE_VALIDATION_CPP_VS_FORECAST_COT_TRAJECTORY_VISUAL", "test_used": False,
        "selection": "temporal median among daytime validation station-sequences with all predeclared +60,+120,+180,+240 min CPP-reference targets; no AGRI, COT, GHI, or model-error selection",
        "candidate_count": candidate_count, "station": station_name, "station_index": station, "sequence": sequence,
        "base_pack_rows": row_indexes.tolist(), "cpp_pack_rows": cpp_indexes,
        "leads_minutes": [int((lead + 1) * 15) for lead in LEAD_INDEXES],
        "init_time_bjt": row_metadata[0]["init_time_bjt"],
        "target_time_bjt": [item["target_time_bjt"] for item in row_metadata],
        "rgb_contract": {"red": "C02", "green": "C03", "blue": "C01", "gamma": 2.2,
                         "shared_percentile_stretch": [1, 99], "low": rgb_low.tolist(), "high": rgb_high.tolist()},
        "cot_contract": "physical CPP COT reference versus expm1(frozen R(SimVP forecast AGRI)); CPP is not independent physical truth",
        "difference_contract": "forecast COT minus CPP COT; symmetric display range spans the selected valid values without clipping",
        "inputs": {"base_rows_csv": sha(root / "data/head_pack_trainval_20260908_v1/rows.csv"),
                   "cpp_rows_csv": sha(root / "data/cot_repaired_pack_20260907_v1/rows.csv"),
                   "forecast_cot_sidecar": sha(root / "data/forecast_cot_trajectory_trainval_20260918_v1/val_forecast_cot_log1p.npy")},
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
