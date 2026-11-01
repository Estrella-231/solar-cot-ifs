"""Render a validation trajectory of R(observed AGRI) versus R(SimVP AGRI)."""
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

LEADS = (3, 7, 11, 15)  # +60, +120, +180, +240 min
RGB = (1, 2, 0)  # R=C02, G=C03, B=C01
STATIONS = ("sili", "zhujia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select(root):
    """Select temporal median before inspecting imagery, COT, GHI, or errors."""
    base = root / "data/head_pack_trainval_20260908_v1"
    rows = read_csv(base / "rows.csv")
    seq = np.load(base / "sequence.npy", mmap_mode="r")
    sta = np.load(base / "station.npy", mmap_mode="r")
    lead = np.load(base / "lead.npy", mmap_mode="r")
    split = np.load(base / "split.npy", mmap_mode="r")
    cpp_rows = read_csv(root / "data/cot_repaired_pack_20260907_v1/rows.csv")
    cpp = {(x["station_id"], x["timestamp_utc"].replace("Z", "+00:00")): int(x["index"])
           for x in cpp_rows if x["split"] == "validation" and x["status"] == "ok"}
    groups = {}
    for i, row in enumerate(rows):
        if int(split[i]) != 1 or row["day_mask_sza_lt_90"] != "True":
            continue
        groups.setdefault((int(sta[i]), int(seq[i])), {})[int(lead[i])] = (i, row)
    candidates = []
    for key, values in groups.items():
        if not all(item in values for item in LEADS):
            continue
        selected = [values[item] for item in LEADS]
        refs = [cpp.get((item[1]["station"], item[1]["target_time_utc"].replace("Z", "+00:00"))) for item in selected]
        if all(item is not None for item in refs):
            candidates.append((key, selected, refs))
    if not candidates:
        raise RuntimeError("no daytime validation trajectory has all declared AGRI/COT keys")
    candidates.sort(key=lambda x: (x[1][0][1]["init_time_utc"], x[0]))
    return candidates[len(candidates) // 2], len(candidates)


def forecast_agri(root, sequence, station):
    cache = root / "data/frozen_forecast_trainval_20260907_v1/val"
    specs = json.loads((root / "data/head_pack_trainval_20260908_v1/source_shards.json").read_text())
    for spec in (x for x in specs if x["split"] == "val"):
        with np.load(cache / spec["name"], allow_pickle=False) as z:
            hit = np.flatnonzero(z["indices"] == sequence)
            if len(hit) == 1:
                return z["predicted_agri_physical"][int(hit[0]), station, list(LEADS)].copy()
    raise RuntimeError("selected sequence absent from frozen validation cache")


def retrieve(root, raw):
    sys.path.insert(0, str(root / "scripts"))
    from train_cot_repaired import COTUNet
    cfg = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    norm = json.loads((root / "data/cot_repaired_pack_20260907_v1/norm.json").read_text())
    model = COTUNet().float().eval().requires_grad_(False)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu")["model"], strict=True)
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    x = (raw.astype(np.float32) - mean) / std
    x[~np.isfinite(x)] = 0
    with torch.no_grad():
        result = model(torch.from_numpy(np.ascontiguousarray(x))).numpy()[:, 0]
    if not np.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("frozen R returned invalid log1p(COT)")
    return result, cfg["checkpoint_sha256"]


def make_rgb(observed, predicted):
    both = np.moveaxis(np.concatenate((observed, predicted))[:, RGB], 1, -1).astype(np.float64)
    low = np.nanquantile(both, .01, axis=(0, 1, 2), keepdims=True)
    high = np.nanquantile(both, .99, axis=(0, 1, 2), keepdims=True)
    rgb = np.clip((both - low) / np.maximum(high - low, 1e-12), 0, 1) ** (1 / 2.2)
    rgb[~np.isfinite(rgb).all(axis=-1)] = .5
    return rgb[:len(observed)], rgb[len(observed):], low.ravel(), high.ravel()


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite a trajectory figure directory")
    ((station, sequence), selected, cpp_rows), candidate_count = select(root)
    meta = [item[1] for item in selected]
    observed_raw = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/x_raw.npy", mmap_mode="r")[cpp_rows], dtype=np.float32)
    mask = np.asarray(np.load(root / "data/cot_repaired_pack_20260907_v1/mask.npy", mmap_mode="r")[cpp_rows, 0], dtype=bool)
    observed_log, r_sha = retrieve(root, observed_raw)
    pred_agri = forecast_agri(root, sequence, station)
    pred_log = np.asarray(np.load(root / "data/forecast_cot_trajectory_trainval_20260918_v1/val_forecast_cot_log1p.npy", mmap_mode="r")[sequence, station, list(LEADS), 0], dtype=np.float32)
    observed_rgb, predicted_rgb, rgb_low, rgb_high = make_rgb(observed_raw[:, :13], pred_agri)
    difference = np.expm1(pred_log) - np.expm1(observed_log)
    for array in (observed_log, pred_log, difference):
        array[~mask] = np.nan
    cot_limit = float(np.nanmax((observed_log, pred_log)))
    diff_limit = float(np.nanmax(np.abs(difference)))

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 5, figsize=(11.1, 8.6), layout="constrained")
    titles = ("Observed future AGRI\nfalse-colour", "SimVP forecast AGRI\nfalse-colour", "R(observed AGRI)\nlog1p COT", "R(SimVP AGRI)\nlog1p COT", "Forecast - observed\nphysical COT")
    cot_image = diff_image = None
    for frame, lead in enumerate(LEADS):
        fields = (observed_rgb[frame], predicted_rgb[frame], observed_log[frame], pred_log[frame], difference[frame])
        for col, (ax, field) in enumerate(zip(axes[frame], fields)):
            if col < 2:
                ax.imshow(field, interpolation="nearest")
            elif col == 4:
                diff_image = ax.imshow(field, cmap="RdBu_r", vmin=-diff_limit, vmax=diff_limit, interpolation="nearest")
            else:
                cot_image = ax.imshow(field, cmap="magma", vmin=0, vmax=cot_limit, interpolation="nearest")
            ax.scatter(8, 8, marker="*", s=65, facecolor="white", edgecolor="black", linewidth=.8, zorder=3)
            ax.set(xticks=[], yticks=[])
            if frame == 0:
                ax.set_title(titles[col], fontsize=9)
        axes[frame, 0].set_ylabel(f"+{(lead + 1) * 15} min", fontsize=9)
    fig.colorbar(cot_image, ax=axes[:, (2, 3)], shrink=.85, label="log1p(COT)")
    fig.colorbar(diff_image, ax=axes[:, 4], shrink=.85, label="COT difference")
    fig.suptitle(f"Validation COT trajectory: {STATIONS[station]}, init {meta[0]['init_time_bjt']}", fontsize=12)
    fig.text(.5, .006, "RGB: R=C02, G=C03, B=C01; shared 1st-99th percentile stretch. Stars mark patch [8,8].", ha="center", fontsize=7.5, color="#444444")
    output.mkdir(parents=True)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"real_vs_forecast_cot_trajectory.{ext}", dpi=300)
    plt.close(fig)
    manifest = {"state": "COMPLETE_VALIDATION_REAL_VS_FORECAST_COT_TRAJECTORY", "test_used": False,
        "selection": "temporal median among daytime validation station-sequences with +60,+120,+180,+240 min observed-AGRI and CPP-mask availability; selection ignores AGRI/COT/GHI/error",
        "candidate_count": candidate_count, "station": STATIONS[station], "station_index": station, "sequence": sequence,
        "base_pack_rows": [int(x[0]) for x in selected], "cpp_pack_rows": cpp_rows, "leads_minutes": [int((x + 1) * 15) for x in LEADS],
        "init_time_bjt": meta[0]["init_time_bjt"], "target_time_bjt": [x["target_time_bjt"] for x in meta], "r_checkpoint_sha256": r_sha,
        "rgb_contract": {"red": "C02", "green": "C03", "blue": "C01", "gamma": 2.2, "shared_percentile_stretch": [1, 99], "low": rgb_low.tolist(), "high": rgb_high.tolist()},
        "cot_contract": "frozen R(observed target AGRI) versus frozen R(SimVP forecast AGRI); future observed AGRI is validation-only diagnosis, never GHI input",
        "difference_contract": "physical COT R(SimVP AGRI) minus R(observed AGRI); shared symmetric range, no clipping",
        "output_files": {p.name: sha(p) for p in output.iterdir() if p.suffix in (".png", ".pdf")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
