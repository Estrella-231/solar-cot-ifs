"""Infer only the official CPP tiles needed for the two station 3x3 patches.

Uses the portable FY4B 105E COT weights, official input loaders, normalization,
tile layout, and overlap-fusion routine. It never reads validation/test payloads;
the requested timestamp must be explicitly present in the train candidate table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--candidate-manifest", type=Path, required=True)
    p.add_argument("--run-time-utc", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run_time = datetime.fromisoformat(a.run_time_utc)
    if run_time.tzinfo is not None:
        run_time = run_time.astimezone().replace(tzinfo=None)
    stamp = run_time.strftime("%Y%m%d%H%M")

    rows = []
    with a.candidate_manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["target_time_utc"] == run_time.strftime("%Y-%m-%d %H:%M:%S"):
                rows.append(row)
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError(f"time is not exclusively in the frozen train candidates: {stamp}")
    frames = {Path(row["agri_frame_relpath"]).stem for row in rows}
    if frames != {stamp}:
        raise ValueError(f"candidate UTC time and AGRI frame path disagree: {frames}, {stamp}")
    stations = {}
    for row in rows:
        stations[row["station"]] = tuple(map(int, row["true_hdf_rc"].split(",")))
    expected_stations = {"sili", "zhujia"}
    if set(stations) != expected_stations:
        raise ValueError(f"expected both stations, got {sorted(stations)}")

    os.environ["GLOBAL_CPP_RESOURCE_ROOT"] = str(a.package_root)
    os.environ["GLOBAL_CPP_RESULT_ROOT"] = str(a.output.parent / "isolated_empty_result_root")
    os.environ["GLOBAL_CPP_DEVICE"] = "cpu"
    os.environ["GLOBAL_CPP_TORCH_THREADS"] = "1"
    os.environ["GLOBAL_CPP_DATALOADER_WORKERS"] = "0"
    code_dir = a.package_root / "Code"
    model_dir = a.package_root / "Model" / "SmaATUnet"
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(model_dir))

    import torch
    from Data_Preprocess_FD import Cut, Postprocessing
    from Get_ERA_LCCS import Get_ERA_LCCS
    from Retrieval_FY4B_105E import Load_BT, Spec
    from Retrieval_CPP import Cloud_Vmax_COT, Cloud_Vmin_COT
    from SmaAtUNet import SmaAt_UNet

    torch.set_num_threads(1)
    bt = Load_BT(run_time)
    era = Get_ERA_LCCS(
        run_time, Spec.Region.Border_N, Spec.Region.Border_S,
        Spec.Region.Border_E, Spec.Region.Border_W,
    )
    if Spec.ERA_Transform is not None:
        era = Spec.ERA_Transform(era)
    if bt.shape != (6, 4051, 4051) or era.shape != (10, 4051, 4051):
        raise ValueError(f"unexpected input shapes: BT={bt.shape}, ERA={era.shape}")
    combined = np.concatenate((bt, era), axis=0).astype(np.float32, copy=False)
    tile_source = Cut(combined)

    needed = set()
    for row, col in stations.values():
        # Official tiles are 128x128 with stride 64. These are the two
        # overlapping tiles contributing to each station pixel in each axis.
        r0, c0 = row // 64, col // 64
        needed.update((r, c) for r in (r0 - 1, r0) for c in (c0 - 1, c0))
    indices = sorted(r * 63 + c for r, c in needed)

    model_path = Path(Spec.Model_Paths[3])
    model = SmaAt_UNet(n_channels=16, n_classes=1).to(torch.device("cpu"), dtype=torch.float32)
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"COT checkpoint lacks model_state_dict: {model_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint

    upper, lower = Spec.XMAXMIN[0], Spec.XMAXMIN[1]
    scale = upper - lower
    if np.any(scale == 0):
        raise ValueError("input normalization has a zero-width channel")
    output_tiles = np.zeros((63 * 63, 128, 128), dtype=np.float32)
    with torch.inference_mode():
        for index in indices:
            tile = np.asarray(tile_source[index], dtype=np.float32).copy()
            np.nan_to_num(tile, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            tile = (tile - lower[:, None, None]) / scale[:, None, None]
            pred = model(torch.from_numpy(tile[None]).to(dtype=torch.float32)).cpu().numpy()[0, 0]
            output_tiles[index] = np.clip(
                pred * (Cloud_Vmax_COT - Cloud_Vmin_COT) + Cloud_Vmin_COT,
                Cloud_Vmin_COT, Cloud_Vmax_COT,
            )

    # The omitted tiles are irrelevant to these small windows; the official
    # overlap-fusion routine only combines spatial neighbors around each crop.
    cot_map = Postprocessing(output_tiles)
    valid_obs = np.all(np.isfinite(bt[:5]), axis=0) & np.isfinite(bt[5])
    patches = {}
    for station, (row, col) in stations.items():
        patch = cot_map[row - 1:row + 2, col - 1:col + 2].copy()
        valid = valid_obs[row - 1:row + 2, col - 1:col + 2].copy()
        patch[~valid] = np.nan
        patches[f"{station}_cot_3x3"] = patch
        patches[f"{station}_input_valid_3x3"] = valid
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **patches)
    report = {
        "state": "STATION_CPP_COT_TILE_INFERENCE_COMPLETE",
        "split": "train",
        "run_time_utc": run_time.isoformat(sep=" "),
        "agri_manifest_frame_time_contract": "UTC; BJT is UTC+8",
        "stations_hdf_rc": {k: list(v) for k, v in stations.items()},
        "tile_indices_inferred": indices,
        "tile_count": len(indices),
        "cot_checkpoint": str(model_path),
        "cot_checkpoint_sha256": sha256(model_path),
        "input_shape": list(combined.shape),
        "official_overlap_fusion": True,
        "observation_invalid_pixels_retained_as_nan": True,
        "output_npz": str(a.output),
        "test_payload_opened": False,
        "patches": {
            key: {
                "shape": list(value.shape),
                "finite_count": int(np.isfinite(value).sum()),
                "min": float(np.nanmin(value)) if np.isfinite(value).any() else None,
                "max": float(np.nanmax(value)) if np.isfinite(value).any() else None,
                "mean": float(np.nanmean(value)) if np.isfinite(value).any() else None,
            }
            for key, value in patches.items() if key.endswith("_cot_3x3")
        },
    }
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
