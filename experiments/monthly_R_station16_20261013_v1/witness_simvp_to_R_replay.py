#!/usr/bin/env python3
"""Single-window witness for the monthly SimVP -> frozen-R forecast chain.

This is an engineering/replay witness only. It reads one deterministic
validation window, never a test row or payload, and checks that target-time
observed AGRI inverted through train-only normalization reproduces the saved
monthly-R COT bank before evaluating predicted AGRI through the same R.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EXPECTED = {
    "candidate_sha256": "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807",
    "simvp_checkpoint_sha256": "f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4",
    "simvp_norm_sha256": "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d",
    "r_checkpoint_sha256": "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001",
    "r_source_sha256": "7a2e5cb441cdb5132a537cd03580017801595bdea839719d1773398d267210ce",
    "r_bank_candidate_sha256": "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807",
}
LEADS = list(range(15, 241, 15))
STATIONS = ("sili", "zhujia")
CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--simvp-root", type=Path, required=True)
    p.add_argument("--r-root", type=Path, required=True)
    p.add_argument("--agri-root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-index", type=int, default=0)
    a = p.parse_args()

    simvp_root, r_root = a.simvp_root, a.r_root
    source = simvp_root / "code"
    manifests = simvp_root / "manifests_v1"
    checkpoint = simvp_root / "runs/simvp_seed42_v1/training/best.pt"
    norm_path = manifests / "hunan_train_norm.json"
    candidate = r_root / "input/target_candidates_train_val.csv"
    r_checkpoint = r_root / "results/seed42_v1/best.pt"
    r_norm_path = r_root / "results/seed42_v1/normalization.json"
    r_source = r_root / "scripts/train_station16_R.py"
    bank = r_root / "results/new_monthly_R_seed42_cot_bank_v1"
    keys_path = bank / "frame_keys.csv"
    cot_bank_path = bank / "cot_log1p.npy"

    for name, path, key in (
        ("candidate", candidate, "candidate_sha256"),
        ("SimVP checkpoint", checkpoint, "simvp_checkpoint_sha256"),
        ("SimVP norm", norm_path, "simvp_norm_sha256"),
        ("R checkpoint", r_checkpoint, "r_checkpoint_sha256"),
        ("R source", r_source, "r_source_sha256"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        actual = sha256(path)
        if actual != EXPECTED[key]:
            raise ValueError(f"{name} hash mismatch: {actual}")
        print(f"SOURCE_SHA_PASS {name} {actual}", flush=True)
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SimVP replay witness")
    if not keys_path.is_file() or not cot_bank_path.is_file():
        raise FileNotFoundError("frozen R-COT bank or frame keys are missing")

    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    if norm.get("validation_test_used") is not False:
        raise ValueError("train-only normalization does not certify test exclusion")
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    if mean.shape != (13,) or std.shape != (13,) or not np.isfinite(mean).all() or (std <= 0).any():
        raise ValueError("invalid SimVP train-only AGRI normalization")

    bank_rows = list(csv.DictReader(keys_path.open(newline="", encoding="utf-8")))
    if not bank_rows or any(r["split"] not in ("train", "val") for r in bank_rows):
        raise ValueError("R-COT bank keys must be train/val only")
    frame_index = {(r["split"], r["target_time_utc"]): int(r["index"]) for r in bank_rows}
    frame_coord = {(r["split"], r["target_time_utc"]): {
        "sili": tuple(map(int, r["sili_grid_rc"].split(","))),
        "zhujia": tuple(map(int, r["zhujia_grid_rc"].split(","))),
    } for r in bank_rows}
    cot_bank = np.load(cot_bank_path, mmap_mode="r", allow_pickle=False)
    if cot_bank.shape != (55733, 2, 1, 16, 16):
        raise ValueError(f"unexpected R-COT bank shape {cot_bank.shape}")

    sys.path.insert(0, str(source))
    sys.path.insert(0, str(r_source.parent))
    from train_monthly_simvp_20261002 import MonthlyDataset
    from triad_models import SimVPBackbone
    from train_station16_R import COTUNet

    args = SimpleNamespace(source=str(source), root=str(a.agri_root), cache=str(a.cache),
                           manifests=str(manifests))
    dataset = MonthlyDataset(args, norm, "val")
    if not 0 <= a.val_index < len(dataset):
        raise IndexError(f"val-index must be in [0,{len(dataset)-1}]")
    row = dataset.rows[a.val_index]
    paths = row["data_relpaths"].split("|")
    if row["split"] != "val" or len(paths) != 24:
        raise ValueError("selected row is not a 24-frame validation sequence")
    times = [datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") for rel in paths]
    if any((b - x).total_seconds() != 900 for x, b in zip(times, times[1:])):
        raise ValueError("selected window does not have 15-minute-cadence AGRI frames")
    target_keys = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times[8:]]
    idxs = [frame_index.get(("val", k)) for k in target_keys]
    if any(i is None for i in idxs):
        raise ValueError("a validation target frame is absent from the frozen R bank")

    x, geometry, y, valid, _month = dataset[a.val_index]
    x_t = torch.from_numpy(x[None]).cuda(non_blocking=True)
    g_t = torch.from_numpy(geometry[None]).cuda(non_blocking=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("metadata", {}).get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("SimVP checkpoint split provenance mismatch")
    if payload.get("metadata", {}).get("test_payloads_read") != 0:
        raise ValueError("SimVP checkpoint does not certify test exclusion")
    simvp = SimVPBackbone(str(source)).cuda().eval()
    simvp.load_state_dict(payload["model"], strict=True)
    del payload
    torch.set_num_threads(4)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        forecast_norm = simvp(x_t, g_t).float().cpu().numpy()[0]
    if forecast_norm.shape != (16, 13, 256, 256) or not np.isfinite(forecast_norm).all():
        raise ValueError(f"invalid SimVP output {forecast_norm.shape}")

    # Invert the exact train-only SimVP normalization. The observed target is
    # retained only as an offline R-retrieval reference, never fed to SimVP.
    forecast_physical = forecast_norm * std[None, :, None, None] + mean[None, :, None, None]
    observed_physical = y * std[None, :, None, None] + mean[None, :, None, None]
    if not np.isfinite(forecast_physical).all() or not np.isfinite(observed_physical).all():
        raise ValueError("nonfinite physical AGRI after inverse normalization")

    rnorm = json.loads(r_norm_path.read_text(encoding="utf-8"))
    rmean = np.asarray(rnorm["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or rstd.shape != (16, 1, 1) or (rstd <= 0).any():
        raise ValueError("invalid frozen R train-only normalization")
    rp = torch.load(r_checkpoint, map_location="cpu", weights_only=False)
    if rp.get("metadata", {}).get("training_script_sha256") != EXPECTED["r_source_sha256"]:
        raise ValueError("R checkpoint/source binding mismatch")
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(rp["model"], strict=True)
    del rp

    # R's frame-key table carries the station-centered 16x16 grid coordinates.
    r_inputs, r_ref_inputs, forecast_agri_crops, observed_agri_crops, target_valid_crops, meta = [], [], [], [], [], []
    for lead, key, bank_idx in zip(LEADS, target_keys, idxs):
        k = ("val", key)
        coords = frame_coord[k]
        for si, station in enumerate(STATIONS):
            gr, gc = coords[station]
            r0, c0 = gr - 8, gc - 8
            if r0 < 0 or c0 < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                raise ValueError(f"station patch out of bounds: {station} {coords[station]}")
            lead_index = lead // 15 - 1
            geo = geometry[8 + lead_index, :, r0:r0+16, c0:c0+16]
            pred_crop = forecast_physical[lead_index, :, r0:r0+16, c0:c0+16]
            obs_crop = observed_physical[lead_index, :, r0:r0+16, c0:c0+16]
            forecast_agri_crops.append(pred_crop.astype(np.float32, copy=False))
            observed_agri_crops.append(obs_crop.astype(np.float32, copy=False))
            target_valid_crops.append(valid[lead_index, :, r0:r0+16, c0:c0+16])
            for agri_crop, target in ((pred_crop, r_inputs), (obs_crop, r_ref_inputs)):
                raw_input = np.concatenate((agri_crop, geo), axis=0)
                normalized = (raw_input - rmean) / rstd
                if normalized.shape != (16, 16, 16) or not np.isfinite(normalized).all():
                    raise ValueError("invalid R input after physical AGRI/geometry normalization")
                target.append(normalized.astype(np.float32, copy=False))
            meta.append({"lead_minutes": lead, "station": station, "target_time_utc": key,
                         "bank_index": int(bank_idx), "grid_rc": [gr, gc]})

    # Use the same R in FP32 for predicted AGRI and observed-AGRI replay.
    all_x = torch.from_numpy(np.stack(r_inputs + r_ref_inputs)).cuda()
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        all_pred = rmodel(all_x).float().cpu().numpy()
    if all_pred.shape != (64, 1, 16, 16) or not np.isfinite(all_pred).all():
        raise ValueError(f"invalid frozen-R output {all_pred.shape}")
    forecast_cot = all_pred[:32].reshape(16, 2, 1, 16, 16)
    observed_cot = all_pred[32:].reshape(16, 2, 1, 16, 16)
    bank_ref = np.stack([cot_bank[i] for i in idxs]).astype(np.float32)
    replay_delta = np.abs(observed_cot - bank_ref)
    max_delta = float(np.max(replay_delta))
    if max_delta > 5e-3:
        raise ValueError(f"observed-AGRI -> R replay does not match saved bank: max={max_delta}")

    a.output.mkdir(parents=True)
    arrays_path = a.output / "single_val_window_cot_replay.npz"
    np.savez_compressed(
        arrays_path, forecast_log1p_cot=forecast_cot.astype(np.float32),
        observed_agri_retrieval_log1p_cot=observed_cot.astype(np.float32),
        saved_R_bank_log1p_cot=bank_ref,
        forecast_station_agri_physical=np.stack(forecast_agri_crops).reshape(16, 2, 13, 16, 16),
        target_station_agri_physical=np.stack(observed_agri_crops).reshape(16, 2, 13, 16, 16),
        valid_target_station_mask=np.stack(target_valid_crops).reshape(16, 2, 13, 16, 16).astype(np.uint8),
    )
    result = {
        "state": "PASS_SINGLE_VALIDATION_WINDOW_PHYSICAL_AGRI_R_REPLAY",
        "scope": "engineering witness only; one validation window; no test payload",
        "split": "val", "val_index": a.val_index, "seq_id": row["seq_id"],
        "BJT_start": row["BJT_start"], "target_times_utc": target_keys,
        "lead_minutes": LEADS, "sample_keys": meta,
        "SimVP_checkpoint_sha256": EXPECTED["simvp_checkpoint_sha256"],
        "SimVP_normalization_sha256": EXPECTED["simvp_norm_sha256"],
        "candidate_manifest_sha256": EXPECTED["candidate_sha256"],
        "R_checkpoint_sha256": EXPECTED["r_checkpoint_sha256"],
        "R_source_sha256": EXPECTED["r_source_sha256"],
        "R_observed_replay_max_abs_log1p_delta": max_delta,
        "SimVP_inference_precision": "bfloat16 autocast; outputs converted to FP32 before inverse train-only normalization",
        "R_inference_precision": "FP32; no autocast",
        "target_observed_AGRI_role": "offline COT retrieval reference only; never a SimVP input",
        "future_geometry_role": "known target-time solar geometry, aligned to each +15...+240 min lead",
        "test_payload_opened": False,
        "output_npz": str(arrays_path),
    }
    (a.output / "witness.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
