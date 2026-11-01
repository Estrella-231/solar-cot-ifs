#!/usr/bin/env python3
"""Predict future AGRI and frozen-R COT for one heldout train-date fold.

Only the eight observed AGRI history frames are opened. Future inputs are
deterministic geometry sidecars; no future AGRI, CPP, GHI, or test payload is
read. Outputs are keyed by the official sequence id and can be merged into the
existing train/validation bank after all five folds pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_observed_frame(load_frame, data_root: Path, rel: str, cache: Path):
    import fcntl
    dst = cache / (rel + ".npz")
    if dst.is_file():
        with np.load(dst, allow_pickle=False) as z:
            return z["a"], z["v"], z["g"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(dst) + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dst.is_file():
            with np.load(dst, allow_pickle=False) as z:
                return z["a"], z["v"], z["g"]
        image, valid, geom = load_frame(data_root, rel)
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            np.savez(f, a=image, v=valid, g=geom)
        tmp.replace(dst)
        return image, valid, geom


def observed_paths(row: dict[str, str]) -> list[str]:
    paths = row["data_relpaths"].split("|")
    if len(paths) != 24:
        raise ValueError(f"expected 24 frame paths, got {len(paths)}")
    return paths[:8]


def future_geometry_paths(row: dict[str, str]) -> list[str]:
    paths = row["data_relpaths"].split("|")
    if len(paths) != 24:
        raise ValueError(f"expected 24 frame paths, got {len(paths)}")
    return paths[8:]


def main() -> None:
    import torch
    p = argparse.ArgumentParser()
    for name in ("project", "simvp_source", "fold_manifest", "fold_run", "fold_oof_ids",
                 "bank_index", "r_root", "data_root", "cache", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--batch-sequences", type=int, default=16)
    p.add_argument("--r-batch-patches", type=int, default=1024)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OOF AGRI-to-COT inference")
    source = a.simvp_source
    manifest = a.fold_manifest
    checkpoint = a.fold_run / "training/epoch_012.pt"
    norm_path = manifest / "hunan_train_norm.json"
    audit_path = manifest / "manifest_audit.json"
    ids_path = a.fold_oof_ids
    if not checkpoint.is_file() or not norm_path.is_file():
        raise FileNotFoundError("fold checkpoint or fold normalization is missing")
    training_complete = json.loads((checkpoint.parent / "training_complete.json").read_text())
    if training_complete.get("selected_checkpoint") != "epoch_012.pt" or training_complete.get("heldout_payload_read") is not False:
        raise ValueError("fold trainer did not certify fixed-epoch, heldout-isolated training")
    split_audit = json.loads(audit_path.read_text())
    if split_audit.get("frame_disjoint_splits") is not True or split_audit.get("train_val_frame_intersection") != 0:
        raise ValueError("fold frame-disjoint audit failed")
    norm = json.loads(norm_path.read_text())
    if norm.get("validation_test_used") is not False:
        raise ValueError("fold normalization is not train-only")

    oof_ids = ids_path.read_text(encoding="utf-8").splitlines()
    if not oof_ids or len(oof_ids) != len(set(oof_ids)):
        raise ValueError("fold OOF ID list is empty or duplicated")
    held_rows = {r["seq_id"]: r for r in rows(manifest / "hunan_8to16.csv") if r["split"] == "val"}
    if not set(oof_ids).issubset(held_rows):
        raise ValueError("OOF IDs do not map to heldout rows in this fold manifest")
    bank_train = [r for r in rows(a.bank_index) if r["split"] == "train"]
    bank_ids = [r["seq_id"] for r in bank_train]
    if len(bank_ids) != len(set(bank_ids)) or not set(oof_ids).issubset(bank_ids):
        raise ValueError("fold OOF IDs do not uniquely map to the GHI training bank")
    selected = [held_rows[s] for s in oof_ids]

    r_script = a.r_root / "scripts/train_station16_R.py"
    r_ckpt = a.r_root / "results/seed42_v1/best.pt"
    r_norm_path = a.r_root / "results/seed42_v1/normalization.json"
    r_keys_path = a.r_root / "results/new_monthly_R_seed42_cot_bank_v1/frame_keys.csv"
    source_bank_manifest = a.output.parent.parent / "future_simvp_r_bank_v1/bank_manifest.json"
    if not source_bank_manifest.is_file():
        # The normal runtime path is under this experiment's results directory.
        source_bank_manifest = a.project / "refine-logs/cot_aware_simvp_20261017/results/future_simvp_r_bank_v1/bank_manifest.json"
    source_meta = json.loads(source_bank_manifest.read_text())
    if sha(r_ckpt) != source_meta["r_checkpoint_sha256"] or sha(r_script) != source_meta["r_source_sha256"]:
        raise ValueError("frozen R checkpoint/source differs from the validated forecast bank")

    sys.path.insert(0, str(source))
    sys.path.insert(0, str(r_script.parent))
    from hunan_data import load_frame, resolve_aux
    from triad_models import SimVPBackbone
    from train_station16_R import COTUNet

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 12 or payload.get("metadata", {}).get("fold") != split_audit["fold"]:
        raise ValueError("fold checkpoint metadata does not match fixed-epoch manifest")
    simvp = SimVPBackbone(str(source)).cuda().eval()
    simvp.load_state_dict(payload["model"], strict=True)
    del payload
    amean = np.asarray(norm["mean"], np.float32)
    astd = np.asarray(norm["std"], np.float32)
    rpayload = torch.load(r_ckpt, map_location="cpu", weights_only=False)
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(rpayload["model"], strict=True)
    if rpayload.get("metadata", {}).get("training_script_sha256") != sha(r_script):
        raise ValueError("R checkpoint/source binding mismatch")
    del rpayload
    rnorm = json.loads(r_norm_path.read_text())
    rmean = np.asarray(rnorm["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or rstd.shape != (16, 1, 1) or (rstd <= 0).any():
        raise ValueError("invalid frozen-R normalization")

    frame_coord = {}
    with r_keys_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                frame_coord[r["target_time_utc"]] = {s: tuple(map(int, r[f"{s}_grid_rc"].split(","))) for s in STATIONS}

    fold = int(split_audit["fold"])
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    contract = {"fold": fold, "fold_manifest_sha256": sha(manifest / "hunan_8to16.csv"),
                "fold_checkpoint_sha256": sha(checkpoint), "fold_norm_sha256": sha(norm_path),
                "fold_oof_ids_sha256": sha(ids_path), "r_checkpoint_sha256": sha(r_ckpt),
                "r_source_sha256": sha(r_script), "bank_index_sha256": sha(a.bank_index),
                "batch_sequences": a.batch_sequences, "r_batch_patches": a.r_batch_patches}
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    progress_path = out / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("state") != "BUILDING" or progress.get("contract_sha256") != contract_sha:
            raise ValueError("existing partial fold bank has another contract")
        start = int(progress["sequences_complete"])
        agri = np.load(out / "forecast_agri_raw.npy", mmap_mode="r+", allow_pickle=False)
        cot = np.load(out / "forecast_cot_log1p.npy", mmap_mode="r+", allow_pickle=False)
        if agri.shape != (len(selected), 2, 16, 13, 3, 3) or cot.shape != (len(selected), 2, 16, 1, 3, 3):
            raise ValueError("partial fold bank array shape mismatch")
    else:
        if list(out.iterdir()):
            raise FileExistsError("output exists without a resumable progress receipt")
        agri = np.lib.format.open_memmap(out / "forecast_agri_raw.npy", mode="w+", dtype=np.float32,
                                          shape=(len(selected), 2, 16, 13, 3, 3))
        cot = np.lib.format.open_memmap(out / "forecast_cot_log1p.npy", mode="w+", dtype=np.float32,
                                       shape=(len(selected), 2, 16, 1, 3, 3))
        start = 0
        atomic_json(progress_path, {"state": "BUILDING", "sequences_complete": 0,
                                   "sequences_total": len(selected), "contract_sha256": contract_sha,
                                   "future_observed_agri_read": False, "cpp_read": False})

    cache = a.cache
    cache.mkdir(parents=True, exist_ok=True)
    for lo in range(start, len(selected), a.batch_sequences):
        batch_rows = selected[lo:lo + a.batch_sequences]
        x_rows, g_rows, path_rows, coords_rows, target_rows = [], [], [], [], []
        for row in batch_rows:
            paths = row["data_relpaths"].split("|")
            x_hist, g_hist = [], []
            for rel in observed_paths(row):
                image, valid, geom = load_observed_frame(load_frame, a.data_root, rel, cache)
                if image.shape != (13, 256, 256) or geom.shape != (3, 256, 256):
                    raise ValueError(f"invalid observed AGRI/geometry shape for {rel}")
                x_hist.append(np.where(valid, (image - amean[:, None, None]) / astd[:, None, None], 0).astype(np.float32))
                g_hist.append(geom.astype(np.float32))
            g_future, coords, targets = [], [], []
            for rel in future_geometry_paths(row):
                stamp = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
                aux = resolve_aux(a.data_root, stamp + timedelta(hours=8))
                with np.load(aux, allow_pickle=False) as z:
                    geo = np.stack([z[k] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
                if geo.shape != (3, 256, 256) or not np.isfinite(geo).all():
                    raise ValueError(f"invalid future geometry sidecar for {rel}")
                key = stamp.strftime("%Y-%m-%d %H:%M:%S")
                if key not in frame_coord:
                    raise ValueError(f"missing station-grid coordinates for {key}")
                g_future.append(geo); coords.append(frame_coord[key]); targets.append(key)
            x_rows.append(np.stack(x_hist)); g_rows.append(np.stack(g_hist + g_future))
            path_rows.append(paths); coords_rows.append(coords); target_rows.append(targets)
        x = torch.from_numpy(np.stack(x_rows)).cuda(non_blocking=True)
        g = torch.from_numpy(np.stack(g_rows)).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            forecast = simvp(x, g).float().cpu().numpy()
        forecast = forecast * astd[None, None, :, None, None] + amean[None, None, :, None, None]
        if forecast.shape != (len(batch_rows), 16, 13, 256, 256) or not np.isfinite(forecast).all():
            raise ValueError("invalid OOF SimVP physical AGRI forecast")

        r_inputs, dests = [], []
        for bi in range(len(batch_rows)):
            for ti in range(16):
                for si, station in enumerate(STATIONS):
                    gr, gc = coords_rows[bi][ti][station]
                    r0, c0 = gr - 8, gc - 8
                    if min(r0, c0) < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                        raise ValueError(f"station patch outside 256 grid: {station} {(gr,c0)}")
                    pa = forecast[bi, ti, :, r0:r0+16, c0:c0+16]
                    pg = g_rows[bi][8 + ti][:, r0:r0+16, c0:c0+16]
                    inp = (np.concatenate((pa, pg), axis=0) - rmean) / rstd
                    if not np.isfinite(inp).all():
                        raise ValueError("nonfinite R input")
                    r_inputs.append(inp.astype(np.float32))
                    dests.append((bi, si, ti, pa.copy()))
        cot_patch = np.empty((len(r_inputs), 1, 16, 16), np.float32)
        with torch.inference_mode():
            for rlo in range(0, len(r_inputs), a.r_batch_patches):
                batch = torch.from_numpy(np.stack(r_inputs[rlo:rlo+a.r_batch_patches])).cuda(non_blocking=True)
                cot_patch[rlo:rlo+len(batch)] = rmodel(batch).float().cpu().numpy()
        if not np.isfinite(cot_patch).all() or (cot_patch < 0).any():
            raise ValueError("invalid frozen-R log1p-COT output")
        for ri, (bi, si, ti, pa) in enumerate(dests):
            agri[lo+bi, si, ti] = pa[:, 7:10, 7:10]
            cot[lo+bi, si, ti] = cot_patch[ri, :, 7:10, 7:10]
        agri.flush(); cot.flush()
        atomic_json(progress_path, {"state": "BUILDING", "sequences_complete": lo + len(batch_rows),
                                    "sequences_total": len(selected), "contract_sha256": contract_sha,
                                    "future_observed_agri_read": False, "cpp_read": False})
        print("OOF_FORECAST_PROGRESS", lo + len(batch_rows), len(selected), flush=True)

    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("bank_index", "split", "seq_id"), lineterminator="\n")
        w.writeheader()
        for i, row in enumerate(selected):
            w.writerow({"bank_index": i, "split": "train", "seq_id": row["seq_id"]})
    result = {"state": "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS", "fold": fold,
              "rows": len(selected), "stations": list(STATIONS), "leads": 16,
              "layouts": {"forecast_agri_raw.npy": list(agri.shape), "forecast_cot_log1p.npy": list(cot.shape)},
              "forecast_source": "fixed epoch-12 date-fold SimVP; fold-heldout train dates only",
              "cot_source": "frozen R applied to predicted AGRI and deterministic future geometry",
              "fold_checkpoint_sha256": sha(checkpoint), "fold_norm_sha256": sha(norm_path),
              "r_checkpoint_sha256": sha(r_ckpt), "r_source_sha256": sha(r_script),
              "manifest_sha256": sha(manifest / "hunan_8to16.csv"), "fold_oof_ids_sha256": sha(ids_path),
              "future_observed_agri_read": False, "cpp_read": False,
              "validation_test_payload_opened": False, "buffer_payload_opened": False,
              "contract_sha256": contract_sha,
              "array_sha256": {"forecast_agri_raw.npy": sha(out / "forecast_agri_raw.npy"),
                               "forecast_cot_log1p.npy": sha(out / "forecast_cot_log1p.npy")},
              "index_sha256": sha(out / "index.csv")}
    atomic_json(out / "bank_manifest.json", result)
    atomic_json(progress_path, {"state": "COMPLETE", "sequences_complete": len(selected),
                                "sequences_total": len(selected), "contract_sha256": contract_sha,
                                "future_observed_agri_read": False, "cpp_read": False})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
