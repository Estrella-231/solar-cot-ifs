#!/usr/bin/env python3
"""Build candidate full-train SimVP -> frozen-R COT forecasts for val keys.

Only the eight observed AGRI history frames are loaded. Future frame paths are
used to fetch deterministic geometry sidecars; future AGRI and CPP payloads are
never opened. The output is aligned exactly to the existing baseline bank's
validation seq_id order (which is a subset of the full monthly val manifest).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from droppath0_validation_manifest import select_validation_rows

STATIONS = ("sili", "zhujia")
ARRAY_SHAPES = {
    "forecast_agri_raw.npy": (13, 3, 3),
    "forecast_cot_log1p.npy": (1, 3, 3),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="DropPath-0 source snapshot")
    p.add_argument("--fulltrain-run", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--r-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--base-bank", type=Path, required=True)
    p.add_argument("--frame-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-sequences", type=int, default=8)
    p.add_argument("--r-batch-patches", type=int, default=1024)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build forecast-derived COT")

    manifest_path = a.manifests / "hunan_8to16.csv"
    norm_path = a.manifests / "hunan_train_norm.json"
    audit_path = a.manifests / "manifest_audit.json"
    ckpt_path = a.fulltrain_run / "training/best.pt"
    train_receipt_path = a.fulltrain_run / "training/training_complete.json"
    base_index_path = a.base_bank / "index.csv"
    r_script = a.r_root / "scripts/train_station16_R.py"
    r_ckpt_path = a.r_root / "results/seed42_v1/best.pt"
    r_norm_path = a.r_root / "results/seed42_v1/normalization.json"
    r_frame_keys_path = a.r_root / "results/new_monthly_R_seed42_cot_bank_v1/frame_keys.csv"

    for path in (manifest_path, norm_path, audit_path, ckpt_path,
                 train_receipt_path, base_index_path, r_script,
                 r_ckpt_path, r_norm_path, r_frame_keys_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_meta = json.loads((a.base_bank / "bank_manifest.json").read_text())
    if base_meta.get("state") != "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK":
        raise ValueError("base forecast bank is not complete")
    for field in ("test_payload_opened", "buffer_payload_opened",
                  "future_observed_agri_read", "cpp_read"):
        if base_meta.get(field) is not False:
            raise ValueError(f"base bank has invalid provenance: {field}")

    audit = json.loads(audit_path.read_text())
    norm = json.loads(norm_path.read_text())
    manifest_sha = sha(manifest_path)
    if audit.get("frame_disjoint_splits") is not True or audit.get("train_val_frame_intersection") != 0:
        raise ValueError("monthly manifest split isolation audit failed")
    if norm.get("validation_test_used") is not False or norm.get("manifest_sha256") != manifest_sha:
        raise ValueError("normalization is not bound to train-only candidate manifest")
    if norm.get("train_frames_sha256") != sha(a.manifests / "train_frames.txt"):
        raise ValueError("normalization/train-frame hash mismatch")

    train_receipt = json.loads(train_receipt_path.read_text())
    ckpt_sha = sha(ckpt_path)
    if train_receipt.get("state") != "COMPLETE_DROPPATH0_FULLTRAIN_VALIDATION_SELECTED":
        raise ValueError("full-train candidate is not complete")
    if train_receipt.get("best_checkpoint_sha256") != ckpt_sha:
        raise ValueError("selected full-train checkpoint hash mismatch")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    md = payload.get("metadata", {})
    source_hashes = {
        "triad_models.py": sha(a.source / "triad_models.py"),
        "simvp.py": sha(a.source / "simvp.py"),
        "hunan_data.py": sha(a.source / "hunan_data.py"),
        "train_monthly_simvp_20261002.py": sha(a.source / "train_monthly_simvp_20261002.py"),
    }
    expected_md = {
        "drop_path_prob": 0.0,
        "manifest_sha256": manifest_sha,
        "normalization_sha256": sha(norm_path),
        "model_code_sha256": source_hashes["triad_models.py"],
        "data_code_sha256": source_hashes["hunan_data.py"],
        "simvp_code_sha256": source_hashes["simvp.py"],
    }
    for k, v in expected_md.items():
        if md.get(k) != v:
            raise ValueError(f"selected candidate checkpoint mismatch for {k}: {md.get(k)!r} != {v!r}")
    if md.get("test_payloads_read") != 0 or md.get("validation_future_agri_used_as_model_input") is not False:
        raise ValueError("candidate checkpoint provenance gate failed")
    if md.get("buffer_payload_opened") is not False or md.get("future_observed_agri_inference_input") is not False or md.get("cpp_read") is not False:
        raise ValueError("candidate checkpoint opened a forbidden source")

    manifest_rows = read_csv(manifest_path)
    base_rows = read_csv(base_index_path)
    selected, val_ids, manifest_indices = select_validation_rows(manifest_rows, base_rows)
    val_rows = [r for r in manifest_rows if r["split"] == "val"]

    # Station pixel positions are static grid metadata. Confirm every available
    # R frame-key row agrees before reusing them for validation-time coordinates.
    key_rows = read_csv(r_frame_keys_path)
    if not key_rows:
        raise ValueError("R frame-key table is empty")
    coords = {}
    for station in STATIONS:
        values = {tuple(map(int, r[f"{station}_grid_rc"].split(","))) for r in key_rows}
        if len(values) != 1:
            raise ValueError(f"station pixel coordinates are not static for {station}: {values}")
        coords[station] = next(iter(values))
    if coords != {"sili": (163, 148), "zhujia": (73, 150)}:
        raise ValueError(f"station coordinate contract mismatch: {coords}")

    r_payload = torch.load(r_ckpt_path, map_location="cpu", weights_only=False)
    if r_payload.get("metadata", {}).get("training_script_sha256") != sha(r_script):
        raise ValueError("frozen-R checkpoint/source mismatch")
    rnorm = json.loads(r_norm_path.read_text())
    rmean = np.asarray(rnorm["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or rstd.shape != (16, 1, 1) or (rstd <= 0).any():
        raise ValueError("invalid frozen-R normalization")

    contract = {
        "state": "DROPPATH0_FULLTRAIN_VALIDATION_FORECAST_BANK_BUILDING",
        "candidate_checkpoint_sha256": ckpt_sha,
        "candidate_source_hashes": source_hashes,
        "manifest_sha256": manifest_sha,
        "normalization_sha256": sha(norm_path),
        "base_bank_manifest_sha256": sha(a.base_bank / "bank_manifest.json"),
        "base_validation_index_sha256": sha(base_index_path),
        "r_checkpoint_sha256": sha(r_ckpt_path),
        "r_script_sha256": sha(r_script),
        "selected_val_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "validation_sequences": len(val_ids),
        "validation_manifest_sequences": len(val_rows),
        "validation_sequences_missing_from_base_bank": len(val_rows) - len(val_ids),
        "station_grid_rc": {k: list(v) for k, v in coords.items()},
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_read": False, "cpp_read": False,
        "validation_payload_used_as_input": False,
    }
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    n = len(selected)
    if a.resume:
        progress_path = a.output / "progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError("--resume needs an existing progress.json")
        prior = json.loads(progress_path.read_text())
        if prior.get("state") != "BUILDING" or prior.get("contract_sha256") != contract_sha:
            raise ValueError("partial validation bank contract mismatch")
        start = int(prior["sequences_complete"])
        if not 0 <= start < n:
            raise ValueError("invalid or already-complete resume index")
        arrays = {name: np.load(a.output / name, mmap_mode="r+", allow_pickle=False)
                  for name in ARRAY_SHAPES}
        if any(arr.shape != (n, 2, 16, *ARRAY_SHAPES[name]) or arr.dtype != np.float32
               for name, arr in arrays.items()):
            raise ValueError("partial bank array shape/dtype mismatch")
    else:
        if a.output.exists():
            raise FileExistsError(f"refusing to overwrite {a.output}")
        a.output.mkdir(parents=True)
        start = 0
        arrays = {name: np.lib.format.open_memmap(
            a.output / name, mode="w+", dtype=np.float32,
            shape=(n, 2, 16, *shape)) for name, shape in ARRAY_SHAPES.items()}
        with (a.output / "index.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=("bank_index", "split", "sequence_index_within_split", "seq_id"),
                               lineterminator="\n")
            w.writeheader()
            for i, row in enumerate(selected):
                w.writerow({"bank_index": i, "split": "val",
                            "sequence_index_within_split": manifest_indices[i], "seq_id": val_ids[i]})
        write_json(a.output / "progress.json", {"state": "BUILDING", "sequences_complete": 0,
                    "sequences_total": n, "contract_sha256": contract_sha,
                    "test_payload_opened": False, "future_observed_agri_read": False,
                    "cpp_read": False})

    sys.path.insert(0, str(a.source))
    sys.path.insert(0, str(r_script.parent))
    from hunan_data import load_frame, resolve_aux
    from triad_models import SimVPBackbone
    from train_station16_R import COTUNet

    norm_mean = np.asarray(norm["mean"], np.float32)
    norm_std = np.asarray(norm["std"], np.float32)
    if norm_mean.shape != (13,) or norm_std.shape != (13,) or (norm_std <= 0).any():
        raise ValueError("invalid candidate AGRI normalization")
    simvp = SimVPBackbone(str(a.source)).cuda().eval()
    simvp.load_state_dict(payload["model"], strict=True)
    del payload
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(r_payload["model"], strict=True)
    del r_payload
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True

    @lru_cache(maxsize=128)
    def future_geometry(rel: str):
        utc = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
        aux = resolve_aux(a.data_root, utc + timedelta(hours=8))
        with np.load(aux, allow_pickle=False) as z:
            g = np.stack([z[k] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
        if g.shape != (3, 256, 256) or not np.isfinite(g).all():
            raise ValueError(f"invalid future deterministic geometry for {rel}")
        return g

    def cached_frame(rel: str):
        dst = a.frame_cache / (rel + ".npz")
        if dst.is_file():
            with np.load(dst, allow_pickle=False) as z:
                return z["a"], z["v"], z["g"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(dst) + ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dst.is_file():
                with np.load(dst, allow_pickle=False) as z:
                    return z["a"], z["v"], z["g"]
            image, valid, geom = load_frame(a.data_root, rel)
            tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
            with tmp.open("wb") as f:
                np.savez(f, a=image, v=valid, g=geom)
            tmp.replace(dst)
        return image, valid, geom

    started = time.monotonic()
    for lo in range(start, n, a.batch_sequences):
        batch_rows = selected[lo:lo + a.batch_sequences]
        x_batch, g_batch, path_rows = [], [], []
        for row in batch_rows:
            paths = row["data_relpaths"].split("|")
            hist = []
            geos = []
            for rel in paths[:8]:
                image, valid, geom = cached_frame(rel)
                hist.append(np.where(valid, (image - norm_mean[:, None, None]) /
                                     norm_std[:, None, None], 0).astype(np.float32))
                geos.append(geom.astype(np.float32))
            future_geos = [future_geometry(rel) for rel in paths[8:]]
            if len(hist) != 8 or len(future_geos) != 16:
                raise ValueError("expected 8 observed inputs and 16 deterministic future geometry frames")
            x_batch.append(np.stack(hist))
            g_batch.append(np.stack(geos + future_geos))
            path_rows.append(paths)
        x = torch.from_numpy(np.stack(x_batch)).cuda(non_blocking=True)
        g = torch.from_numpy(np.stack(g_batch)).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_norm = simvp(x, g).float().cpu().numpy()
        pred_phys = pred_norm * norm_std[None, None, :, None, None] + norm_mean[None, None, :, None, None]
        if pred_phys.shape != (len(batch_rows), 16, 13, 256, 256) or not np.isfinite(pred_phys).all():
            raise ValueError("candidate SimVP generated invalid future AGRI")

        inputs = []
        destinations = []
        for bi, paths in enumerate(path_rows):
            for lead in range(16):
                t = datetime.strptime(Path(paths[lead + 8]).stem, "%Y%m%d%H%M")
                geom = g_batch[bi][lead + 8]
                for si, station in enumerate(STATIONS):
                    gr, gc = coords[station]
                    r0, c0 = gr - 8, gc - 8
                    ag = pred_phys[bi, lead, :, r0:r0 + 16, c0:c0 + 16]
                    ge = geom[:, r0:r0 + 16, c0:c0 + 16]
                    rin = (np.concatenate((ag, ge), axis=0) - rmean) / rstd
                    if not np.isfinite(rin).all():
                        raise ValueError("nonfinite candidate predicted-AGRI R input")
                    inputs.append(rin.astype(np.float32))
                    destinations.append((bi, si, lead, ag.copy()))
        cots = np.empty((len(inputs), 1, 16, 16), np.float32)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            for rlo in range(0, len(inputs), a.r_batch_patches):
                xb = torch.from_numpy(np.stack(inputs[rlo:rlo + a.r_batch_patches])).cuda(non_blocking=True)
                cots[rlo:rlo + len(xb)] = rmodel(xb).float().cpu().numpy()
        if not np.isfinite(cots).all() or (cots < 0).any():
            raise ValueError("candidate frozen-R output is invalid")
        cot_seq = np.full((len(batch_rows), 2, 16, 1, 3, 3), np.nan, np.float32)
        for j, (bi, si, lead, ag) in enumerate(destinations):
            arrays["forecast_agri_raw.npy"][lo + bi, si, lead] = ag[:, 7:10, 7:10]
            cot_seq[bi, si, lead] = cots[j, :, 7:10, 7:10]
        if not np.isfinite(cot_seq).all():
            raise ValueError("candidate COT map has unfilled station/lead values")
        arrays["forecast_cot_log1p.npy"][lo:lo + len(batch_rows)] = cot_seq
        for arr in arrays.values():
            arr.flush()
        done = lo + len(batch_rows)
        write_json(a.output / "progress.json", {"state": "BUILDING", "sequences_complete": done,
                    "sequences_total": n, "contract_sha256": contract_sha,
                    "elapsed_seconds": time.monotonic() - started,
                    "test_payload_opened": False, "buffer_payload_opened": False,
                    "future_observed_agri_read": False, "cpp_read": False})
        print(f"DROPPATH0_VALIDATION_FORECAST_PROGRESS {done}/{n}", flush=True)

    index_sha = sha(a.output / "index.csv")
    array_sha = {name: sha(a.output / name) for name in ARRAY_SHAPES}
    complete = {
        **contract,
        "state": "COMPLETE_DROPPATH0_FULLTRAIN_VALIDATION_FORECASTS",
        "selected_checkpoint_sha256": ckpt_sha,
        "simvp_checkpoint_sha256": ckpt_sha,
        "index_sha256": index_sha,
        "array_sha256": array_sha,
        "layouts": {name: list(arr.shape) for name, arr in arrays.items()},
        "forecast_source": "DropPath-0 full-train validation-selected SimVP; only observed eight AGRI input frames",
        "forecast_cot_source": "frozen R applied to predicted AGRI plus deterministic geometry; no CPP",
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_read": False, "cpp_read": False,
        "validation_payload_used_as_input": False,
        "producer_sha256": sha(Path(__file__)),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(a.output / "bank_manifest.json", complete)
    (a.output / "progress.json").unlink()
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
