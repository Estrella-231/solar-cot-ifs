"""Build AFNO->frozen-R COT forecasts for a frozen Hunan GHI-head swap.

Only historical AGRI and deterministic target-time solar geometry are read.
No future AGRI, CPP, GHI, or test rows are accessed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CausalInputs(Dataset):
    """History AGRI + nominal geometry only; future AGRI is never loaded."""
    def __init__(self, source_root, data_root, manifest, norm, cache_dir, split):
        sys.path.insert(0, str(source_root))
        from hunan_data import HunanDataset, load_frame, resolve_aux
        if split not in ("train", "val"):
            raise ValueError("test split is intentionally unavailable")
        self.base = HunanDataset(data_root, manifest, norm, split)
        self.load_frame, self.resolve_aux = load_frame, resolve_aux
        self.data_root = Path(data_root)
        self.mean, self.std = self.base.mean, self.base.std
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.base)

    def historical_frame(self, rel):
        import fcntl
        target = self.cache_dir / (str(rel) + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as pack:
                return pack["a"], pack["v"], pack["g"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(target.suffix + ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                with np.load(target, allow_pickle=False) as pack:
                    return pack["a"], pack["v"], pack["g"]
            a, valid, geometry = self.load_frame(self.data_root, rel)
            temp = target.with_name(target.name + ".%d.tmp" % os.getpid())
            with temp.open("wb") as stream:
                np.savez(stream, a=a, v=valid, g=geometry)
            temp.replace(target)
        return a, valid, geometry

    def __getitem__(self, index):
        row = self.base.rows[index]
        rels = row["data_relpaths"].split("|")
        images, geos = [], []
        for rel in rels[:8]:
            agri, valid, geometry = self.historical_frame(rel)
            images.append(np.where(valid, (agri - self.mean) / self.std, 0).astype(np.float32))
            geos.append(geometry)
        # Resolve geometry from auxiliary files for future nominal times.  Do
        # not call load_frame on rels[8:] because it would load future AGRI.
        from datetime import datetime, timedelta
        for rel in rels[8:24]:
            utc = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
            bjt = utc + timedelta(hours=8)
            aux_path = self.resolve_aux(self.data_root, bjt)
            with np.load(aux_path, allow_pickle=False) as aux:
                geos.append(np.stack([aux[key] for key in ("cosSOZ", "cosRAA", "day_mask")]))
        geometry = np.stack(geos[:16]).astype(np.float32)
        if len(images) != 8 or geometry.shape != (16, 3, 256, 256):
            raise RuntimeError("AFNO forecast input contract mismatch")
        return index, np.stack(images), geometry


def patch(tensor, patches):
    return torch.stack([tensor[..., r0:r1, c0:c1] for r0, r1, c0, c1 in patches], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triad-root", type=Path, required=True)
    parser.add_argument("--hunan-source", type=Path, required=True)
    parser.add_argument("--solar-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    s_config = json.loads((args.solar_root / "configs/s_frozen_hunan_seed42.json").read_text())
    r_config = json.loads((args.solar_root / "configs/r_frozen_repaired_seed42.json").read_text())
    triad_config = args.triad_root / "runs/afno_transformer_hunan_full_e12_b8x8_seed42_v1"
    afno_path = triad_config / "best.pt"
    for path, digest in ((Path(s_config["manifest"]), s_config["manifest_sha256"]),
                         (Path(s_config["normalization"]), s_config["normalization_sha256"]),
                         (Path(s_config["checkpoint"]), s_config["checkpoint_sha256"]),
                         (Path(r_config["checkpoint"]), r_config["checkpoint_sha256"])):
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("frozen Hunan asset hash mismatch: " + str(path))
    if sha(args.data_root / "grid_static.npz") != s_config["grid_sha256"]:
        raise RuntimeError("Hunan grid contract mismatch")
    if not afno_path.is_file():
        raise RuntimeError("missing trained Hunan AFNO checkpoint")

    afno_module = load_module(args.triad_root / "triad_models.py", "triad_models_afno_fixed_ghi")
    cot_module = load_module(args.solar_root / "scripts/train_cot_repaired.py", "cot_unet_afno_fixed_ghi")
    norm = json.loads(Path(s_config["normalization"]).read_text())
    afno_state = torch.load(afno_path, map_location="cpu", weights_only=False)
    if (afno_state["metadata"]["manifest_sha256"] != s_config["manifest_sha256"] or
            afno_state["metadata"]["normalization_sha256"] != s_config["normalization_sha256"]):
        raise RuntimeError("AFNO training contract is not the fixed GHI cohort")
    afno = afno_module.AFNOTransformer().cuda().eval().requires_grad_(False)
    afno.load_state_dict(afno_state["model"], strict=True)
    r_state = torch.load(r_config["checkpoint"], map_location="cpu", weights_only=False)
    if r_state["metadata"]["source_code_sha256"] != sha(args.solar_root / "scripts/train_cot_repaired.py"):
        raise RuntimeError("frozen R source code hash mismatch")
    r = cot_module.COTUNet().cuda().eval().requires_grad_(False)
    r.load_state_dict(r_state["model"], strict=True)
    rnorm = r_state["metadata"]["norm"]

    cache_key = sha(s_config["manifest"])[:16]
    dataset = CausalInputs(args.hunan_source, args.data_root, s_config["manifest"], norm,
                           f"/tmp/slfu_hunan_triad_frames_{cache_key}", args.split)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=1 if args.workers else None)
    patches = [tuple(v) for v in s_config["station_patches"].values()]
    mean = torch.as_tensor(dataset.mean, device="cuda")[None, None]
    std = torch.as_tensor(dataset.std, device="cuda")[None, None]
    output_array = np.lib.format.open_memmap(args.output / "forecast_cot_log1p.npy", mode="w+",
                                             dtype=np.float32, shape=(len(dataset), 2, 16, 1, 16, 16))
    seen = np.zeros(len(dataset), dtype=np.bool_)
    started = time.monotonic()
    torch.backends.cudnn.benchmark = True
    for batch_no, (indices, history, geometry) in enumerate(loader):
        idx = indices.numpy().astype(np.int64)
        if seen[idx].any():
            raise RuntimeError("duplicate %s sequence indices" % args.split)
        history = history.cuda(non_blocking=True)
        geometry = geometry.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized_prediction = afno(history, geometry)
        physical = normalized_prediction.float() * std + mean
        physical_patch = patch(physical, patches)
        geometry_patch = patch(geometry, patches)
        raw = torch.cat((physical_patch, geometry_patch), dim=3).flatten(0, 2)
        rmean = torch.as_tensor(rnorm["mean"], device="cuda")[None, :, None, None]
        rstd = torch.as_tensor(rnorm["std"], device="cuda")[None, :, None, None]
        normalized = (raw - rmean) / rstd
        normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        with torch.autocast("cuda", enabled=False):
            cot = r(normalized.float()).reshape(len(idx), 2, 16, 16, 16)
        if not torch.isfinite(cot).all():
            raise RuntimeError("AFNO COT prediction contains non-finite values")
        output_array[idx, :, :, 0] = cot.float().cpu().numpy()
        seen[idx] = True
        if batch_no % 25 == 0:
            status = {"state": "RUNNING", "samples": int(seen.sum()), "total": len(dataset),
                      "seconds": time.monotonic() - started}
            atomic_json(args.output / "status.json", status)
            print("PROGRESS", json.dumps(status), flush=True)
    output_array.flush()
    if not seen.all():
        raise RuntimeError("incomplete AFNO %s COT bank" % args.split)

    index_path = args.output / "index.csv"
    with index_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "seq_id", "split", "BJT_start"])
        for index, row in enumerate(dataset.base.rows):
            writer.writerow([index, row["seq_id"], row["split"], row["BJT_start"]])
    del output_array
    contract = {
        "state": "COMPLETE_AFNO_FORECAST_COT_TRAINVAL" if args.split == "train" else "COMPLETE_AFNO_FORECAST_COT_VALIDATION",
        "samples": len(dataset), "test_used": False,
        "split": args.split, "region": "Hunan", "stations": list(s_config["station_patches"]),
        "shape": [len(dataset), 2, 16, 1, 16, 16], "layout": "sequence, station, lead, channel, row, col",
        "prediction_source": "Hunan AFNO+Transformer best validation checkpoint",
        "retriever": "frozen Hunan repaired COTUNet; FP32 forward",
        "inputs": "8 observed AGRI frames + deterministic geometry at history and target times",
        "future_AGRI_read": False, "future_CPP_read": False, "GHI_read": False,
        "lead_minutes": s_config["lead_minutes"], "station_patches": s_config["station_patches"],
        "afno_checkpoint": str(afno_path), "afno_checkpoint_sha256": sha(afno_path),
        "R_checkpoint": r_config["checkpoint"], "R_checkpoint_sha256": r_config["checkpoint_sha256"],
        "manifest": s_config["manifest"], "manifest_sha256": s_config["manifest_sha256"],
        "normalization_sha256": s_config["normalization_sha256"],
        "baseline_forecast_cot_audit_sha256": sha(args.solar_root / "data/forecast_cot_trajectory_trainval_20260918_v1/audit.json"),
        "comparison_scope": "exploratory AFNO-COT bank for matched GHI-head retraining",
        "script_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - started,
        "forecast_cot_sha256": sha(args.output / "forecast_cot_log1p.npy"),
        "index_sha256": sha(index_path),
    }
    atomic_json(args.output / "complete.json", contract)
    print("AFNO_COT_COMPLETE", json.dumps({"split": args.split, "samples": len(dataset), "test_used": False,
                                            "seconds": contract["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
