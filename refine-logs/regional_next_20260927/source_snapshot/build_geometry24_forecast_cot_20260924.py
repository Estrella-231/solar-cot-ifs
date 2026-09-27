"""Build geometry24 Hunan backbone -> frozen R COT forecasts.

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
    def __init__(self, source_root, data_root, manifest, norm, cache_dir, split, limit=None):
        sys.path.insert(0, str(source_root))
        from hunan_data import HunanDataset, load_frame, resolve_aux
        if split not in ("train", "val"):
            raise ValueError("test split is intentionally unavailable")
        self.base = HunanDataset(data_root, manifest, norm, split)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.base.rows = self.base.rows[:limit]
        self.load_frame, self.resolve_aux = load_frame, resolve_aux
        self.data_root = Path(data_root)
        self.mean, self.std = self.base.mean, self.base.std
        self.cache_dir = Path(cache_dir)
        self.geometry_cache = self.cache_dir.parent / (self.cache_dir.name + "_geometry")

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

    def nominal_geometry(self, rel):
        """Read only deterministic target-time geometry, never target AGRI."""
        import fcntl
        from datetime import datetime, timedelta
        utc = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
        bjt = utc + timedelta(hours=8)
        target = self.geometry_cache / f"{bjt:%Y%m%d%H%M}.npy"
        if target.exists():
            return np.load(target, allow_pickle=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                return np.load(target, allow_pickle=False)
            aux_path = self.resolve_aux(self.data_root, bjt)
            with np.load(aux_path, allow_pickle=False) as aux:
                geometry = np.stack([aux[key] for key in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
            if geometry.shape != (3, 256, 256) or not np.isfinite(geometry).all():
                raise RuntimeError(f"invalid target-time geometry: {aux_path}")
            temp = target.with_name(target.name + ".%d.tmp" % os.getpid())
            with temp.open("wb") as stream:
                np.save(stream, geometry)
            temp.replace(target)
        return geometry

    def __getitem__(self, index):
        row = self.base.rows[index]
        rels = row["data_relpaths"].split("|")
        if len(rels) != 24:
            raise RuntimeError("not a 24-frame manifest row")
        from datetime import datetime, timedelta
        times = [datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") for rel in rels]
        if any(right-left != timedelta(minutes=15) for left, right in zip(times, times[1:])):
            raise RuntimeError("noncanonical frame order")
        if times[0] + timedelta(hours=8) != datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S"):
            raise RuntimeError("BJT/UTC mismatch")
        images, geos = [], []
        for rel in rels[:8]:
            agri, valid, geometry = self.historical_frame(rel)
            images.append(np.where(valid, (agri - self.mean) / self.std, 0).astype(np.float32))
            geos.append(geometry)
        # Resolve geometry from auxiliary files for future nominal times.  Do
        # not call load_frame on rels[8:] because it would load future AGRI.
        for rel in rels[8:24]:
            geos.append(self.nominal_geometry(rel))
        geometry = np.stack(geos).astype(np.float32)
        if len(images) != 8 or geometry.shape != (24, 3, 256, 256):
            raise RuntimeError("geometry24 forecast input contract mismatch")
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
    parser.add_argument("--backbone", choices=("simvp", "afno_transformer"), default="simvp")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-sequences", type=int)
    args = parser.parse_args()

    s_config = json.loads((args.solar_root / "configs/s_frozen_hunan_seed42.json").read_text())
    r_config = json.loads((args.solar_root / "configs/r_frozen_repaired_seed42.json").read_text())
    run_name = ("simvp_geometry24_formal_e12_v1" if args.backbone == "simvp"
                else "afno_transformer_geometry24_formal_e12_v1")
    backbone_path = args.triad_root / "geometry24_v2/runs" / run_name / "best.pt"
    for path, digest in ((Path(s_config["manifest"]), s_config["manifest_sha256"]),
                         (Path(s_config["normalization"]), s_config["normalization_sha256"]),
                         (Path(r_config["checkpoint"]), r_config["checkpoint_sha256"])):
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("frozen Hunan asset hash mismatch: " + str(path))
    if sha(args.data_root / "grid_static.npz") != s_config["grid_sha256"]:
        raise RuntimeError("Hunan grid contract mismatch")
    if not backbone_path.is_file():
        raise RuntimeError("missing trained geometry24 Hunan checkpoint")
    if args.limit_sequences is None and not (backbone_path.parent / "training_complete.json").is_file():
        raise RuntimeError("formal backbone bank requires finished training")

    cot_module = load_module(args.solar_root / "scripts/train_cot_repaired.py", "cot_unet_afno_fixed_ghi")
    norm = json.loads(Path(s_config["normalization"]).read_text())
    backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=False)
    metadata = backbone_state["metadata"]
    model_source = args.triad_root / "geometry24_v2/triad_models.py"
    if sha(model_source) != metadata["model_code_sha256"]:
        raise RuntimeError("geometry24 model source hash does not match checkpoint")
    backbone_module = load_module(model_source, "triad_models_geometry24_cot")
    if (metadata["manifest_sha256"] != s_config["manifest_sha256"] or
            metadata["normalization_sha256"] != s_config["normalization_sha256"] or
            metadata.get("geometry_contract") != "history_8_then_target_16_v2" or
            metadata.get("model") != args.backbone or metadata.get("region") != "Hunan"):
        raise RuntimeError("backbone training cohort or geometry contract mismatch")
    if args.backbone == "simvp":
        backbone = backbone_module.SimVPBackbone(args.hunan_source)
    else:
        backbone = backbone_module.AFNOTransformer()
    backbone = backbone.cuda().eval().requires_grad_(False)
    backbone.load_state_dict(backbone_state["model"], strict=True)
    r_state = torch.load(r_config["checkpoint"], map_location="cpu", weights_only=False)
    if r_state["metadata"]["source_code_sha256"] != sha(args.solar_root / "scripts/train_cot_repaired.py"):
        raise RuntimeError("frozen R source code hash mismatch")
    r = cot_module.COTUNet().cuda().eval().requires_grad_(False)
    r.load_state_dict(r_state["model"], strict=True)
    rnorm = r_state["metadata"]["norm"]

    args.output.mkdir(parents=True, exist_ok=False)
    cache_key = sha(s_config["manifest"])[:16]
    dataset = CausalInputs(args.hunan_source, args.data_root, s_config["manifest"], norm,
                           f"/tmp/slfu_hunan_triad_frames_{cache_key}", args.split, args.limit_sequences)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=1 if args.workers else None)
    patches = [tuple(v) for v in s_config["station_patches"].values()]
    mean = torch.as_tensor(dataset.mean, device="cuda")[None, None]
    std = torch.as_tensor(dataset.std, device="cuda")[None, None]
    output_array = np.lib.format.open_memmap(args.output / "forecast_cot_log1p.npy", mode="w+",
                                             dtype=np.float32, shape=(len(dataset), 2, 16, 1, 16, 16))
    image_array = np.lib.format.open_memmap(args.output / "forecast_image_physical.npy", mode="w+",
                                            dtype=np.float32, shape=(len(dataset), 2, 16, 16, 16, 16))
    seen = np.zeros(len(dataset), dtype=np.bool_)
    started = time.monotonic()
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    for batch_no, (indices, history, geometry) in enumerate(loader):
        idx = indices.numpy().astype(np.int64)
        if seen[idx].any():
            raise RuntimeError("duplicate %s sequence indices" % args.split)
        history = history.cuda(non_blocking=True)
        geometry = geometry.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized_prediction = backbone(history, geometry)
        physical = normalized_prediction.float() * std + mean
        physical_patch = patch(physical, patches)
        geometry_patch = patch(geometry[:, 8:], patches)
        if physical_patch.shape != (len(idx), 2, 16, 13, 16, 16) or geometry_patch.shape != (len(idx), 2, 16, 3, 16, 16):
            raise RuntimeError("future image/geometry patch alignment mismatch")
        image_array[idx] = torch.cat((physical_patch, geometry_patch), dim=3).float().cpu().numpy()
        raw = torch.cat((physical_patch, geometry_patch), dim=3).flatten(0, 2)
        rmean = torch.as_tensor(rnorm["mean"], device="cuda")[None, :, None, None]
        rstd = torch.as_tensor(rnorm["std"], device="cuda")[None, :, None, None]
        normalized = (raw - rmean) / rstd
        normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        with torch.autocast("cuda", enabled=False):
            cot = r(normalized.float()).reshape(len(idx), 2, 16, 16, 16)
        if not torch.isfinite(cot).all():
            raise RuntimeError("geometry24 COT prediction contains non-finite values")
        output_array[idx, :, :, 0] = cot.float().cpu().numpy()
        seen[idx] = True
        if batch_no % 25 == 0:
            status = {"state": "RUNNING", "samples": int(seen.sum()), "total": len(dataset),
                      "seconds": time.monotonic() - started}
            atomic_json(args.output / "status.json", status)
            print("PROGRESS", json.dumps(status), flush=True)
    output_array.flush()
    image_array.flush()
    if not seen.all():
        raise RuntimeError("incomplete geometry24 %s COT bank" % args.split)

    index_path = args.output / "index.csv"
    with index_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "seq_id", "split", "BJT_start"])
        for index, row in enumerate(dataset.base.rows):
            writer.writerow([index, row["seq_id"], row["split"], row["BJT_start"]])
    del output_array, image_array
    contract = {
        "state": "PILOT_GEOMETRY24_COT" if args.limit_sequences is not None else "COMPLETE_GEOMETRY24_FORECAST_COT",
        "samples": len(dataset), "test_used": False,
        "split": args.split, "region": "Hunan", "stations": list(s_config["station_patches"]),
        "shape": [len(dataset), 2, 16, 1, 16, 16], "layout": "sequence, station, lead, channel, row, col",
        "prediction_source": f"Hunan {args.backbone} geometry24 best validation checkpoint",
        "retriever": "frozen Hunan repaired COTUNet; FP32 forward",
        "inputs": "8 observed AGRI frames + deterministic geometry at history and target times",
        "future_AGRI_read": False, "future_CPP_read": False, "GHI_read": False,
        "lead_minutes": s_config["lead_minutes"], "station_patches": s_config["station_patches"],
        "backbone": args.backbone, "geometry_contract": "history_8_then_target_16_v2",
        "geometry_forward_indices": list(range(8, 24)),
        "backbone_checkpoint": str(backbone_path), "backbone_checkpoint_sha256": sha(backbone_path),
        "backbone_model_source": str(model_source), "backbone_model_source_sha256": sha(model_source),
        "R_checkpoint": r_config["checkpoint"], "R_checkpoint_sha256": r_config["checkpoint_sha256"],
        "manifest": s_config["manifest"], "manifest_sha256": s_config["manifest_sha256"],
        "normalization_sha256": s_config["normalization_sha256"],
        "comparison_scope": "geometry24 COT bank for matched source-conditioned GHI-head retraining",
        "script_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "throughput_sequences_per_second": len(dataset) / max(time.monotonic() - started, 1e-9),
        "batch": args.batch, "workers": args.workers,
        "forecast_cot_sha256": sha(args.output / "forecast_cot_log1p.npy"),
        "forecast_image_sha256": sha(args.output / "forecast_image_physical.npy"),
        "index_sha256": sha(index_path),
    }
    atomic_json(args.output / "complete.json", contract)
    print("GEOMETRY24_COT_COMPLETE", json.dumps({"split": args.split, "samples": len(dataset), "test_used": False,
                                            "seconds": contract["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
