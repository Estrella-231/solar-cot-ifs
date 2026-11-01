#!/usr/bin/env python3
"""Build U-only R_U teacher frame cache and train paired P-C/P-XC forecasters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

STATIONS = ("sili", "zhujia")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_sha(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cache(a) -> dict:
    """Create a unique-frame station-patch cache; only U frame paths are opened."""
    urows = list(csv.DictReader((a.support / "u_sequences.csv").open(newline="", encoding="utf-8")))
    if not urows:
        raise ValueError("empty U sequence manifest")
    support = load_json(a.support / "support_audit.json")
    if support.get("state") != "PASS_24_FRAME_SUPPORT_DISJOINT_U_H_V":
        raise ValueError("U/H/V 24-frame support audit did not pass")
    if support.get("test_or_buffer_numeric_payloads_opened") != 0 or support.get("test_rows_selected") != 0:
        raise ValueError("support audit violates test/buffer gate")

    # Map UTC AGRI relative paths to exact unique-frame indices. Keep manifest order.
    paths = sorted({p for r in urows for p in r["data_relpaths"].split("|")})
    if len(paths) != int(support["sets"]["U"]["unique_support_frames"]):
        raise ValueError("unique U path count differs from frozen support receipt")
    path_to_i = {p: i for i, p in enumerate(paths)}
    seq_indices = np.asarray([[path_to_i[p] for p in r["data_relpaths"].split("|")] for r in urows], dtype=np.int32)
    if seq_indices.shape != (len(urows), 24):
        raise ValueError("U sequence path index array is not N x 24")
    if len({r["seq_id"] for r in urows}) != len(urows):
        raise ValueError("duplicate U sequence id")

    # Recover station centers from the actual audited R_U source rows.
    centers = {}
    for r in csv.DictReader((a.r_pack / "rows_u_supervised.csv").open(newline="", encoding="utf-8")):
        rc = tuple(map(int, r["station_grid_rc"].split(",")))
        if r["station"] in centers and centers[r["station"]] != rc:
            raise ValueError(f"station center changes over time: {r['station']}")
        centers[r["station"]] = rc
    if set(centers) != set(STATIONS):
        raise ValueError(f"station centers missing: {centers}")

    rstatus = load_json(a.r_run / "status.json")
    rmeta = load_json(a.r_run / "run_metadata.json")
    rnorm_obj = load_json(a.r_run / "normalization.json")
    rckpt_path = a.r_run / "epoch_050.pt"
    if rstatus.get("state") != "COMPLETE_R_U_FIXED_EPOCH_50" or rstatus.get("last_epoch") != 50:
        raise ValueError("R_U fixed-50 receipt missing")
    for name, expected in (("checkpoint_sha256", sha(rckpt_path)),
                           ("normalization_sha256", sha(a.r_run / "normalization.json"))):
        if rstatus.get(name) != expected:
            raise ValueError(f"R_U {name} mismatch")
    if rmeta.get("h_rows_used") != 0 or rmeta.get("outer_validation_rows_used") != 0:
        raise ValueError("R_U metadata includes heldout supervision")

    cache = a.cache
    cache.mkdir(parents=True, exist_ok=False)
    n = len(paths)
    ag = np.lib.format.open_memmap(cache / "agri.npy", mode="w+", dtype=np.float32,
                                   shape=(n, 2, 13, 16, 16))
    qa = np.lib.format.open_memmap(cache / "qa.npy", mode="w+", dtype=np.uint8,
                                   shape=(n, 2, 13, 16, 16))
    geo = np.lib.format.open_memmap(cache / "geometry.npy", mode="w+", dtype=np.float32,
                                    shape=(n, 2, 3, 16, 16))
    teacher = np.lib.format.open_memmap(cache / "cot_teacher.npy", mode="w+", dtype=np.float32,
                                        shape=(n, 2, 1, 16, 16))
    validcot = np.lib.format.open_memmap(cache / "cot_valid.npy", mode="w+", dtype=np.uint8,
                                         shape=(n, 2, 1, 16, 16))

    sys.path.insert(0, str(a.source))
    from hunan_data import CHANNELS, resolve_aux
    sys.path.insert(0, str(Path(__file__).parent))
    from train_r_u import COTUNet
    sys.path.insert(0, str(a.geometry_source.parent))
    from solar_geometry_corrected import compute_solar_angles
    with np.load(a.grid_static, allow_pickle=False) as z:
        lat = np.asarray(z["lat"], np.float32)
        lon = np.asarray(z["lon"], np.float32)
        saa = np.asarray(z["saa"], np.float32)
    if any(q.shape != (256, 256) for q in (lat, lon, saa)):
        raise ValueError("corrected R_U geometry requires the audited Hunan 256x256 grid")

    rmodel = COTUNet().cuda().eval().requires_grad_(False)
    rpayload = torch.load(rckpt_path, map_location="cpu", weights_only=False)
    rmodel.load_state_dict(rpayload["model"], strict=True)
    del rpayload
    rmean = np.asarray(rnorm_obj["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm_obj["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or rstd.shape != rmean.shape or np.any(rstd <= 0):
        raise ValueError("invalid R_U normalization")

    vcount = 0
    for start in range(0, n, a.frame_batch):
        end = min(start + a.frame_batch, n)
        model_inputs = []
        valid_rows, geom_rows = [], []
        for ix in range(start, end):
            dt = datetime.strptime(Path(paths[ix]).stem, "%Y%m%d%H%M")
            raw_path = a.data_root / "data" / paths[ix]
            raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
            if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
                raise ValueError(f"unexpected raw AGRI frame for U path {paths[ix]}: {raw.shape}/{raw.dtype}")
            aux = resolve_aux(a.data_root, dt + timedelta(hours=8))
            with np.load(aux, allow_pickle=False) as z:
                bits = np.asarray(z["agri_valid_bits"], np.uint16)
            frame_patch = []
            frame_valid = []
            frame_geo = []
            for si, station in enumerate(STATIONS):
                rr, cc = centers[station]
                r0, c0 = rr - 8, cc - 8
                sl = np.s_[r0:r0 + 16, c0:c0 + 16]
                if r0 < 0 or c0 < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                    raise ValueError(f"station patch outside 256 grid: {station} {centers[station]}")
                # Read only the 13-channel station crop from the memory-mapped
                # full frame; materializing the 256x256 cube per timestamp
                # made the first attempt storage-bound for no scientific gain.
                aa = np.asarray(raw[np.asarray(CHANNELS), sl[0], sl[1]], np.float32)
                patch_bits = bits[sl]
                vv = np.stack([(patch_bits & (1 << k)) != 0 for k in CHANNELS]) & np.isfinite(aa)
                soz, soa = compute_solar_angles(lat[sl], lon[sl], dt)
                physical_night = soz >= 90.0
                soa = np.where(physical_night, 0.0, soa)
                gg = np.stack((np.cos(np.deg2rad(soz)),
                               np.cos(np.deg2rad(soa - saa[sl])),
                               (soz < 85.0).astype(np.float32))).astype(np.float32)
                if not np.isfinite(gg).all():
                    raise ValueError(f"nonfinite geometry for {paths[ix]} {station}")
                frame_patch.append(aa)
                frame_valid.append(vv)
                frame_geo.append(gg)
                inp_raw = np.concatenate((aa, gg), axis=0)
                inp = (inp_raw - rmean) / rstd
                inp[~np.isfinite(inp)] = 0.0
                # The archived CPP/R pack marks a pixel's 13 AGRI channels
                # jointly invalid with NaN when any channel QA fails; the R_U
                # trainer normalizes first and maps those nonfinite slots to 0.
                inp[:13, ~vv.all(axis=0)] = 0.0
                model_inputs.append(inp.astype(np.float32))
            ag[ix] = np.stack(frame_patch)
            qa[ix] = np.stack(frame_valid).astype(np.uint8)
            geo[ix] = np.stack(frame_geo)
            valid_rows.append(np.stack(frame_valid))
            geom_rows.append(np.stack(frame_geo))
        tin = torch.from_numpy(np.stack(model_inputs)).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pred = rmodel(tin).float().cpu().numpy()
        if pred.shape != (2 * (end - start), 1, 16, 16) or not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError(f"R_U teacher output invalid at frame range {start}:{end}")
        pred = pred.reshape(end - start, 2, 1, 16, 16)
        vv = np.stack(valid_rows)
        gg = np.stack(geom_rows)
        daylight_support = (gg[:, :, 2:3] > 0.5) & (gg[:, :, 0:1] > math.cos(math.radians(80)))
        q_support = vv.all(axis=2, keepdims=True)
        cmask = daylight_support & q_support & np.isfinite(pred)
        teacher[start:end] = pred
        validcot[start:end] = cmask.astype(np.uint8)
        vcount += int(cmask.sum())
        if end == n or end % max(a.frame_batch * 10, a.frame_batch) == 0:
            print("FRAME_CACHE", json.dumps({"frames": end, "total": n, "valid_cot_pixels": vcount}), flush=True)
    for arr in (ag, qa, geo, teacher, validcot):
        arr.flush()

    # Train-only normalization for AGRI and log1p historical COT inputs.
    ag_mean, ag_std = np.asarray(load_json(a.simvp_norm)["mean"], np.float64), np.asarray(load_json(a.simvp_norm)["std"], np.float64)
    if ag_mean.shape != (13,) or ag_std.shape != (13,) or (ag_std <= 0).any():
        raise ValueError("invalid fold1 U-only SimVP AGRI normalization")
    # Stream moments on U COT support to avoid loading the full teacher array.
    total = 0.0; second = 0.0; count = 0
    for start in range(0, n, 256):
        end = min(start + 256, n)
        mask = np.asarray(validcot[start:end], bool)
        z = np.log1p(np.asarray(teacher[start:end], np.float32))
        vals = z[mask & np.isfinite(z)].astype(np.float64)
        total += vals.sum(dtype=np.float64); second += np.square(vals, dtype=np.float64).sum(); count += vals.size
    if count == 0:
        raise ValueError("no valid U history COT support for normalization")
    cot_mean = total / count
    cot_std = max(math.sqrt(max(second / count - cot_mean * cot_mean, 1e-8)), 1e-4)

    # Sequence indices and normalization/evidence are small; data arrays are cache-only.
    np.save(cache / "sequence_indices.npy", seq_indices, allow_pickle=False)
    seqids_path = cache / "sequence_ids.txt"
    seqids_path.write_text("\n".join(r["seq_id"] for r in urows) + "\n", encoding="utf-8")
    cache_meta = {
        "state": "COMPLETE_U_ONLY_RU_TEACHER_CACHE", "frames": n,
        "sequence_rows": len(urows), "station_windows": len(urows) * 2,
        "frame_path_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "u_sequence_manifest_sha256": sha(a.support / "u_sequences.csv"),
        "support_audit_sha256": sha(a.support / "support_audit.json"),
        "r_checkpoint_sha256": sha(rckpt_path), "r_normalization_sha256": sha(a.r_run / "normalization.json"),
        "simvp_u_normalization_sha256": sha(a.simvp_norm),
        "station_centers": {k: list(v) for k, v in centers.items()},
        "agri_mean": ag_mean.tolist(), "agri_std": ag_std.tolist(),
        "history_cot_log1p_mean": cot_mean, "history_cot_log1p_std": cot_std,
        "history_cot_norm_fit": "U unique frame station pixels with corrected day_mask>0.5, cosSOZ>cos80 and any valid AGRI channel",
        "teacher": "frozen U-only R_U applied to real AGRI only for U teacher targets; no H/V future AGRI used",
        "training_target_mask": "future corrected geometry support and all 13 future AGRI QA channels valid; matches corrected R pack eligibility and is used only for U teacher loss",
        "u_frames_only": True, "h_payloads_read": 0, "outer_validation_payloads_read": 0,
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "frame_batch": a.frame_batch, "valid_cot_pixels": vcount,
        "artifacts_sha256": {k: sha(cache / k) for k in ("agri.npy", "qa.npy", "geometry.npy", "cot_teacher.npy", "cot_valid.npy", "sequence_indices.npy", "sequence_ids.txt")},
    }
    atomic_json(cache / "cache_manifest.json", cache_meta)
    return cache_meta


class UFrameSequenceDataset(Dataset):
    def __init__(self, cache: Path, mode: str):
        self.cache = cache
        self.mode = mode
        self.meta = load_json(cache / "cache_manifest.json")
        if self.meta.get("state") != "COMPLETE_U_ONLY_RU_TEACHER_CACHE":
            raise ValueError("U teacher cache is incomplete")
        for k, digest in self.meta["artifacts_sha256"].items():
            if sha(cache / k) != digest:
                raise ValueError(f"U cache artifact hash mismatch: {k}")
        self.indices = np.load(cache / "sequence_indices.npy", mmap_mode="r", allow_pickle=False)
        self.agri = np.load(cache / "agri.npy", mmap_mode="r", allow_pickle=False)
        self.qa = np.load(cache / "qa.npy", mmap_mode="r", allow_pickle=False)
        self.geo = np.load(cache / "geometry.npy", mmap_mode="r", allow_pickle=False)
        self.cot = np.load(cache / "cot_teacher.npy", mmap_mode="r", allow_pickle=False)
        self.valid = np.load(cache / "cot_valid.npy", mmap_mode="r", allow_pickle=False)
        self.ag_mean = np.asarray(self.meta["agri_mean"], np.float32)[None, :, None, None]
        self.ag_std = np.asarray(self.meta["agri_std"], np.float32)[None, :, None, None]
        self.cot_mean = float(self.meta["history_cot_log1p_mean"])
        self.cot_std = float(self.meta["history_cot_log1p_std"])

    def __len__(self):
        return len(self.indices) * 2

    def __getitem__(self, item):
        seq = item // 2
        station = item % 2
        ix = np.asarray(self.indices[seq], np.int64)
        hist = ix[:8]
        fut = ix[8:]
        a = np.asarray(self.agri[hist, station], np.float32)
        q = np.asarray(self.qa[hist, station], np.float32)
        g = np.asarray(self.geo[hist, station], np.float32)
        c = np.asarray(self.cot[hist, station], np.float32)
        cv = np.asarray(self.valid[hist, station], np.float32)
        an = (a - self.ag_mean) / self.ag_std
        an[~np.isfinite(an)] = 0.0
        # QA is retained for both groups. Only the 13 AGRI value slots differ.
        hist_cot = np.log1p(np.maximum(c, 0.0))
        hist_cot = (hist_cot - self.cot_mean) / self.cot_std
        hist_cot = np.where(cv > 0.5, hist_cot, 0.0).astype(np.float32)
        cv = np.where(cv > 0.5, 1.0, 0.0).astype(np.float32)
        x = np.concatenate((an, q, hist_cot, cv, g), axis=1)  # [8,31,H,W]
        if self.mode == "C":
            x[:, :13] = 0.0
        elif self.mode != "XC":
            raise ValueError(self.mode)
        fg = np.asarray(self.geo[fut, station], np.float32)
        y = np.asarray(self.cot[fut, station], np.float32)
        m = np.asarray(self.valid[fut, station], bool)
        if not np.isfinite(x).all() or not np.isfinite(fg).all() or not np.isfinite(y[m]).all():
            raise ValueError(f"nonfinite U sequence item {item}")
        return x, fg, y, m


class ConvGNAct(nn.Module):
    def __init__(self, ci, co, kernel=3, stride=1, padding=1):
        super().__init__()
        groups = 8 if co % 8 == 0 else 4
        self.block = nn.Sequential(nn.Conv3d(ci, co, kernel, stride=stride, padding=padding, bias=False),
                                   nn.GroupNorm(groups, co), nn.SiLU())
    def forward(self, x): return self.block(x)


class DirectCotForecaster(nn.Module):
    """3-scale 8→16 field forecaster with time-projected spatial skip features."""
    def __init__(self, in_channels=31):
        super().__init__()
        self.hi1, self.hi2 = ConvGNAct(in_channels, 32), ConvGNAct(32, 32)
        self.down1 = ConvGNAct(32, 64, kernel=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.mid1, self.mid2 = ConvGNAct(64, 64), ConvGNAct(64, 64)
        self.down2 = ConvGNAct(64, 96, kernel=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.low1, self.low2 = ConvGNAct(96, 96), ConvGNAct(96, 96)
        self.time_hi, self.time_mid, self.time_low = nn.Linear(8, 16), nn.Linear(8, 16), nn.Linear(8, 16)
        self.low_to_mid = nn.Conv3d(96, 64, 1)
        self.mid_dec = ConvGNAct(128, 64)
        self.mid_to_hi = nn.Conv3d(64, 32, 1)
        self.geo_embed = nn.Conv3d(3, 16, 1)
        self.hi_dec = ConvGNAct(32 + 32 + 16, 32)
        self.out = nn.Conv3d(32, 1, 1)

    @staticmethod
    def temporal(module, x):
        # [B,C,T,H,W] -> [B,C,16,H,W]
        return module(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

    def forward(self, x, future_geometry):
        # x [B,8,31,16,16]; geometry [B,16,3,16,16]
        x = x.permute(0, 2, 1, 3, 4)
        hi = self.hi2(self.hi1(x))
        mid = self.mid2(self.mid1(self.down1(hi)))
        low = self.low2(self.low1(self.down2(mid)))
        hi = self.temporal(self.time_hi, hi)
        mid = self.temporal(self.time_mid, mid)
        low = self.temporal(self.time_low, low)
        low = F.interpolate(self.low_to_mid(low), size=(16, 8, 8), mode="trilinear", align_corners=False)
        mid = self.mid_dec(torch.cat((mid, low), dim=1))
        mid = F.interpolate(self.mid_to_hi(mid), size=(16, 16, 16), mode="trilinear", align_corners=False)
        geom = self.geo_embed(future_geometry.permute(0, 2, 1, 3, 4))
        hi = self.hi_dec(torch.cat((hi, mid, geom), dim=1))
        z = F.softplus(self.out(hi))
        return torch.expm1(torch.clamp(z, max=20.0)).clamp_min(0.0)


def safe_huber(pred, target, mask):
    # Compare in log1p(COT); eliminate invalid values before arithmetic.
    p = torch.where(mask, torch.log1p(pred.clamp_min(0)), torch.zeros_like(pred))
    t = torch.where(mask, torch.log1p(target.clamp_min(0)), torch.zeros_like(target))
    e = F.huber_loss(p, t, reduction="none", delta=0.1)
    return (e * mask).sum() / mask.sum().clamp_min(1)


def profile_model(model, batch, device):
    x, g, y, m = [z.to(device) for z in batch]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x, g)
        loss = safe_huber(pred, y, m)
    if not torch.isfinite(loss): raise RuntimeError("P profile produced nonfinite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    torch.cuda.synchronize(device)
    result = {"state": "PASS_DIRECT_COT_SINGLE_GPU_PROFILE", "batch": len(x),
              "device": torch.cuda.get_device_name(device), "loss": float(loss.detach()),
              "grad_norm": float(grad), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
              "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
              "total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
              "profile_update_used_for_training": False}
    result["peak_allocated_fraction"] = result["peak_allocated_bytes"] / result["total_bytes"]
    result["peak_reserved_fraction"] = result["peak_reserved_bytes"] / result["total_bytes"]
    if result["peak_allocated_fraction"] >= 0.85 or result["peak_reserved_fraction"] >= 0.90:
        raise RuntimeError("P batch128 profile exceeds planned GPU headroom")
    return result


def train_one(mode: str, a, dataset: Dataset, init_state: dict, init_digest: str, output: Path, seed=42):
    output.mkdir(parents=True, exist_ok=False)
    model = DirectCotForecaster().cuda()
    model.load_state_dict(init_state, strict=True)
    if init_sha(model) != init_digest:
        raise RuntimeError("paired initial state hash mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader_gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0,
                        generator=loader_gen, drop_last=False)
    steps = len(loader)
    if steps != math.ceil(len(dataset) / 128):
        raise RuntimeError("unexpected U update count")
    metrics = output / "metrics.jsonl"
    metadata = {
        "candidate": f"P-{mode}", "initialization_seed": seed, "initial_state_sha256": init_digest,
        "architecture": "three-scale 16x16/8x8/4x4 causal-time encoder; temporal 8-to-16 projections; spatial skip decoder; GN+SiLU",
        "input_contract": "history [B,8,31,16,16], future geometry [B,16,3,16,16]; C zeros normalized AGRI-value slots only; XC retains them",
        "u_sequence_manifest_sha256": sha(a.support / "u_sequences.csv"),
        "u_cache_manifest_sha256": sha(a.cache / "cache_manifest.json"),
        "train_windows": len(dataset), "effective_batch": 128, "epochs": 12,
        "updates_per_epoch": steps, "planned_updates": 12 * steps,
        "optimizer": "AdamW lr=1e-4 wd=1e-4 cosine to 1e-6; grad clip 1.0",
        "loss": "masked Huber(delta=0.1) in log1p physical COT; mask uses future U AGRI QA and computable geometry only",
        "precision": "CUDA BF16 autocast after finite loss/profile check",
        "p_code_sha256": sha(Path(__file__)), "r_u_checkpoint_sha256": load_json(a.r_run / "status.json")["checkpoint_sha256"],
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "h_payloads_read": 0, "outer_validation_payloads_read": 0,
        "pbs_job_id": os.environ.get("PBS_JOBID"), "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(output / "run_metadata.json", metadata)
    start = time.monotonic()
    for epoch in range(1, 13):
        model.train()
        epoch_loss = 0.0; valid_pix = 0; epoch_start = time.monotonic()
        # Same order in P-C/P-XC: dataset length and generator seed are identical.
        order_gen = torch.Generator().manual_seed(seed + epoch - 1)
        sampler = torch.utils.data.RandomSampler(dataset, generator=order_gen)
        epoch_loader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=a.workers,
                                  pin_memory=True, persistent_workers=a.workers > 0, drop_last=False)
        for step, (x, g, y, mask) in enumerate(epoch_loader, start=1):
            x = x.cuda(non_blocking=True); g = g.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True).bool()
            progress = ((epoch - 1) + (step - 1) / steps) / 12.0
            lr = 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): pred = model(x, g)
            loss = safe_huber(pred, y, mask)
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite P-{mode} loss e{epoch}s{step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            nvalid = int(mask.sum())
            epoch_loss += float(loss.detach()) * nvalid; valid_pix += nvalid
            if step % 100 == 0:
                print("P_TRAIN_STEP", json.dumps({"mode": mode, "epoch": epoch,
                      "step": (epoch - 1) * steps + step, "loss": float(loss.detach()), "lr": lr}), flush=True)
        rec = {"mode": mode, "epoch": epoch, "steps": steps,
               "train_huber_log1p": epoch_loss / max(valid_pix, 1), "valid_pixels": valid_pix,
               "epoch_seconds": time.monotonic() - epoch_start, "elapsed_seconds": time.monotonic() - start,
               "fixed_budget": True}
        with metrics.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, allow_nan=False) + "\n")
        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
              "metadata": metadata, "record": rec}
        tmp = output / "last.pt.tmp"; torch.save(ck, tmp); tmp.replace(output / "last.pt")
        if epoch == 12:
            tmp = output / "epoch_012.pt.tmp"; torch.save(ck, tmp); tmp.replace(output / "epoch_012.pt")
        atomic_json(output / "status.json", {"state": "TRAINING", **rec})
        print("P_EPOCH_END", json.dumps(rec), flush=True)
    complete = {"state": f"COMPLETE_P_{mode}_FIXED_EPOCH_12", "epochs": 12,
                "updates": 12 * steps, "checkpoint_sha256": sha(output / "epoch_012.pt"),
                "run_metadata_sha256": sha(output / "run_metadata.json"),
                "test_payloads_read": 0, "buffer_payloads_read": 0,
                "heldout_payload_read": False, "elapsed_seconds": time.monotonic() - start}
    atomic_json(output / "status.json", complete)
    print("P_COMPLETE", json.dumps({"mode": mode, **complete}), flush=True)
    return complete


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("cache", "train"), required=True)
    for name in ("support", "r-pack", "r-run", "simvp-norm", "source", "data-root", "grid-static", "geometry-source", "cache", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--frame-batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("U R_U teacher generation/training requires CUDA")
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    if a.mode == "cache":
        audit = build_cache(a)
        print("U_TEACHER_CACHE_COMPLETE", json.dumps(audit), flush=True)
        return
    meta = load_json(a.cache / "cache_manifest.json")
    if meta.get("state") != "COMPLETE_U_ONLY_RU_TEACHER_CACHE": raise ValueError("missing complete U cache")
    if a.output.exists(): raise FileExistsError(a.output)
    dataset_xc = UFrameSequenceDataset(a.cache, "XC")
    dataset_c = UFrameSequenceDataset(a.cache, "C")
    if len(dataset_c) != len(dataset_xc): raise ValueError("paired P data size mismatch")
    # Fixed smoke/profile from the first U batch. It is discarded and not part of training.
    seed_all(42)
    init_model = DirectCotForecaster()
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    init_digest = init_sha(init_model)
    rng = torch.Generator().manual_seed(4242)
    sample_ids = torch.randperm(len(dataset_xc), generator=rng)[:128].tolist()
    profile_batch = [torch.stack([dataset_xc[i][j] for i in sample_ids]) for j in range(4)]
    profile_model_obj = DirectCotForecaster().cuda(); profile_model_obj.load_state_dict(init_state)
    profile = profile_model(profile_model_obj, profile_batch, torch.device("cuda:0"))
    atomic_json(a.output.parent / "profile.json", profile)
    print("P_SINGLE_GPU_PROFILE", json.dumps(profile), flush=True)
    del profile_model_obj
    torch.cuda.empty_cache()
    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output / "paired_initialization.json", {"state_sha256": init_digest,
                "seed": 42, "same_architecture": True, "same_u_order": True,
                "profile_update_used_for_training": False})
    result_c = train_one("C", a, dataset_c, init_state, init_digest, a.output / "P-C", 42)
    result_xc = train_one("XC", a, dataset_xc, init_state, init_digest, a.output / "P-XC", 42)
    atomic_json(a.output / "paired_training_complete.json", {
        "state": "COMPLETE_P_C_AND_P_XC_FIXED_EPOCH_12", "P-C": result_c, "P-XC": result_xc,
        "same_initialization_sha256": init_digest, "u_only": True,
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "h_payloads_read": 0, "outer_validation_payloads_read": 0,
    })
    print("P_PAIRED_COMPLETE", json.dumps({"P-C": result_c, "P-XC": result_xc}), flush=True)


if __name__ == "__main__": main()
