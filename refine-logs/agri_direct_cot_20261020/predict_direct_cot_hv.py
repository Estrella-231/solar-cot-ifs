#!/usr/bin/env python3
"""Causal H/V inference for direct future-COT and common SimVP AGRI inputs.

This inference-only stage reads the 8 observed AGRI history frames and known
geometry sidecars for H/V sequences. It never opens future AGRI, CPP, GHI,
test, or buffer payloads. Reference labels are joined by a separate builder
after this script writes its inference-complete receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
SPLITS = ("h", "v")
CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
SOZ_COS_80 = float(np.cos(np.deg2rad(80.0)))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def validate_r_geometry_provenance(r_pack: Path, geometry_source: Path) -> dict:
    """Bind inference geometry to the immutable U-only pack audit receipt."""
    audit_path = r_pack / "pack_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("state") != "PASS_U_ONLY_R_PACK" or audit.get("train_rows_only") is not True:
        raise ValueError("R_U pack audit is not a train-only approved pack")
    if audit.get("test_payloads_read") != 0 or audit.get("buffer_payloads_read") != 0:
        raise ValueError("R_U pack audit reports test/buffer payload access")
    if sha(geometry_source) != audit.get("geometry_source_sha256"):
        raise ValueError("R_U geometry source hash mismatch against pack audit")
    return audit


def atomic_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FrameLRU:
    """Bounded cache over observed full AGRI frames only."""
    def __init__(self, loader, root: Path, capacity: int = 256):
        self.loader, self.root, self.capacity = loader, root, capacity
        self.data: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, rel: str):
        if rel in self.data:
            self.data.move_to_end(rel)
            return self.data[rel]
        image, valid, geometry = self.loader(self.root, rel)
        image = np.asarray(image, np.float32)
        valid = np.asarray(valid, bool)
        geometry = np.asarray(geometry, np.float32)
        if image.shape != (13, 256, 256) or valid.shape != image.shape or geometry.shape != (3, 256, 256):
            raise ValueError(f"invalid observed frame contract for {rel}: {image.shape}/{valid.shape}/{geometry.shape}")
        value = image, valid, geometry
        self.data[rel] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return value


class ArrayLRU:
    """Bounded cache for geometry arrays only; never caches future imagery."""
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.data: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, key: tuple[str, str], loader):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        value = np.asarray(loader(), np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite geometry for {key[1]}")
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return value


def load_geometry(resolve_aux, data_root: Path, rel: str) -> np.ndarray:
    stamp = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
    aux = resolve_aux(data_root, stamp + timedelta(hours=8))
    with np.load(aux, allow_pickle=False) as z:
        g = np.stack([z[k] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
    if g.shape != (3, 256, 256) or not np.isfinite(g).all():
        raise ValueError(f"invalid deterministic geometry for {rel}")
    return g


def load_support(support: Path) -> tuple[dict[str, list[dict[str, str]]], dict]:
    audit = json.loads((support / "support_audit.json").read_text(encoding="utf-8"))
    if audit.get("state") != "PASS_24_FRAME_SUPPORT_DISJOINT_U_H_V":
        raise ValueError("U/H/V support audit failed")
    if audit.get("test_or_buffer_numeric_payloads_opened") != 0 or audit.get("test_rows_selected") != 0:
        raise ValueError("support audit opened/selected test or buffer data")
    result = {}
    for split in SPLITS:
        rows = read_csv(support / f"{split}_sequences.csv")
        if not rows or len({r["seq_id"] for r in rows}) != len(rows):
            raise ValueError(f"empty or duplicate {split.upper()} sequence manifest")
        for r in rows:
            if len(r["data_relpaths"].split("|")) != 24:
                raise ValueError(f"{split.upper()} sequence lacks 24-frame support: {r['seq_id']}")
        result[split] = rows
    return result, audit


def station_centers(r_pack: Path) -> dict[str, tuple[int, int]]:
    centers = {}
    for row in read_csv(r_pack / "rows_u_supervised.csv"):
        rc = tuple(map(int, row["station_grid_rc"].split(",")))
        if row["station"] in centers and centers[row["station"]] != rc:
            raise ValueError(f"station center changes over U: {row['station']}")
        centers[row["station"]] = rc
    if set(centers) != set(STATIONS):
        raise ValueError(f"missing station centers: {centers}")
    return centers


def corrected_geometry(rel: str, lat: np.ndarray, lon: np.ndarray, saa: np.ndarray, compute_solar_angles):
    dt = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
    soz, soa = compute_solar_angles(lat, lon, dt)
    physical_night = soz >= 90.0
    soa = np.where(physical_night, 0.0, soa)
    return np.stack((np.cos(np.deg2rad(soz)), np.cos(np.deg2rad(soa - saa)),
                     (soz < 85.0).astype(np.float32))).astype(np.float32)


def make_arrays(out: Path, n: int):
    specs = {
        "agri_raw.npy": (n, 2, 16, 13, 3, 3),
        "geometry_raw.npy": (n, 2, 16, 2, 3, 3),
        "cot_simvp_r.npy": (n, 2, 16, 1, 16, 16),
        "cot_p_c.npy": (n, 2, 16, 1, 16, 16),
        "cot_p_xc.npy": (n, 2, 16, 1, 16, 16),
        "cot_last.npy": (n, 2, 16, 1, 16, 16),
    }
    arrays = {}
    for name, shape in specs.items():
        p = out / name
        if p.exists():
            a = np.load(p, mmap_mode="r+", allow_pickle=False)
            if a.dtype != np.float32 or a.shape != shape:
                raise ValueError(f"partial array schema mismatch: {p}")
        else:
            a = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=shape)
        arrays[name] = a
    return arrays


def station_major_geometry(geo: np.ndarray, batch_size: int) -> np.ndarray:
    """P contract: [sequence,station,lead,3,16,16] -> [sequence*station,lead,3,16,16]."""
    expected = (batch_size, 2, 16, 3, 16, 16)
    if geo.shape != expected or not np.isfinite(geo).all():
        raise ValueError(f"future geometry contract mismatch: {geo.shape} != {expected}")
    return np.ascontiguousarray(geo.reshape(batch_size * 2, 16, 3, 16, 16))


def p_output_to_sequence_station(pred: np.ndarray, batch_size: int) -> np.ndarray:
    """Convert P output [sequence*station,1,lead,16,16] to sequence-bank order."""
    expected = (batch_size * 2, 1, 16, 16, 16)
    if pred.shape != expected or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError(f"P forecast contract mismatch: {pred.shape} != {expected}")
    return np.ascontiguousarray(pred[:, 0].reshape(batch_size, 2, 16, 16, 16)[:, :, :, None])


def normalize_history_agri(agri: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Match P training: channel z-score, then replace nonfinite slots with zero."""
    out = (np.asarray(agri, np.float32) - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def infer_split(split: str, rows: list[dict[str, str]], args, models, arrays, progress_path: Path):
    import torch
    simvp, rmodel, p_c, p_xc = models
    n = len(rows)
    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("state") != "BUILDING" or progress.get("split") != split or progress.get("contract_sha256") != args.contract_sha:
            raise ValueError(f"{split.upper()} partial output has another contract")
        start = int(progress["sequences_complete"])
    atomic_json(progress_path, {"state": "BUILDING", "split": split,
                               "sequences_complete": start, "sequences_total": n,
                               "contract_sha256": args.contract_sha,
                               "future_observed_agri_read": False, "cpp_read": False,
                               "ghi_labels_read": False})
    cache = FrameLRU(args.load_frame, args.data_root, args.frame_cache)
    geometry_cache = ArrayLRU(128)
    hmean = args.agri_mean[None, :, None, None]
    hstd = args.agri_std[None, :, None, None]
    rmean, rstd = args.r_mean, args.r_std
    p_norm_c, p_norm_xc = args.p_norm_c, args.p_norm_xc
    lat, lon, saa = args.lat, args.lon, args.saa

    for lo in range(start, n, args.batch_sequences):
        hi = min(n, lo + args.batch_sequences)
        batch_rows = rows[lo:hi]
        b = len(batch_rows)
        sim_x, sim_g = [], []
        hist_p = []
        future_geo_arr = np.empty((b, 2, 16, 3, 16, 16), np.float32)
        agri_patches = np.empty((b, 2, 16, 13, 3, 3), np.float32)
        geo_patches = np.empty((b, 2, 16, 2, 3, 3), np.float32)
        hist_r_inputs, hist_info = [], []
        fut_r_inputs = []
        for bi, row in enumerate(batch_rows):
            paths = row["data_relpaths"].split("|")
            xhist, ghist, p_station_inputs = [], [], []
            for rel in paths[:8]:
                image, valid, geom = cache.get(rel)
                normalized = np.where(valid, (image - hmean[0]) / hstd[0], 0.0).astype(np.float32)
                xhist.append(normalized); ghist.append(geom)
                geom_corr = geometry_cache.get(("corrected", rel),
                    lambda rel=rel: corrected_geometry(rel, lat, lon, saa, args.compute_solar_angles))
                for si, station in enumerate(STATIONS):
                    rr, cc = args.centers[station]; r0, c0 = rr - 8, cc - 8
                    sl = np.s_[r0:r0 + 16, c0:c0 + 16]
                    aa = image[:, sl[0], sl[1]].astype(np.float32, copy=False)
                    vv = valid[:, sl[0], sl[1]]
                    gg = geom_corr[:, sl[0], sl[1]]
                    inp = (np.concatenate((aa, gg), axis=0) - rmean) / rstd
                    inp[~np.isfinite(inp)] = 0.0
                    inp[:13, ~vv.all(axis=0)] = 0.0
                    hist_r_inputs.append(inp.astype(np.float32))
                    hist_info.append((bi, si, len(xhist) - 1, gg, vv))
                    p_station_inputs.append((aa, vv, gg))
            for ti, rel in enumerate(paths[8:]):
                gf = geometry_cache.get(("sidecar", rel),
                    lambda rel=rel: load_geometry(args.resolve_aux, args.data_root, rel))
                # S_U receives the same deterministic sidecars as its original inference path.
                ghist.append(gf)
                geom_corr = geometry_cache.get(("corrected", rel),
                    lambda rel=rel: corrected_geometry(rel, lat, lon, saa, args.compute_solar_angles))
                for si, station in enumerate(STATIONS):
                    rr, cc = args.centers[station]; r0, c0 = rr - 8, cc - 8
                    sl = np.s_[r0:r0 + 16, c0:c0 + 16]
                    gg = geom_corr[:, sl[0], sl[1]]
                    future_geo_arr[bi, si, ti] = gg
                    geo_patches[bi, si, ti] = gg[:2, 7:10, 7:10]
            sim_x.append(np.stack(xhist)); sim_g.append(np.stack(ghist))
            hist_p.append(p_station_inputs)

        # R_U on observed history AGRI supplies the only historical COT feature.
        with torch.inference_mode():
            r_hist = np.empty((len(hist_r_inputs), 1, 16, 16), np.float32)
            for rlo in range(0, len(hist_r_inputs), args.r_batch):
                tin = torch.from_numpy(np.stack(hist_r_inputs[rlo:rlo + args.r_batch])).cuda(non_blocking=True)
                r_hist[rlo:rlo + len(tin)] = rmodel(tin).float().cpu().numpy()
        cot_hist = np.zeros((b, 2, 8, 1, 16, 16), np.float32)
        valid_hist = np.zeros_like(cot_hist)
        qa_hist = np.zeros((b, 2, 8, 13, 16, 16), np.float32)
        ag_hist = np.zeros_like(qa_hist)
        geo_hist_corr = np.zeros((b, 2, 8, 3, 16, 16), np.float32)
        for ix, (bi, si, ti, gg, vv) in enumerate(hist_info):
            cot_hist[bi, si, ti] = np.expm1(np.maximum(r_hist[ix], 0.0)).astype(np.float32)
            support = (gg[2:3] > 0.5) & (gg[0:1] > SOZ_COS_80) & vv.all(axis=0, keepdims=True)
            valid_hist[bi, si, ti] = support.astype(np.float32)
            aa, q, _ = hist_p[bi][ti * 2 + si]
            ag_hist[bi, si, ti] = aa
            qa_hist[bi, si, ti] = q.astype(np.float32)
            geo_hist_corr[bi, si, ti] = gg

        r_hist_log1p = np.zeros_like(cot_hist)
        for ix, (bi, si, ti, _gg, _vv) in enumerate(hist_info):
            r_hist_log1p[bi, si, ti] = r_hist[ix]
        p_inputs = []
        for bi in range(b):
            for si in range(2):
                # Match P training: normalize the frozen R_U log1p output directly.
                c_norm_c = (r_hist_log1p[bi, si] - p_norm_c["mean"]) / p_norm_c["std"]
                c_norm_xc = (r_hist_log1p[bi, si] - p_norm_xc["mean"]) / p_norm_xc["std"]
                c_norm_c = np.where(valid_hist[bi, si] > 0.5, c_norm_c, 0.0).astype(np.float32)
                c_norm_xc = np.where(valid_hist[bi, si] > 0.5, c_norm_xc, 0.0).astype(np.float32)
                blocks = []
                for norm_cot in (c_norm_c, c_norm_xc):
                    blocks.append(np.concatenate((
                        normalize_history_agri(ag_hist[bi, si], hmean[0], hstd[0]),
                        qa_hist[bi, si], norm_cot, valid_hist[bi, si], geo_hist_corr[bi, si]
                    ), axis=1).astype(np.float32))
                p_inputs.append(blocks)
        x_c = np.stack([p_inputs[i][0] for i in range(len(p_inputs))])
        x_xc = np.stack([p_inputs[i][1] for i in range(len(p_inputs))])
        x_c[:, :, :13] = 0.0
        future_geo_tensor = station_major_geometry(future_geo_arr, b)

        # Fixed S_U predicts a common full-domain AGRI field from observed history only.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            forecast_n = simvp(torch.from_numpy(np.stack(sim_x)).cuda(non_blocking=True),
                               torch.from_numpy(np.stack(sim_g)).cuda(non_blocking=True)).float().cpu().numpy()
        forecast = forecast_n * args.agri_std[None, None, :, None, None] + args.agri_mean[None, None, :, None, None]
        if forecast.shape != (b, 16, 13, 256, 256) or not np.isfinite(forecast).all():
            raise ValueError(f"bad S_U forecast shape/value: {forecast.shape}")
        cot_simvp = np.empty((b, 2, 16, 1, 16, 16), np.float32)
        for bi in range(b):
            for si, station in enumerate(STATIONS):
                rr, cc = args.centers[station]; r0, c0 = rr - 8, cc - 8
                sl = np.s_[r0:r0 + 16, c0:c0 + 16]
                for ti in range(16):
                    pa = forecast[bi, ti, :, sl[0], sl[1]]
                    agri_patches[bi, si, ti] = pa[:, 7:10, 7:10]
                    pg = future_geo_arr[bi, si, ti]
                    inp = (np.concatenate((pa, pg), axis=0) - rmean) / rstd
                    inp[~np.isfinite(inp)] = 0.0
                    fut_r_inputs.append(inp.astype(np.float32))
        simvp_r_inputs = fut_r_inputs
        simvp_r = np.empty((len(simvp_r_inputs), 1, 16, 16), np.float32)
        with torch.inference_mode():
            for rlo in range(0, len(simvp_r_inputs), args.r_batch):
                tin = torch.from_numpy(np.stack(simvp_r_inputs[rlo:rlo + args.r_batch])).cuda(non_blocking=True)
                simvp_r[rlo:rlo + len(tin)] = rmodel(tin).float().cpu().numpy()
        cot_simvp = np.expm1(np.maximum(simvp_r.reshape(b, 2, 16, 1, 16, 16), 0.0)).astype(np.float32)
        x_c_t = torch.from_numpy(x_c).cuda(non_blocking=True)
        x_xc_t = torch.from_numpy(x_xc).cuda(non_blocking=True)
        fg_t = torch.from_numpy(future_geo_tensor).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_c_raw = p_c(x_c_t, fg_t).float().cpu().numpy()
            pred_xc_raw = p_xc(x_xc_t, fg_t).float().cpu().numpy()
        pred_c = p_output_to_sequence_station(pred_c_raw, b)
        pred_xc = p_output_to_sequence_station(pred_xc_raw, b)
        cot_last = np.repeat(cot_hist[:, :, -1:, :, :, :], 16, axis=2)
        arrays["agri_raw.npy"][lo:hi] = agri_patches
        arrays["geometry_raw.npy"][lo:hi] = geo_patches
        arrays["cot_simvp_r.npy"][lo:hi] = cot_simvp
        arrays["cot_p_c.npy"][lo:hi] = pred_c
        arrays["cot_p_xc.npy"][lo:hi] = pred_xc
        arrays["cot_last.npy"][lo:hi] = cot_last
        for a in arrays.values():
            a.flush()
        atomic_json(progress_path, {"state": "BUILDING", "split": split,
                                   "sequences_complete": hi, "sequences_total": n,
                                   "contract_sha256": args.contract_sha,
                                   "future_observed_agri_read": False, "cpp_read": False,
                                   "ghi_labels_read": False})
        print("DIRECT_COT_HV_PROGRESS", json.dumps({"split": split, "sequences": hi,
              "total": n, "batch_seconds": time.monotonic() - args.batch_started}), flush=True)
        args.batch_started = time.monotonic()
    atomic_json(progress_path, {"state": "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE", "split": split,
                               "sequences_complete": n, "sequences_total": n,
                               "contract_sha256": args.contract_sha,
                               "future_observed_agri_read": False, "cpp_read": False,
                               "ghi_labels_read": False})


def main() -> None:
    import torch
    p = argparse.ArgumentParser()
    for name in ("support", "simvp-source", "simvp-run", "simvp-manifest", "r-pack", "r-run",
                 "p-run", "data-root", "grid-static", "geometry-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch-sequences", type=int, default=4)
    p.add_argument("--r-batch", type=int, default=256)
    p.add_argument("--frame-cache", type=int, default=256)
    a = p.parse_args()
    if a.output.exists() and not (a.output / "progress_h.json").exists():
        raise FileExistsError(a.output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this job requires exactly one PBS-visible GPU")
    support_rows, support_audit = load_support(a.support)
    centers = station_centers(a.r_pack)
    source_rows = read_csv(a.support / "u_sequences.csv")
    if len({r["seq_id"] for r in source_rows}) != len(source_rows):
        raise ValueError("duplicate U sequence ids")
    sstatus = json.loads((a.simvp_run / "training" / "training_complete.json").read_text())
    simvp_ckpt = a.simvp_run / "training" / "epoch_012.pt"
    simvp_norm_path = a.simvp_manifest / "hunan_train_norm.json"
    if sstatus.get("selected_checkpoint") != "epoch_012.pt" or sstatus.get("heldout_payload_read") is not False:
        raise ValueError("S_U fixed epoch-12 provenance failed")
    rstatus = json.loads((a.r_run / "status.json").read_text())
    if rstatus.get("state") != "COMPLETE_R_U_FIXED_EPOCH_50":
        raise ValueError("R_U fixed-epoch completion receipt missing")
    pcomplete = json.loads((a.p_run / "paired_training_complete.json").read_text())
    if pcomplete.get("state") != "COMPLETE_P_C_AND_P_XC_FIXED_EPOCH_12" or pcomplete.get("test_payloads_read") != 0 or pcomplete.get("buffer_payloads_read") != 0:
        raise ValueError("P-C/P-XC fixed-budget receipt or isolation gate failed")
    if a.p_run.name != "predictor_u_v1":
        raise ValueError("P output path is not the planned U-only predictor")
    norm = json.loads(simvp_norm_path.read_text())
    amean, astd = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
    if amean.shape != (13,) or astd.shape != (13,) or (astd <= 0).any():
        raise ValueError("invalid S_U normalization")

    r_script = Path(__file__).parent / "train_r_u.py"
    r_ckpt = a.r_run / "epoch_050.pt"
    r_norm_path = a.r_run / "normalization.json"
    rmeta = json.loads((a.r_run / "run_metadata.json").read_text())
    if rstatus.get("checkpoint_sha256") != sha(r_ckpt) or rstatus.get("normalization_sha256") != sha(r_norm_path):
        raise ValueError("R_U checkpoint/normalization hash mismatch")
    r_pack_audit = validate_r_geometry_provenance(a.r_pack, a.geometry_source)
    if rmeta.get("geometry") != "corrected EOT radians-to-degrees; frozen Hunan grid_static":
        raise ValueError("R_U run metadata geometry description mismatch")
    p_meta_c = json.loads((a.p_run / "P-C" / "run_metadata.json").read_text())
    p_meta_x = json.loads((a.p_run / "P-XC" / "run_metadata.json").read_text())
    p_ckpt_c = a.p_run / "P-C" / "epoch_012.pt"
    p_ckpt_x = a.p_run / "P-XC" / "epoch_012.pt"
    if sha(p_ckpt_c) != pcomplete["P-C"]["checkpoint_sha256"] or sha(p_ckpt_x) != pcomplete["P-XC"]["checkpoint_sha256"]:
        raise ValueError("P checkpoint hash mismatch")
    if p_meta_c.get("p_code_sha256") != p_meta_x.get("p_code_sha256"):
        raise ValueError("paired P code hash mismatch")

    sys.path.insert(0, str(a.simvp_source))
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(a.geometry_source.parent))
    from hunan_data import load_frame, resolve_aux
    from triad_models import SimVPBackbone
    from train_r_u import COTUNet
    from train_future_cot_u import DirectCotForecaster
    from solar_geometry_corrected import compute_solar_angles

    simvp_payload = torch.load(simvp_ckpt, map_location="cpu", weights_only=False)
    simvp = SimVPBackbone(str(a.simvp_source)).cuda().eval()
    simvp.load_state_dict(simvp_payload["model"], strict=True)
    del simvp_payload
    rpayload = torch.load(r_ckpt, map_location="cpu", weights_only=False)
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(rpayload["model"], strict=True)
    del rpayload
    pmods = []
    for mode, ckpt in (("P-C", p_ckpt_c), ("P-XC", p_ckpt_x)):
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if payload.get("epoch") != 12 or payload.get("metadata", {}).get("candidate") != mode:
            raise ValueError(f"bad fixed checkpoint metadata for {mode}")
        model = DirectCotForecaster().cuda().eval()
        model.load_state_dict(payload["model"], strict=True)
        pmods.append(model)
    del payload
    with np.load(a.grid_static, allow_pickle=False) as z:
        lat, lon, saa = (np.asarray(z[k], np.float32) for k in ("lat", "lon", "saa"))
    if any(q.shape != (256, 256) for q in (lat, lon, saa)):
        raise ValueError("invalid Hunan grid geometry")
    rnorm = json.loads(r_norm_path.read_text())
    rmean = np.asarray(rnorm["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or (rstd <= 0).any():
        raise ValueError("invalid R_U input normalization")
    p_norm_c = p_meta_c["history_cot_log1p_normalization"]
    p_norm_xc = p_meta_x["history_cot_log1p_normalization"]
    if p_norm_c["fit_partition"] != "U" or p_norm_xc["fit_partition"] != "U":
        raise ValueError("P history COT normalization is not U-only")
    args = argparse.Namespace(**vars(a))
    args.agri_mean, args.agri_std = amean, astd
    args.r_mean, args.r_std = rmean, rstd
    args.p_norm_c, args.p_norm_xc = p_norm_c, p_norm_xc
    args.centers = centers
    args.load_frame, args.resolve_aux = load_frame, resolve_aux
    args.compute_solar_angles = compute_solar_angles
    args.lat, args.lon, args.saa = lat, lon, saa
    source_script = Path(__file__)
    contract = {
        "support_audit_sha256": sha(a.support / "support_audit.json"),
        "h_support_sha256": sha(a.support / "h_sequences.csv"),
        "v_support_sha256": sha(a.support / "v_sequences.csv"),
        "simvp_checkpoint_sha256": sha(simvp_ckpt), "simvp_norm_sha256": sha(simvp_norm_path),
        "r_checkpoint_sha256": sha(r_ckpt), "r_norm_sha256": sha(r_norm_path),
        "r_pack_audit_sha256": sha(a.r_pack / "pack_audit.json"),
        "geometry_source_sha256": sha(a.geometry_source),
        "p_c_checkpoint_sha256": sha(p_ckpt_c), "p_xc_checkpoint_sha256": sha(p_ckpt_x),
        "predictor_source_sha256": sha(source_script), "test_and_buffer_payloads_opened": False,
    }
    args.contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.batch_started = time.monotonic()
    if not a.output.exists():
        a.output.mkdir(parents=True)
    for split in SPLITS:
        out = a.output / split.upper()
        out.mkdir(parents=True, exist_ok=True)
        rows = support_rows[split]
        arrays = make_arrays(out, len(rows))
        if not (out / "index.csv").exists():
            with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=("sequence_index_within_split", "seq_id", "BJT_start"), lineterminator="\n")
                w.writeheader()
                w.writerows({"sequence_index_within_split": i, "seq_id": r["seq_id"], "BJT_start": r["BJT_start"]} for i, r in enumerate(rows))
        infer_split(split, rows, args, (simvp, rmodel, *pmods), arrays, a.output / f"progress_{split}.json")
        for arr in arrays.values(): arr.flush()
    artifacts = {}
    for split in SPLITS:
        d = a.output / split.upper()
        for pth in sorted(d.iterdir()):
            if pth.is_file() and pth.suffix == ".npy":
                artifacts[f"{split.upper()}/{pth.name}"] = sha(pth)
        if json.loads((a.output / f"progress_{split}.json").read_text()).get("state") != "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE":
            raise RuntimeError(f"{split.upper()} inference incomplete")
    manifest = {
        "state": "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE", "stations": list(STATIONS),
        "sequence_counts": {s.upper(): len(support_rows[s]) for s in SPLITS},
        "arms": ["simvp_r", "p_c", "p_xc", "last"], "prediction_shape": "per sequence, station, +15..+240 min; AGRI 3x3 and COT 16x16",
        "inference_inputs": "8 observed AGRI history frames; history R_U COT; known deterministic geometry; no target labels",
        "future_observed_agri_read": False, "cpp_read": False, "ghi_labels_read": False,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "split_support_audit_sha256": contract["support_audit_sha256"],
        "normalization_fit_partition": "U", "contract": contract,
        "contract_sha256": args.contract_sha, "artifact_sha256": artifacts,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
    }
    atomic_json(a.output / "inference_manifest.json", manifest)
    print("DIRECT_COT_HV_INFERENCE_COMPLETE", json.dumps({"sequences": manifest["sequence_counts"],
          "future_observed_agri_read": False, "cpp_read": False, "ghi_labels_read": False}), flush=True)


if __name__ == "__main__":
    main()
