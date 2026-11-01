#!/usr/bin/env python3
"""Actual-data GPU forward/backward and batch-size throughput profile."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
COS80 = float(np.cos(np.deg2rad(80.0)))


def parse_stamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def aux_for(root: Path, rel: str) -> Path:
    t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") + timedelta(hours=8)
    matches = list((root / "aux_256" / t.strftime("%Y%m")).glob(
        t.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"aux sidecar count={len(matches)} for {rel}")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite profile: {a.output}")
    sys.path.insert(0, str(a.source_dir))
    from retrieved_cot_forecaster_20261013 import RetrievedCOTForecaster, cot_supervision_masked_mse

    bank_manifest = json.loads((a.bank / "bank_manifest.json").read_text())
    if bank_manifest.get("state") != "COMPLETE" or bank_manifest.get("test_payload_opened") is not False:
        raise ValueError("bank provenance/test gate failed")
    cot = np.load(a.bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    rows = list(csv.DictReader((a.bank / "frame_keys.csv").open(encoding="utf-8")))
    by_time = {r["target_time_utc"]: i for i, r in enumerate(rows) if r["split"] == "train"}
    # Fixed known daytime training initialization: 2024-10-22 04:45 UTC.
    init = datetime(2024, 10, 22, 4, 45)
    times = [init - timedelta(minutes=15*k) for k in range(7, -1, -1)] + [init + timedelta(minutes=15*k) for k in range(1, 17)]
    keys = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
    if any(k not in by_time for k in keys):
        raise ValueError("fixed train sequence is not fully present in R bank")
    ids = [by_time[k] for k in keys]
    hx, hm, gy, ty, tm = [], [], [], [], []
    for idx in ids:
        r = rows[idx]
        rel = r["agri_frame_relpath"]
        with np.load(aux_for(a.data_root, rel), allow_pickle=False) as z:
            bits = z["agri_valid_bits"]
            geos = []
            masks = []
            physes = []
            for si, station in enumerate(("sili", "zhujia")):
                row, col = map(int, r[f"{station}_grid_rc"].split(","))
                r0, c0 = row - 8, col - 8
                sl = np.s_[r0:r0+16, c0:c0+16]
                agri_ok = np.ones((16, 16), dtype=bool)
                for ch in CHANNELS:
                    agri_ok &= (bits[sl] & (1 << ch)) != 0
                geo = np.stack([z[k][sl] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
                phys = np.expm1(np.asarray(cot[idx, si, 0], dtype=np.float64)).astype(np.float32)
                geom_finite = np.isfinite(geo).all(axis=0)
                valid = agri_ok & geom_finite & (geo[2] > 0.5) & (geo[0] > COS80) & np.isfinite(phys)
                geo = np.where(np.isfinite(geo), geo, 0.0)
                geos.append(geo); masks.append(valid[None]); physes.append(phys[None])
        gy.append(np.stack(geos))
        if len(hx) < 8:
            hx.append(np.stack([np.log1p(np.maximum(x, 0.0))/np.log(101.0) for x in physes]))
            hm.append(np.stack(masks))
        else:
            ty.append(np.stack(physes)); tm.append(np.stack(masks))

    # [station, time, ...] -> two examples, each has causal 8-history/16 targets.
    history = np.stack(hx, axis=1) if False else np.asarray(hx)  # [8, station, 1,H,W]
    hist_mask = np.asarray(hm)
    geo_all = np.asarray(gy)  # [24, station,3,H,W]
    target = np.asarray(ty)
    target_mask = np.asarray(tm)
    # Convert to [station, T, C, H, W], then build two station examples.
    history = np.stack([history[:, s] for s in range(2)], axis=0)
    hist_mask = np.stack([hist_mask[:, s] for s in range(2)], axis=0)
    geometry = np.stack([geo_all[:, s] for s in range(2)], axis=0)
    target = np.stack([target[:, s] for s in range(2)], axis=0)
    target_mask = np.stack([target_mask[:, s] for s in range(2)], axis=0)
    device = "cuda"
    history_t = torch.from_numpy(history).float().to(device)
    hist_mask_t = torch.from_numpy(hist_mask).bool().to(device)
    geometry_t = torch.from_numpy(geometry).float().to(device)
    target_t = torch.from_numpy(target).float().to(device)
    target_mask_t = torch.from_numpy(target_mask).bool().to(device)
    if not target_mask_t.any():
        raise ValueError("fixed train batch has no valid target pixels")

    model = RetrievedCOTForecaster(width=96).to(device)
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    results = []
    max_mem = torch.cuda.get_device_properties(0).total_memory
    for bs in (8, 16, 32, 64, 128, 256, 512, 1024):
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        try:
            h = history_t.repeat(bs, 1, 1, 1, 1)
            m = hist_mask_t.repeat(bs, 1, 1, 1, 1)
            g = geometry_t.repeat(bs, 1, 1, 1, 1)
            y = target_t.repeat(bs, 1, 1, 1, 1)
            ym = target_mask_t.repeat(bs, 1, 1, 1, 1)
            for _ in range(3):
                pred = model(h, m, g)
                loss = cot_supervision_masked_mse(pred, y, ym)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            steps = 8
            for _ in range(steps):
                pred = model(h, m, g)
                loss = cot_supervision_masked_mse(pred, y, ym)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            allocated = torch.cuda.max_memory_allocated()
            results.append({"batch": bs, "steps": steps, "seconds": elapsed,
                            "windows_per_second": bs*steps/elapsed,
                            "peak_allocated_gib": allocated/1024**3,
                            "memory_fraction_of_gpu": allocated/max_mem,
                            "loss_finite": bool(torch.isfinite(loss).item())})
            print(f"PROFILE batch={bs} samples_per_sec={bs*steps/elapsed:.2f} peak_gib={allocated/1024**3:.2f}", flush=True)
            del h, m, g, y, ym, pred, loss
            torch.cuda.empty_cache()
            if allocated/max_mem >= 0.82:
                break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            results.append({"batch": bs, "oom": True})
            break
    safe = [x for x in results if "windows_per_second" in x and x["memory_fraction_of_gpu"] <= 0.82]
    safe_peak = max(safe, key=lambda x: x["windows_per_second"])
    choice = max((x for x in safe if x["windows_per_second"] >= 0.95*safe_peak["windows_per_second"]), key=lambda x: x["batch"])
    out = {"state": "PASS_REAL_DATA_GPU_PROFILE", "split": "train", "init_time_utc": keys[7],
           "stations": ["sili", "zhujia"], "target_times_utc": keys[8:],
           "test_payload_opened": False, "gpu_name": torch.cuda.get_device_name(0),
           "gpu_total_gib": max_mem/1024**3, "mask_rule": "13 AGRI bits + day_mask>0.5 + cosSOZ>cos80 + finite R-COT",
           "batch_trials": results, "best_safe_throughput": safe_peak,
           "selected_batch_max_safe_within_5pct": choice,
           "criterion": "largest batch within 5% of measured peak throughput and <=82% allocated device memory"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
