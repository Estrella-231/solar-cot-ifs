#!/usr/bin/env python3
"""CPU-reload all A/AC/AP checkpoints and compare against saved A800 outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from head_candidate_v0 import LocalGHIHead


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    meta_path = a.pack / "pack_manifest.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("state") != "COMPLETE" or meta.get("test_payload_opened") is not False:
        raise ValueError("pack completion/test gate failed")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8")))
    split = np.asarray([r["split"] for r in rows])
    station = np.asarray([r["station"] for r in rows])
    ghi = np.asarray([float(r["ghi"]) for r in rows], dtype=np.float32)
    clear = np.asarray([float(r["clear"]) for r in rows], dtype=np.float32)
    ag = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(a.pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    geo = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    am = np.asarray(meta["AGRI_norm_mean"], dtype=np.float32)[None, :, None, None]
    ast = np.asarray(meta["AGRI_norm_std"], dtype=np.float32)[None, :, None, None]
    cm, cs = np.float32(meta["COT_norm_mean"]), np.float32(meta["COT_norm_std"])
    x = np.where(valid, (np.asarray(ag) - am) / ast, 0).astype(np.float32)
    g = np.array(geo, dtype=np.float32, copy=True)
    c = ((np.asarray(cot, dtype=np.float32) - cm) / cs).astype(np.float32)
    result = {"state": "PASS_ALL_CHECKPOINTS_CPU_RELOAD", "device": "CPU", "rows_sha256": sha256(a.pack / "rows.csv"),
              "pack_manifest_sha256": sha256(meta_path), "stations": {}, "test_payload_opened": False}
    for st in ("sili", "zhujia"):
        ids = np.flatnonzero((split == "val") & (station == st))
        if not len(ids):
            raise ValueError(f"empty validation rows for {st}")
        result["stations"][st] = {"n": int(len(ids)), "arms": {}}
        tx = torch.from_numpy(np.array(x[ids], copy=True))
        tg = torch.from_numpy(np.array(g[ids], copy=True))
        tc = torch.from_numpy(np.array(c[ids], copy=True))
        tclear = torch.from_numpy(clear[ids])
        y = ghi[ids].astype(np.float64)
        for seed in (42, 43, 44):
            sd = a.results / f"ghi_heads_new_r_seed{seed}_v1" / st
            run = json.loads((sd / "run_metadata.json").read_text())
            if run.get("seed") != seed or run.get("station") != st or run.get("test_payload_opened") is not False:
                raise ValueError(f"run metadata mismatch {seed}/{st}")
            for arm in ("A", "AC", "AP"):
                ckpt_path = sd / f"{arm}_best.pt"
                saved_path = sd / f"{arm}_val_prediction_ghi_wm2.npy"
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if ckpt.get("arm") != arm or ckpt.get("seed") != seed:
                    raise ValueError(f"checkpoint identity mismatch {seed}/{st}/{arm}")
                model = LocalGHIHead(arm, 2).cpu()
                model.load_state_dict(ckpt["model"], strict=True)
                model.eval()
                pred = []
                with torch.inference_mode():
                    for start in range(0, len(ids), a.batch_size):
                        end = min(start + a.batch_size, len(ids))
                        cot_input = tc[start:end] if arm == "AC" else None
                        pred.append((model(tx[start:end], tg[start:end], cot_input) * tclear[start:end]).numpy())
                reloaded = np.concatenate(pred).astype(np.float64)
                saved = np.asarray(np.load(saved_path, allow_pickle=False), dtype=np.float64).reshape(-1)
                if len(saved) != len(y) or not np.isfinite(reloaded).all() or not np.isfinite(saved).all():
                    raise ValueError(f"prediction shape/finite failure {seed}/{st}/{arm}")
                diff = reloaded - saved
                entry = {
                    "seed": seed,
                    "max_abs_reload_vs_saved_wm2": float(np.max(np.abs(diff))),
                    "mean_abs_reload_vs_saved_wm2": float(np.mean(np.abs(diff), dtype=np.float64)),
                    "saved_rmse_wm2": float(np.sqrt(np.mean((saved - y) ** 2, dtype=np.float64))),
                    "reloaded_cpu_rmse_wm2": float(np.sqrt(np.mean((reloaded - y) ** 2, dtype=np.float64))),
                    "rmse_delta_cpu_minus_saved_wm2": float(np.sqrt(np.mean((reloaded - y) ** 2, dtype=np.float64)) - np.sqrt(np.mean((saved - y) ** 2, dtype=np.float64))),
                    "checkpoint_sha256": sha256(ckpt_path),
                    "prediction_sha256": sha256(saved_path),
                }
                result["stations"][st]["arms"].setdefault(arm, {})[str(seed)] = entry
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"state": result["state"], "device": result["device"],
                      "max_point_delta_wm2": max(
                          v["max_abs_reload_vs_saved_wm2"]
                          for st in result["stations"].values() for arm in st["arms"].values() for v in arm.values()),
                      "max_abs_rmse_delta_wm2": max(
                          abs(v["rmse_delta_cpu_minus_saved_wm2"])
                          for st in result["stations"].values() for arm in st["arms"].values() for v in arm.values()),
                      "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
