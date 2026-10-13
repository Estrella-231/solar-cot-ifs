#!/usr/bin/env python3
"""Validation-only frozen-head COT input interventions (no retraining)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
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


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2, dtype=np.float64)))


def bjt_month(utc_text: str) -> str:
    dt = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m")


def predict(model, x, g, c, clear, batch_size):
    out = []
    with torch.inference_mode():
        for st in range(0, len(x), batch_size):
            sl = slice(st, min(st + batch_size, len(x)))
            out.append((model(x[sl], g[sl], c[sl]) * clear[sl]).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261013)
    a = p.parse_args()
    torch.set_num_threads(1)
    meta_path = a.pack / "pack_manifest.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("state") != "COMPLETE" or meta.get("test_payload_opened") is not False:
        raise ValueError("pack completion/test gate failed")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8")))
    split = np.asarray([r["split"] for r in rows])
    stations = np.asarray([r["station"] for r in rows])
    keys = np.asarray([r["target_time_utc"] for r in rows])
    months = np.asarray([bjt_month(x) for x in keys])
    y_all = np.asarray([float(r["ghi"]) for r in rows], np.float64)
    ag = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(a.pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    geo = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    am = np.asarray(meta["AGRI_norm_mean"], np.float32)[None, :, None, None]
    ast = np.asarray(meta["AGRI_norm_std"], np.float32)[None, :, None, None]
    cm, cs = np.float32(meta["COT_norm_mean"]), np.float32(meta["COT_norm_std"])
    x_all = np.where(valid, (np.asarray(ag) - am) / ast, 0).astype(np.float32)
    g_all = np.array(geo, dtype=np.float32, copy=True)
    c_all = ((np.asarray(cot, dtype=np.float32) - cm) / cs).astype(np.float32)
    zero_norm = np.float32((0.0 - float(cm)) / float(cs))
    neighbor_positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    permutation = np.random.default_rng(a.seed).permutation(len(neighbor_positions))
    result = {
        "state": "PASS_VALIDATION_FROZEN_AC_COT_INTERVENTIONS",
        "split_id": meta["split_id"],
        "pack_manifest_sha256": sha256(meta_path),
        "rows_sha256": sha256(a.pack / "rows.csv"),
        "interventions": {
            "physical_zero": "set all COT pixels to the normalized value corresponding to physical COT=0",
            "center_only": "keep center COT; replace eight neighbors with physical COT=0",
            "neighbor_permutation": "keep center and permute the same eight neighbors within each 3x3 map",
            "same_station_month_mismatch": "replace each COT map with a different validation target from the same station and BJT month using a deterministic cyclic shift",
        },
        "per_seed_station": {},
        "test_payload_opened": False,
        "claim_scope": "frozen-checkpoint validation input-dependence diagnostics; perturbations can shift input distribution and do not establish physical causality",
    }
    for seed in (42, 43, 44):
        seed_result = {}
        for station in ("sili", "zhujia"):
            ids = np.flatnonzero((split == "val") & (stations == station))
            y = y_all[ids]
            tx = torch.from_numpy(np.array(x_all[ids], copy=True))
            tg = torch.from_numpy(np.array(g_all[ids], copy=True))
            tc = torch.from_numpy(np.array(c_all[ids], copy=True))
            clear = torch.from_numpy(np.asarray([float(rows[i]["clear"]) for i in ids], np.float32))
            model_dir = a.results / f"ghi_heads_new_r_seed{seed}_v1" / station
            checkpoint = model_dir / "AC_best.pt"
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if ckpt.get("arm") != "AC" or ckpt.get("seed") != seed:
                raise ValueError(f"checkpoint mismatch {seed}/{station}")
            model = LocalGHIHead("AC", 2).cpu()
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()
            base = predict(model, tx, tg, tc, clear, a.batch_size)
            saved = np.asarray(np.load(model_dir / "AC_val_prediction_ghi_wm2.npy", allow_pickle=False), np.float64).reshape(-1)
            if len(base) != len(saved) or np.max(np.abs(base - saved)) > 1.0:
                raise ValueError(f"checkpoint/prediction mismatch {seed}/{station}")

            alternatives = {}
            c_zero = torch.full_like(tc, float(zero_norm))
            alternatives["physical_zero"] = predict(model, tx, tg, c_zero, clear, a.batch_size)
            c_center = c_zero.clone()
            c_center[:, :, 1, 1] = tc[:, :, 1, 1]
            alternatives["center_only"] = predict(model, tx, tg, c_center, clear, a.batch_size)
            c_perm = tc.clone()
            original = tc.clone()
            for dst_i, (r, col) in enumerate(neighbor_positions):
                src_r, src_c = neighbor_positions[int(permutation[dst_i])]
                c_perm[:, :, r, col] = original[:, :, src_r, src_c]
            alternatives["neighbor_permutation"] = predict(model, tx, tg, c_perm, clear, a.batch_size)

            c_mismatch = np.array(c_all[ids], copy=True)
            groups: dict[str, list[int]] = {}
            for local_i, global_i in enumerate(ids):
                groups.setdefault(months[global_i], []).append(local_i)
            rng = np.random.default_rng(a.seed + seed * 101 + (0 if station == "sili" else 1))
            for month, local_ids in groups.items():
                if len(local_ids) < 2:
                    raise ValueError(f"cannot mismatch within validation month: {station}/{month}")
                shift = int(rng.integers(1, len(local_ids)))
                donor_ids = local_ids[shift:] + local_ids[:shift]
                c_mismatch[np.asarray(local_ids)] = c_all[ids[np.asarray(donor_ids)]]
            alternatives["same_station_month_mismatch"] = predict(model, tx, tg, torch.from_numpy(c_mismatch), clear, a.batch_size)

            base_rmse = rmse(y, saved)
            seed_result[station] = {
                "n": int(len(ids)),
                "checkpoint_sha256": sha256(checkpoint),
                "saved_prediction_sha256": sha256(model_dir / "AC_val_prediction_ghi_wm2.npy"),
                "baseline_saved_rmse_wm2": base_rmse,
                "checkpoint_reload_max_abs_vs_saved_wm2": float(np.max(np.abs(base - saved))),
                "alternatives": {
                    name: {
                        "rmse_wm2": rmse(y, pred),
                        "delta_vs_baseline_wm2": rmse(y, pred) - base_rmse,
                        "mean_abs_prediction_change_wm2": float(np.mean(np.abs(pred - saved), dtype=np.float64)),
                    } for name, pred in alternatives.items()
                },
            }
        result["per_seed_station"][str(seed)] = seed_result

    summary = {}
    for intervention in ("physical_zero", "center_only", "neighbor_permutation", "same_station_month_mismatch"):
        deltas = [result["per_seed_station"][str(seed)][station]["alternatives"][intervention]["delta_vs_baseline_wm2"]
                  for seed in (42, 43, 44) for station in ("sili", "zhujia")]
        changes = [result["per_seed_station"][str(seed)][station]["alternatives"][intervention]["mean_abs_prediction_change_wm2"]
                   for seed in (42, 43, 44) for station in ("sili", "zhujia")]
        summary[intervention] = {
            "mean_station_seed_delta_rmse_wm2": float(np.mean(deltas)),
            "range_station_seed_delta_rmse_wm2": [float(np.min(deltas)), float(np.max(deltas))],
            "mean_of_station_seed_mean_abs_prediction_change_wm2": float(np.mean(changes)),
        }
    result["summary"] = summary
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"state": result["state"], "summary": summary, "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
