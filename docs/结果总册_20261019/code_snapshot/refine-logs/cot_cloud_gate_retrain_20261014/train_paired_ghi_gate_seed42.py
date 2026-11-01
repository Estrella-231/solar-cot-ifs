#!/usr/bin/env python3
"""Paired raw-COT vs conservative-gate AC heads, seed 42, two stations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state_sha(state) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        val = state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(val.dtype).encode()); h.update(val.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for k in ("raw-pack", "gate-pack", "pair-audit", "head-source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2000)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    pair_audit = json.loads(a.pair_audit.read_text())
    if pair_audit.get("state") != "PASS_PAIRED_RAW_GATE_PACKS" or pair_audit.get("test_payloads_read") != 0:
        raise ValueError("paired-pack audit gate failed")
    mod_spec = importlib.util.spec_from_file_location("frozen_ghi_heads", a.head_source)
    if mod_spec is None or mod_spec.loader is None:
        raise RuntimeError("cannot import frozen LocalGHIHead source")
    mod = importlib.util.module_from_spec(mod_spec); mod_spec.loader.exec_module(mod)
    LocalGHIHead = mod.LocalGHIHead
    counts = mod.parameter_counts(2)
    if counts["AC"] != 24489:
        raise ValueError(f"unexpected LocalGHIHead(AC) parameter count: {counts}")

    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one scheduler-assigned CUDA device is required")
    device = torch.device("cuda:0")

    packs = {}
    for name, path in (("raw", a.raw_pack), ("gate", a.gate_pack)):
        meta = json.loads((path / "pack_manifest.json").read_text())
        rows = list(csv.DictReader((path / "rows.csv").open(newline="", encoding="utf-8")))
        if meta.get("state") != "COMPLETE" or meta.get("test_payload_opened") is not False:
            raise ValueError(f"{name} pack is incomplete or opened test")
        if len(rows) != pair_audit["rows"]:
            raise ValueError(f"{name} rows disagree with pair audit")
        keys = [(r["station"], r["target_time_utc"], r["split"]) for r in rows]
        arrays = {field: np.load(path / f"{field}.npy", mmap_mode="r", allow_pickle=False)
                  for field in ("agri_raw", "agri_valid", "geometry", "cot_log1p")}
        packs[name] = {"meta": meta, "rows": rows, "keys": keys, "arrays": arrays, "path": path}
    if packs["raw"]["keys"] != packs["gate"]["keys"]:
        raise ValueError("raw/gate key order differs despite paired audit")
    for name in ("ghi", "clear", "kt"):
        if [r[name] for r in packs["raw"]["rows"]] != [r[name] for r in packs["gate"]["rows"]]:
            raise ValueError(f"label {name} differs")

    a.output.mkdir(parents=True)
    (a.output / "pair_training_contract.json").write_text(json.dumps({
        "state": "TRAINING_SEED42_PAIRED_RAW_VS_GATE",
        "pair_audit_sha256": sha_file(a.pair_audit),
        "raw_pack_manifest_sha256": sha_file(a.raw_pack / "pack_manifest.json"),
        "gate_pack_manifest_sha256": sha_file(a.gate_pack / "pack_manifest.json"),
        "head_source_sha256": sha_file(a.head_source), "seed": a.seed,
        "epochs": a.epochs, "batch_size": a.batch_size, "parameter_counts": counts,
        "test_payload_opened": False,
    }, indent=2) + "\n")

    rng = np.random.default_rng(a.seed)
    all_results = {}
    for station in ("sili", "zhujia"):
        rows = packs["raw"]["rows"]
        train_i = np.asarray([i for i, r in enumerate(rows) if r["station"] == station and r["split"] == "train"], np.int64)
        val_i = np.asarray([i for i, r in enumerate(rows) if r["station"] == station and r["split"] == "val"], np.int64)
        if not len(train_i) or not len(val_i):
            raise ValueError(f"empty {station} split")
        orders = [rng.permutation(train_i) for _ in range(a.epochs)]
        order_hash = hashlib.sha256(b"".join(x.tobytes() for x in orders)).hexdigest()
        gh = np.asarray([float(r["ghi"]) for r in rows], np.float32)
        clear = np.asarray([float(r["clear"]) for r in rows], np.float32)
        tensors = {}
        for arm, pack in packs.items():
            meta, arrays = pack["meta"], pack["arrays"]
            am = np.asarray(meta["AGRI_norm_mean"], np.float32)[None, :, None, None]
            ast = np.asarray(meta["AGRI_norm_std"], np.float32)[None, :, None, None]
            x = np.where(np.asarray(arrays["agri_valid"], bool),
                         (np.asarray(arrays["agri_raw"], np.float32) - am) / ast, 0).astype(np.float32)
            g = np.asarray(arrays["geometry"], np.float32).copy()
            cm, cs = np.float32(meta["COT_norm_mean"]), np.float32(meta["COT_norm_std"])
            if not np.isfinite(cm) or not np.isfinite(cs) or cs <= 0:
                raise ValueError(f"invalid {arm} train-only COT normalization")
            c = ((np.asarray(arrays["cot_log1p"], np.float32) - cm) / cs).astype(np.float32)
            tensors[arm] = tuple(torch.from_numpy(z).to(device) for z in (x, g, c))
        GH = torch.from_numpy(gh).to(device); CL = torch.from_numpy(clear).to(device)

        # Match the registered AC initialization: initialize common A first,
        # copy its AGRI trunk/readout into AC, and retain the seed-42 aux init.
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        base = LocalGHIHead("A", 2).to(device)
        base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        init_states = {}
        for arm in ("raw", "gate"):
            torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
            model = LocalGHIHead("AC", 2).to(device)
            with torch.no_grad():
                model.agri_encoder.load_state_dict(base.agri_encoder.state_dict())
                model.readout[0].weight.zero_()
                model.readout[0].weight[:, :base.readout[0].in_features].copy_(base.readout[0].weight)
                model.readout[0].bias.copy_(base.readout[0].bias)
                model.readout[2:].load_state_dict({k: v for k, v in base.readout[2:].state_dict().items()})
            init_states[arm] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        init_hashes = {arm: state_sha(st) for arm, st in init_states.items()}
        if init_hashes["raw"] != init_hashes["gate"]:
            raise ValueError("paired raw/gate initial state mismatch")
        station_result = {"train_rows": int(len(train_i)), "val_rows": int(len(val_i)),
                          "epoch_orders_sha256": order_hash, "initial_state_sha256": init_hashes["raw"],
                          "arms": {}}

        for arm in ("raw", "gate"):
            x, g, c = tensors[arm]
            model = LocalGHIHead("AC", 2).to(device)
            model.load_state_dict(init_states[arm], strict=True)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
            best = float("inf"); best_state = None; best_epoch = -1; updates = 0
            records = []; started = time.time()
            run_dir = a.output / station / arm; run_dir.mkdir(parents=True)
            for ep, order in enumerate(orders):
                model.train(); total = 0.; seen = 0
                for lo in range(0, len(order), a.batch_size):
                    ix = torch.as_tensor(order[lo:lo + a.batch_size], device=device)
                    predkt = model(x[ix], g[ix], c[ix])
                    err = (predkt * CL[ix] - GH[ix]) / 1000.
                    loss = err.square().mean()
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    opt.step(); updates += 1
                    total += float(loss.detach()) * len(ix); seen += len(ix)
                sched.step(); model.eval()
                with torch.inference_mode():
                    pred_parts = []
                    for lo in range(0, len(val_i), a.batch_size):
                        ix = torch.as_tensor(val_i[lo:lo + a.batch_size], device=device)
                        pred_parts.append((model(x[ix], g[ix], c[ix]) * CL[ix]).cpu().numpy())
                    pred_val = np.concatenate(pred_parts).astype(np.float64)
                truth = gh[val_i].astype(np.float64)
                rmse = float(np.sqrt(np.mean((pred_val - truth) ** 2)))
                rec = {"station": station, "arm": arm, "epoch": ep + 1,
                       "updates": updates, "train_mse_scaled": total / seen,
                       "val_ghi_rmse_wm2": rmse, "elapsed_seconds": time.time() - started}
                records.append(rec)
                if rmse < best:
                    best, best_state, best_epoch = rmse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1
                if ep in (0, a.epochs - 1) or (ep + 1) % 5 == 0:
                    print(json.dumps(rec), flush=True)
            expected_updates = a.epochs * math.ceil(len(train_i) / a.batch_size)
            if updates != expected_updates:
                raise ValueError(f"update count mismatch: {updates} != {expected_updates}")
            model.load_state_dict(best_state, strict=True); model.eval()
            with torch.inference_mode():
                pred_parts = []
                for lo in range(0, len(val_i), a.batch_size):
                    ix = torch.as_tensor(val_i[lo:lo + a.batch_size], device=device)
                    pred_parts.append((model(x[ix], g[ix], c[ix]) * CL[ix]).cpu().numpy())
                pred_val = np.concatenate(pred_parts).astype(np.float64)
            np.save(run_dir / "val_prediction_ghi_wm2.npy", pred_val)
            np.savez_compressed(run_dir / "val_keys_truth.npz",
                station=np.asarray([rows[i]["station"] for i in val_i]),
                target_time_utc=np.asarray([rows[i]["target_time_utc"] for i in val_i]),
                target_time_bjt=np.asarray([(datetime.strptime(rows[i]["target_time_utc"], "%Y-%m-%d %H:%M:%S")
                                             + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S") for i in val_i]),
                ghi=gh[val_i], clear=clear[val_i], prediction_ghi_wm2=pred_val.astype(np.float32))
            torch.save({"model": best_state, "arm": "AC_" + arm, "seed": a.seed,
                        "best_epoch": best_epoch, "best_val_rmse_wm2": best,
                        "initial_state_sha256": init_hashes[arm], "epoch_orders_sha256": order_hash},
                       run_dir / "best.pt")
            valrec = {"station": station, "arm": arm, "seed": a.seed,
                "best_epoch": best_epoch, "actual_updates": updates, "expected_updates": expected_updates,
                "best_val_rmse_wm2": float(np.sqrt(np.mean((pred_val - gh[val_i]) ** 2))),
                "val_mae_wm2": float(np.mean(np.abs(pred_val - gh[val_i]))),
                "val_bias_wm2": float(np.mean(pred_val - gh[val_i])), "n": len(val_i),
                "initial_state_sha256": init_hashes[arm], "epoch_orders_sha256": order_hash,
                "cot_norm_mean": float(packs[arm]["meta"]["COT_norm_mean"]),
                "cot_norm_std": float(packs[arm]["meta"]["COT_norm_std"]),
                "checkpoint_sha256": sha_file(run_dir / "best.pt"), "test_payload_opened": False}
            (run_dir / "validation.json").write_text(json.dumps(valrec, indent=2) + "\n")
            with (run_dir / "epochs.jsonl").open("w") as f:
                for rec in records: f.write(json.dumps(rec) + "\n")
            station_result["arms"][arm] = valrec
            print(f"PAIRED_ARM_COMPLETE station={station} arm={arm} updates={updates}", flush=True)
        all_results[station] = station_result
        del tensors, GH, CL, base
        torch.cuda.empty_cache()

    result = {"state": "COMPLETE_SEED42_RAW_GATE_PAIRED_GHI", "seed": a.seed,
              "epochs": a.epochs, "batch_size": a.batch_size, "stations": all_results,
              "test_payload_opened": False}
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PAIRED_RAW_GATE_GHI_SEED42_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
