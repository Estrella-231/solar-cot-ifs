#!/usr/bin/env python3
"""Paired static/scan-interval A vs AC GHI heads on the frozen train/val pack."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from head_candidate_v0 import LocalGHIHead, parameter_counts


ARMS = ("A_static", "AC_static", "A_interval", "AC_interval")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def rmse_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred.astype(np.float64) - truth.astype(np.float64)
    return {"n": int(err.size), "rmse_wm2": float(np.sqrt(np.mean(err * err))),
            "mae_wm2": float(np.mean(np.abs(err))), "bias_wm2": float(np.mean(err))}


def forward(model, arm: str, x, g, c, weights, scan_clear):
    if arm.endswith("static"):
        return model(x[:, -1], g[:, -1], c[:, -1] if arm.startswith("AC") else None)
    b, t = x.shape[:2]
    raw = model(x.reshape(b * t, *x.shape[2:]), g.reshape(b * t, *g.shape[2:]),
                c.reshape(b * t, *c.shape[2:]) if arm.startswith("AC") else None).reshape(b, t)
    eff = weights * scan_clear
    denom = eff.sum(dim=1, keepdim=True)
    if torch.any(denom <= 0) or not torch.isfinite(denom).all():
        raise ValueError("invalid C_j weighted support")
    return (raw * (eff / denom)).sum(dim=1)


def init_model(arm: str, seed: int, common_state: dict, device):
    torch.manual_seed(seed)
    variant = "AC" if arm.startswith("AC") else "A"
    model = LocalGHIHead(variant, 2).to(device)
    if variant == "A":
        model.load_state_dict(common_state)
    else:
        common = LocalGHIHead("A", 2)
        common.load_state_dict(common_state)
        with torch.no_grad():
            model.agri_encoder.load_state_dict(common.agri_encoder.state_dict())
            model.readout[0].weight.zero_()
            model.readout[0].weight[:, :common.readout[0].in_features].copy_(common.readout[0].weight)
            model.readout[0].bias.copy_(common.readout[0].bias)
            model.readout[2:].load_state_dict(common.readout[2:].state_dict())
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--station", choices=("sili", "zhujia"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--profile-output", type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("interval pilot profile/training requires CUDA")
    if args.output.exists() and not args.profile_only:
        raise FileExistsError(args.output)
    meta = json.loads((args.pack / "pack_manifest.json").read_text())
    if meta.get("state") != "COMPLETE" or meta.get("test_payload_opened") is not False:
        raise ValueError("input pack acceptance gate failed")
    if meta.get("mode") != "observed_AGRI_offline_interval_mean_diagnostic":
        raise ValueError("unexpected input mode")
    rows = list(csv.DictReader((args.pack / "rows.csv").open(newline="", encoding="utf-8")))
    station_ids = np.asarray([r["station"] == args.station for r in rows])
    train_all = np.asarray([r["split"] == "train" for r in rows]) & station_ids
    val_all = np.asarray([r["split"] == "val" for r in rows]) & station_ids
    qc1 = np.asarray([r["candidate_qc1_main_mask"] == "1" for r in rows])
    train = train_all & qc1
    val_qc1 = val_all & qc1
    if not train.any() or not val_qc1.any():
        raise ValueError("empty QC1 train/validation support")

    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ag = np.load(args.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(args.pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    geo = np.load(args.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(args.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    weights = np.stack([np.fromstring(r["weights"], sep="|") for r in rows]).astype(np.float32)
    scan_clear = np.asarray([[float(r[f"scan_clear_{j}_wm2"]) for j in range(3)] for r in rows], dtype=np.float32)
    ghi = np.asarray([float(r["ghi"]) for r in rows], dtype=np.float32)
    clear15 = np.asarray([float(r["clear"]) for r in rows], dtype=np.float32)
    am = np.asarray(meta["AGRI_norm_mean"], dtype=np.float32)[None, None, :, None, None]
    ast = np.asarray(meta["AGRI_norm_std"], dtype=np.float32)[None, None, :, None, None]
    X = np.where(valid, (np.asarray(ag) - am) / ast, 0.0).astype(np.float32)
    G = np.array(geo, dtype=np.float32, copy=True)
    cm, cs = np.float32(meta["COT_norm_mean"]), np.float32(meta["COT_norm_std"])
    C = ((np.asarray(cot, dtype=np.float32) - cm) / cs).astype(np.float32)

    subset = np.flatnonzero(station_ids)
    row_map = subset
    Xg = torch.from_numpy(np.array(X[row_map], copy=True)).to(device)
    Gg = torch.from_numpy(np.array(G[row_map], copy=True)).to(device)
    Cg = torch.from_numpy(np.array(C[row_map], copy=True)).to(device)
    Wg = torch.from_numpy(np.array(weights[row_map], copy=True)).to(device)
    Sg = torch.from_numpy(np.array(scan_clear[row_map], copy=True)).to(device)
    GHg = torch.from_numpy(np.array(ghi[row_map], copy=True)).to(device)
    CLg = torch.from_numpy(np.array(clear15[row_map], copy=True)).to(device)
    local_train = np.flatnonzero(train[row_map])
    local_val_all = np.flatnonzero(val_all[row_map])
    local_val_qc1 = np.flatnonzero(val_qc1[row_map])
    global_train_rows = row_map[local_train]
    global_val_rows = row_map[local_val_all]
    if args.profile_only:
        if args.profile_output is None:
            raise ValueError("--profile-output required with --profile-only")
        profile = []
        model = LocalGHIHead("AC", 2).to(device)
        for batch in (128, 256, 512, 1024, 1536, 2000):
            if batch > len(local_train):
                continue
            ix = torch.as_tensor(local_train[:batch], device=device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(3):
                pred = forward(model, "AC_interval", Xg[ix], Gg[ix], Cg[ix], Wg[ix], Sg[ix])
                loss = (((pred * CLg[ix] - GHg[ix]) / 1000.0) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            steps = 12
            for _ in range(steps):
                pred = forward(model, "AC_interval", Xg[ix], Gg[ix], Cg[ix], Wg[ix], Sg[ix])
                loss = (((pred * CLg[ix] - GHg[ix]) / 1000.0) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            profile.append({"batch": batch, "examples_per_second": batch * steps / elapsed,
                            "seconds_per_update": elapsed / steps,
                            "peak_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                            "updates_per_epoch": math.ceil(len(local_train) / batch)})
            print(json.dumps(profile[-1]), flush=True)
            del opt
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps({"state": "PROFILE_COMPLETE", "station": args.station,
                                                   "results": profile, "profiled_arm": "AC_interval",
                                                   "test_payload_opened": False}, indent=2) + "\n")
        return

    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    base = LocalGHIHead("A", 2).to(device)
    common_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    rng = np.random.default_rng(args.seed)
    epoch_orders = [rng.permutation(local_train) for _ in range(args.epochs)]
    metadata = {"state": "TRAINING", "station": args.station, "seed": args.seed, "arms": list(ARMS),
                "epochs": args.epochs, "batch_size": args.batch_size, "train_n": int(len(local_train)),
                "val_qc0_n": int(len(local_val_all)), "val_qc1_n": int(len(local_val_qc1)),
                "split_id": meta["split_id"], "pack_manifest_sha256": sha256(args.pack / "pack_manifest.json"),
                "head_source_sha256": sha256(Path(__file__).with_name("head_candidate_v0.py")),
                "training_source_sha256": sha256(Path(__file__)), "test_payload_opened": False,
                "parameters": parameter_counts(2), "selection_metric": "QC1 validation GHI RMSE",
                "paired_design": "same per-station random initialization and identical epoch sample orders across all four arms"}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_metrics = []
    experiment_start = time.time()
    truth_all = ghi[global_val_rows].astype(np.float64)
    pred_best = {}
    for arm in ARMS:
        model = init_model(arm, args.seed, common_state, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        best_qc1 = float("inf"); best_epoch = -1; best_state = None
        t0 = time.time()
        for epoch, order in enumerate(epoch_orders):
            model.train(); loss_sum = 0.0; seen = 0
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start + args.batch_size]
                ti = torch.as_tensor(idx, device=device)
                kt_pred = forward(model, arm, Xg[ti], Gg[ti], Cg[ti], Wg[ti], Sg[ti])
                err = (kt_pred * CLg[ti] - GHg[ti]) / 1000.0
                loss = err.square().mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                loss_sum += float(loss.detach()) * len(idx); seen += len(idx)
            scheduler.step(); model.eval()
            with torch.inference_mode():
                qc1_preds = []
                for st in range(0, len(local_val_qc1), args.batch_size):
                    idx = torch.as_tensor(local_val_qc1[st:st + args.batch_size], device=device)
                    pred = forward(model, arm, Xg[idx], Gg[idx], Cg[idx], Wg[idx], Sg[idx]) * CLg[idx]
                    qc1_preds.append(pred.cpu().numpy())
                val_qc1_pred = np.concatenate(qc1_preds).astype(np.float64)
            qc1_truth = ghi[row_map[local_val_qc1]].astype(np.float64)
            qc1_rmse = float(np.sqrt(np.mean((val_qc1_pred - qc1_truth) ** 2)))
            rec = {"arm": arm, "epoch": epoch, "train_mse_scaled": loss_sum / max(seen, 1),
                   "val_qc1_ghi_rmse_wm2": qc1_rmse, "lr": optimizer.param_groups[0]["lr"]}
            all_metrics.append(rec)
            if qc1_rmse < best_qc1:
                best_qc1 = qc1_rmse; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch in (0, args.epochs - 1) or (epoch + 1) % 5 == 0:
                print(json.dumps(rec), flush=True)
        model.load_state_dict(best_state); model.eval()
        with torch.inference_mode():
            full_parts = []
            for st in range(0, len(local_val_all), args.batch_size):
                idx = torch.as_tensor(local_val_all[st:st + args.batch_size], device=device)
                full_parts.append((forward(model, arm, Xg[idx], Gg[idx], Cg[idx], Wg[idx], Sg[idx]) * CLg[idx]).cpu().numpy())
            full_pred = np.concatenate(full_parts).astype(np.float64)
        qc1_mask_in_val = qc1[row_map[local_val_all]]
        metrics = {"arm": arm, "station": args.station, "seed": args.seed, "best_epoch": best_epoch,
                   "qc0": rmse_metrics(full_pred, truth_all),
                   "qc1": rmse_metrics(full_pred[qc1_mask_in_val], truth_all[qc1_mask_in_val]),
                   "checkpoint_sha256": None, "test_payload_opened": False}
        np.save(args.output / f"{arm}_val_prediction_ghi_wm2.npy", full_pred)
        torch.save({"model": best_state, "arm": arm, "seed": args.seed, "best_epoch": best_epoch,
                    "best_qc1_rmse_wm2": best_qc1}, args.output / f"{arm}_best.pt")
        metrics["checkpoint_sha256"] = sha256(args.output / f"{arm}_best.pt")
        (args.output / f"{arm}_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        pred_best[arm] = metrics
        print(f"ARM_COMPLETE {arm} seconds={time.time()-t0:.1f} qc0={metrics['qc0']['rmse_wm2']:.5f} qc1={metrics['qc1']['rmse_wm2']:.5f}", flush=True)
        del optimizer, scheduler, model
    with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_metrics:
            f.write(json.dumps(rec) + "\n")
    np.save(args.output / "val_global_row_indices.npy", global_val_rows)
    final = {"state": "COMPLETE", "station": args.station, "seed": args.seed,
             "epochs": args.epochs, "batch_size": args.batch_size,
             "validation": pred_best, "training_log_sha256": sha256(args.output / "metrics.jsonl"),
             "test_payload_opened": False, "elapsed_s": time.time() - experiment_start}
    (args.output / "result.json").write_text(json.dumps(final, indent=2) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({**metadata, "state": "COMPLETE"}, indent=2) + "\n")
    print("INTERVAL_FOUR_ARM_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
