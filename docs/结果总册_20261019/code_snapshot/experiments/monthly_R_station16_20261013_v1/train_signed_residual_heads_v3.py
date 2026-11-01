#!/usr/bin/env python3
"""Paired signed residual experiment; train/validation only, never opens test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from head_candidate_v0 import LocalGHIHead
from signed_residual_head import SignedResidualHead, trainable_parameter_count


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--stations", nargs="+", choices=["sili", "zhujia"],
                    default=["sili", "zhujia"])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal residual training requires CUDA")
    # The site swc image currently has an incompatible cuDNN shared-library
    # pair. The heads use only tiny 1x1 convolutions; native CUDA kernels avoid
    # that loader path without changing the mathematical operator.
    torch.backends.cudnn.enabled = False
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_root}")

    manifest = json.loads((args.pack / "pack_manifest.json").read_text())
    if manifest.get("state") != "COMPLETE" or manifest.get("test_payload_opened") is not False:
        raise ValueError("pack manifest gate failed")
    if manifest.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split contract")
    rows = list(csv.DictReader((args.pack / "rows.csv").open(
        newline="", encoding="utf-8")))
    n = len(rows)
    agri = np.load(args.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(args.pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(args.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(args.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if agri.shape != (n, 13, 3, 3) or valid.shape != agri.shape or geometry.shape != (n, 2, 3, 3) or cot.shape != (n, 1, 3, 3):
        raise ValueError("feature array shape contract failed")

    am = np.asarray(manifest["AGRI_norm_mean"], np.float32)[None, :, None, None]
    ast = np.asarray(manifest["AGRI_norm_std"], np.float32)[None, :, None, None]
    x = np.where(valid.astype(bool), (np.asarray(agri) - am) / ast, 0).astype(np.float32)
    g = np.array(geometry, dtype=np.float32, copy=True)
    cm, cs = np.float32(manifest["COT_norm_mean"]), np.float32(manifest["COT_norm_std"])
    c = ((np.asarray(cot, dtype=np.float32) - cm) / cs).astype(np.float32)
    target = np.asarray([float(r["ghi"]) for r in rows], np.float32)
    clear = np.asarray([float(r["clear"]) for r in rows], np.float32)
    X = torch.from_numpy(x).cuda()
    G = torch.from_numpy(g).cuda()
    C = torch.from_numpy(c).cuda()
    Y = torch.from_numpy(target).cuda()
    CL = torch.from_numpy(clear).cuda()
    torch.set_num_threads(4)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_summary = {
        "state": "RUNNING", "split_id": manifest["split_id"],
        "pack_manifest_sha256": sha256(args.pack / "pack_manifest.json"),
        "test_payload_opened": False, "arms": ["AGRI", "AGRI_COT"],
        "same_frozen_A_per_station_seed": True,
        "residual_units": "signed W m-2", "loss": "MSE of final physical GHI / 1000",
        "cudnn_enabled": False,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "trainable_parameter_counts": {}, "runs": []}
    all_summary["training_source_sha256"] = sha256(Path(__file__))
    all_summary["model_source_sha256"] = sha256(Path(__file__).with_name("signed_residual_head.py"))
    all_summary["A_model_source_sha256"] = sha256(Path(__file__).with_name("head_candidate_v0.py"))
    (args.output_root / "experiment_status.json").write_text(json.dumps(all_summary, indent=2) + "\n")

    for station in args.stations:
        for seed in args.seeds:
            base_dir = args.baseline_root / f"ghi_heads_new_r_seed{seed}_v1" / station
            checkpoint = base_dir / "A_best.pt"
            a_validation = json.loads((base_dir / "A_validation.json").read_text())
            if a_validation.get("test_payload_opened") is not False:
                raise ValueError(f"baseline provenance gate failed: {base_dir}")
            keep = np.asarray([r["station"] == station for r in rows])
            tr = keep & np.asarray([r["split"] == "train" for r in rows])
            va = keep & np.asarray([r["split"] == "val" for r in rows])
            train_i, val_i = np.flatnonzero(tr), np.flatnonzero(va)
            if not len(train_i) or not len(val_i):
                raise ValueError(f"empty split for {station}")
            a_model = LocalGHIHead("A", geometry_channels=2).cuda()
            a_model.load_state_dict(load_state(checkpoint), strict=True)
            a_model.eval()
            for parameter in a_model.parameters():
                parameter.requires_grad_(False)
            with torch.inference_mode():
                a_parts = []
                for start in range(0, len(val_i), args.batch_size):
                    ix = torch.as_tensor(val_i[start:start + args.batch_size], device="cuda")
                    a_parts.append((a_model(X[ix], G[ix]) * CL[ix]).cpu().numpy())
            a_pred = np.concatenate(a_parts)
            saved_a = np.load(base_dir / "A_val_prediction_ghi_wm2.npy", allow_pickle=False)
            baseline_delta = float(np.max(np.abs(saved_a - a_pred))) if saved_a.shape == a_pred.shape else float("inf")
            if saved_a.shape != a_pred.shape or baseline_delta > 1.0:
                raise ValueError(f"reloaded A mismatch for {station}/{seed}: "
                                 f"saved={saved_a.shape}, reloaded={a_pred.shape}, "
                                 f"max_abs={baseline_delta:.8g} W m-2")
            del a_model

            rng = np.random.default_rng(seed)
            orders = [rng.permutation(train_i) for _ in range(args.epochs)]
            run_dir = args.output_root / f"seed{seed}" / station
            run_dir.mkdir(parents=True, exist_ok=False)
            np.save(run_dir / "A_reloaded_val_prediction_ghi_wm2.npy", a_pred)
            (run_dir / "A_reload_check.json").write_text(json.dumps({
                "station": station, "seed": seed,
                "saved_A_max_abs_delta_wm2": baseline_delta,
                "accepted_tolerance_wm2": 1.0,
                "note": "native CUDA convolution avoids the cluster cuDNN loader conflict; step-zero identity is exact relative to this reloaded A",
                "frozen_A_sha256": sha256(checkpoint),
                "test_payload_opened": False}, indent=2) + "\n")
            arm_records = []
            for arm in ("AGRI", "AGRI_COT"):
                torch.manual_seed(seed)
                model = SignedResidualHead(arm).cuda()
                model.load_frozen_a(load_state(checkpoint))
                count = trainable_parameter_count(model)
                all_summary["trainable_parameter_counts"][arm] = count
                if arm_records and count != all_summary["trainable_parameter_counts"]["AGRI"]:
                    raise AssertionError("residual arms are not capacity matched")
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-5)
                best_rmse, best_state, best_epoch = float("inf"), None, -1
                history = []
                start_time = time.time()
                # Step-zero identity witness on held-out rows before optimization.
                with torch.inference_mode():
                    ix = torch.as_tensor(val_i[:min(512, len(val_i))], device="cuda")
                    initial, initial_base, initial_delta = model(X[ix], G[ix], C[ix], CL[ix])
                    identity_error = float((initial - initial_base).abs().max().cpu())
                    delta_max = float(initial_delta.abs().max().cpu())
                if identity_error != 0.0 or delta_max != 0.0:
                    raise AssertionError("zero-init residual is not exact identity")

                for epoch, order in enumerate(orders):
                    model.train()
                    model.features.eval()
                    sum_loss = 0.0
                    for start in range(0, len(order), args.batch_size):
                        ix_np = order[start:start + args.batch_size]
                        ix = torch.as_tensor(ix_np, device="cuda")
                        pred, _, _ = model(X[ix], G[ix], C[ix], CL[ix])
                        loss = ((pred - Y[ix]) / 1000.0).square().mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 5.0)
                        optimizer.step()
                        sum_loss += float(loss.detach()) * len(ix_np)
                    scheduler.step()
                    model.eval()
                    with torch.inference_mode():
                        vp = []
                        for start in range(0, len(val_i), args.batch_size):
                            ix = torch.as_tensor(val_i[start:start + args.batch_size], device="cuda")
                            pred, _, _ = model(X[ix], G[ix], C[ix], CL[ix])
                            vp.append(pred.cpu().numpy())
                    pred = np.concatenate(vp).astype(np.float64)
                    truth = target[val_i].astype(np.float64)
                    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                    mae = float(np.mean(np.abs(pred - truth)))
                    rec = {"station": station, "seed": seed, "arm": arm,
                           "epoch": epoch, "train_mse_scaled": sum_loss / len(train_i),
                           "val_ghi_rmse_wm2": rmse, "val_ghi_mae_wm2": mae,
                           "lr": optimizer.param_groups[0]["lr"]}
                    history.append(rec)
                    if rmse < best_rmse:
                        best_rmse, best_epoch = rmse, epoch
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                        print(json.dumps(rec), flush=True)
                model.load_state_dict(best_state)
                model.eval()
                with torch.inference_mode():
                    predictions, bases, deltas = [], [], []
                    for start in range(0, len(val_i), args.batch_size):
                        ix = torch.as_tensor(val_i[start:start + args.batch_size], device="cuda")
                        pred, base, delta = model(X[ix], G[ix], C[ix], CL[ix])
                        predictions.append(pred.cpu().numpy())
                        bases.append(base.cpu().numpy())
                        deltas.append(delta.cpu().numpy())
                pred = np.concatenate(predictions).astype(np.float64)
                base = np.concatenate(bases).astype(np.float64)
                delta = np.concatenate(deltas).astype(np.float64)
                truth = target[val_i].astype(np.float64)
                np.save(run_dir / f"{arm}_val_prediction_ghi_wm2.npy", pred)
                np.save(run_dir / f"{arm}_val_signed_residual_wm2.npy", delta)
                torch.save({"model": best_state, "arm": arm, "seed": seed,
                            "station": station, "best_val_rmse_wm2": best_rmse,
                            "frozen_A_sha256": sha256(checkpoint)},
                           run_dir / f"{arm}_best.pt")
                (run_dir / f"{arm}_metrics.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in history))
                payload = {
                    "station": station, "seed": seed, "arm": arm,
                    "n_val": len(val_i), "best_epoch": best_epoch,
                    "rmse_wm2": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "mae_wm2": float(np.mean(np.abs(pred - truth))),
                    "baseline_A_rmse_wm2": float(np.sqrt(np.mean((base - truth) ** 2))),
                    "mean_signed_adjustment_wm2": float(np.mean(delta)),
                    "mean_abs_adjustment_wm2": float(np.mean(np.abs(delta))),
                    "p90_abs_adjustment_wm2": float(np.quantile(np.abs(delta), .9)),
                    "positive_adjustment_fraction": float(np.mean(delta > 0)),
                    "negative_adjustment_fraction": float(np.mean(delta < 0)),
                    "step_zero_identity_max_abs_wm2": identity_error,
                    "trainable_parameters": count,
                    "frozen_A_sha256": sha256(checkpoint),
                    "pack_manifest_sha256": sha256(args.pack / "pack_manifest.json"),
                    "training_source_sha256": all_summary["training_source_sha256"],
                    "model_source_sha256": all_summary["model_source_sha256"],
                    "test_payload_opened": False}
                (run_dir / f"{arm}_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
                arm_records.append(payload)
                all_summary["runs"].append(payload)
                print(f"ARM_COMPLETE station={station} seed={seed} arm={arm} "
                      f"rmse={payload['rmse_wm2']:.4f} elapsed_s={time.time()-start_time:.1f}", flush=True)

            # Stable row/key sidecar for paired scoring; only the declared val rows.
            with (run_dir / "validation_keys.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows[i] for i in val_i)
            if len(arm_records) != 2:
                raise AssertionError("paired arm outputs incomplete")
            (run_dir / "paired_summary.json").write_text(json.dumps({
                "station": station, "seed": seed,
                "A_baseline_rmse_wm2": arm_records[0]["baseline_A_rmse_wm2"],
                "AGRI_residual_rmse_wm2": arm_records[0]["rmse_wm2"],
                "AGRI_COT_residual_rmse_wm2": arm_records[1]["rmse_wm2"],
                "COT_increment_vs_capacity_matched_AGRI_wm2":
                    arm_records[1]["rmse_wm2"] - arm_records[0]["rmse_wm2"],
                "test_payload_opened": False}, indent=2) + "\n")
            all_summary["state"] = "RUNNING"
            (args.output_root / "experiment_status.json").write_text(json.dumps(all_summary, indent=2) + "\n")

    all_summary["state"] = "COMPLETE"
    (args.output_root / "experiment_status.json").write_text(json.dumps(all_summary, indent=2) + "\n")
    print("SIGNED_RESIDUAL_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
