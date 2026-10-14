#!/usr/bin/env python3
"""Train matched signed residual heads on blocked OOF A forecasts/features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from head_candidate_v0 import LocalGHIHead
from signed_residual_head_oof import SignedResidualHeadOOF, trainable_parameter_count


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_state(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("model", payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--oof-root", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--stations", nargs="+", choices=["sili", "zhujia"],
                    default=["sili", "zhujia"])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    torch.backends.cudnn.enabled = False
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_root}")
    manifest_path = args.pack / "pack_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "COMPLETE" or manifest.get("test_payload_opened") is not False:
        raise ValueError("pack manifest gate failed")
    if manifest.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split contract")
    rows = list(csv.DictReader((args.pack / "rows.csv").open(newline="", encoding="utf-8")))
    oof_status_path = args.oof_root / "oof_status.json"
    oof_status = json.loads(oof_status_path.read_text())
    if oof_status.get("state") != "COMPLETE" or oof_status.get("test_payload_opened") is not False:
        raise ValueError("OOF status gate failed")
    if oof_status.get("pack_manifest_sha256") != sha256(manifest_path):
        raise ValueError("OOF predictions were built from a different pack manifest")
    n = len(rows)
    agri = np.load(args.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(args.pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(args.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(args.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if agri.shape != (n, 13, 3, 3) or valid.shape != agri.shape or geometry.shape != (n, 2, 3, 3) or cot.shape != (n, 1, 3, 3):
        raise ValueError("pack feature shape contract failed")
    am = np.asarray(manifest["AGRI_norm_mean"], np.float32)[None, :, None, None]
    ast = np.asarray(manifest["AGRI_norm_std"], np.float32)[None, :, None, None]
    x = np.where(np.asarray(valid, dtype=bool),
                 (np.asarray(agri, dtype=np.float32) - am) / ast, 0).astype(np.float32)
    g = np.array(geometry, dtype=np.float32, copy=True)
    cm, cs = np.float32(manifest["COT_norm_mean"]), np.float32(manifest["COT_norm_std"])
    c = ((np.asarray(cot, dtype=np.float32) - cm) / cs).astype(np.float32)
    target = np.asarray([float(r["ghi"]) for r in rows], np.float32)
    clear = np.asarray([float(r["clear"]) for r in rows], np.float32)
    X, G, C = (torch.from_numpy(v).cuda() for v in (x, g, c))
    Y, CL = (torch.from_numpy(v).cuda() for v in (target, clear))
    torch.set_num_threads(4)
    args.output_root.mkdir(parents=True, exist_ok=True)
    top = {"state": "RUNNING", "split_id": manifest["split_id"],
           "pack_manifest_sha256": sha256(manifest_path),
           "oof_status_sha256": sha256(args.oof_root / "oof_status.json"),
           "test_payload_opened": False, "arms": ["AGRI", "AGRI_COT"],
           "epochs": args.epochs, "batch_size": args.batch_size,
           "trainable_parameter_counts": {}, "runs": [],
           "training_source_sha256": sha256(Path(__file__)),
           "model_source_sha256": sha256(Path(__file__).with_name("signed_residual_head_oof.py"))}
    (args.output_root / "experiment_status.json").write_text(json.dumps(top, indent=2) + "\n")

    for station in args.stations:
        keep = np.asarray([r["station"] == station for r in rows])
        train_i = np.flatnonzero(keep & np.asarray([r["split"] == "train" for r in rows]))
        val_i = np.flatnonzero(keep & np.asarray([r["split"] == "val" for r in rows]))
        if not len(train_i) or not len(val_i):
            raise ValueError(f"empty train/val split for {station}")
        for seed in args.seeds:
            baseline_dir = args.baseline_root / f"ghi_heads_new_r_seed{seed}_v1" / station
            checkpoint = baseline_dir / "A_best.pt"
            oof_dir = args.oof_root / f"seed{seed}" / station
            oof_idx = np.load(oof_dir / "train_pack_row_indices.npy", allow_pickle=False)
            if not np.array_equal(oof_idx, train_i):
                raise ValueError(f"OOF row keys/order mismatch for {station}/{seed}")
            oof_meta = json.loads((oof_dir / "run_metadata.json").read_text())
            if (oof_meta.get("station") != station or oof_meta.get("seed") != seed
                    or oof_meta.get("test_payload_opened") is not False):
                raise ValueError(f"OOF run provenance mismatch for {station}/{seed}")
            oof_pred = np.load(oof_dir / "A_oof_train_prediction_ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
            oof_enc = np.load(oof_dir / "A_oof_train_encoded_features.npy", mmap_mode="r", allow_pickle=False)
            fold_id = np.load(oof_dir / "oof_fold_id.npy", allow_pickle=False)
            if oof_pred.shape != (len(train_i),) or oof_enc.shape != (len(train_i), 32, 3, 3) or fold_id.shape != (len(train_i),):
                raise ValueError("OOF output shape mismatch")
            if not np.isfinite(oof_pred).all() or not np.isfinite(oof_enc).all():
                raise ValueError("nonfinite OOF output")
            a_model = LocalGHIHead("A", geometry_channels=2).cuda()
            a_model.load_state_dict(load_state(checkpoint), strict=True)
            a_model.eval()
            for p in a_model.parameters():
                p.requires_grad_(False)
            with torch.inference_mode():
                a_parts = []
                for start in range(0, len(val_i), args.batch_size):
                    ix = torch.as_tensor(val_i[start:start+args.batch_size], device="cuda")
                    a_parts.append((a_model(X[ix], G[ix]) * CL[ix]).cpu().numpy())
            a_val = np.concatenate(a_parts)
            saved = np.load(baseline_dir / "A_val_prediction_ghi_wm2.npy", allow_pickle=False)
            delta = float(np.max(np.abs(saved - a_val)))
            if saved.shape != a_val.shape or delta > 1.0:
                raise ValueError(f"reloaded full A mismatch {station}/{seed}: {delta}")
            truth = target[val_i].astype(np.float64)
            rng = np.random.default_rng(seed)
            orders = [rng.permutation(train_i) for _ in range(args.epochs)]
            run_dir = args.output_root / f"seed{seed}" / station
            run_dir.mkdir(parents=True, exist_ok=False)
            np.save(run_dir / "A_full_val_prediction_ghi_wm2.npy", a_val.astype(np.float32))
            (run_dir / "A_reload_check.json").write_text(json.dumps({
                "station": station, "seed": seed, "saved_A_max_abs_delta_wm2": delta,
                "accepted_tolerance_wm2": 1.0, "test_payload_opened": False}, indent=2) + "\n")
            oof_base = torch.from_numpy(np.array(oof_pred, dtype=np.float32, copy=True)).cuda()
            oof_features = torch.from_numpy(np.array(oof_enc, dtype=np.float32, copy=True)).cuda()
            del oof_pred, oof_enc
            for arm in ("AGRI", "AGRI_COT"):
                torch.manual_seed(seed)
                model = SignedResidualHeadOOF(arm).cuda()
                model.load_frozen_a(load_state(checkpoint))
                count = trainable_parameter_count(model)
                top["trainable_parameter_counts"][arm] = count
                if len(top["trainable_parameter_counts"]) == 2 and len(set(top["trainable_parameter_counts"].values())) != 1:
                    raise AssertionError("residual arms are not capacity matched")
                # Include the exact untrained identity in checkpoint selection.
                # This makes A an explicit safe candidate when cross-fitted
                # corrections do not transfer to the full-data A model.
                model.eval()
                with torch.inference_mode():
                    zero_parts, zero_delta_parts = [], []
                    for start in range(0, len(val_i), args.batch_size):
                        ix = torch.as_tensor(val_i[start:start+args.batch_size], device="cuda")
                        zp, zb, zd = model(X[ix], G[ix], C[ix], CL[ix])
                        zero_parts.append(zp.cpu().numpy())
                        zero_delta_parts.append(zd.cpu().numpy())
                zero_pred = np.concatenate(zero_parts).astype(np.float64)
                zero_delta = np.concatenate(zero_delta_parts).astype(np.float64)
                zero_identity_error = float(np.max(np.abs(zero_pred-a_val)))
                if zero_identity_error != 0.0 or np.count_nonzero(zero_delta) != 0:
                    raise AssertionError("step-zero residual is not an exact A identity")
                best = float(np.sqrt(np.mean((zero_pred-truth)**2)))
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_epoch = -1
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
                history = [{"station": station, "seed": seed, "arm": arm,
                            "epoch": -1, "candidate": "step_zero_identity",
                            "train_mse_scaled": None, "val_ghi_rmse_wm2": best,
                            "lr": optimizer.param_groups[0]["lr"]}]
                for epoch, order in enumerate(orders):
                    model.train(); model.features.eval()
                    total = 0.0
                    for start in range(0, len(order), args.batch_size):
                        ids = order[start:start+args.batch_size]
                        # map pack row IDs to their positions in the station's ordered OOF table
                        local = np.searchsorted(train_i, ids)
                        ix = torch.as_tensor(ids, device="cuda")
                        oi = torch.as_tensor(local, device="cuda")
                        pred, _, _ = model(X[ix], G[ix], C[ix], CL[ix],
                                           base_override=oof_base[oi],
                                           encoded_override=oof_features[oi])
                        loss = ((pred - Y[ix]) / 1000.0).square().mean()
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                        optimizer.step(); total += float(loss.detach()) * len(ids)
                    scheduler.step(); model.eval()
                    with torch.inference_mode():
                        vp, vd = [], []
                        for start in range(0, len(val_i), args.batch_size):
                            ix = torch.as_tensor(val_i[start:start+args.batch_size], device="cuda")
                            p, _, d = model(X[ix], G[ix], C[ix], CL[ix])
                            vp.append(p.cpu().numpy()); vd.append(d.cpu().numpy())
                    pred = np.concatenate(vp).astype(np.float64)
                    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                    rec = {"station": station, "seed": seed, "arm": arm, "epoch": epoch,
                           "train_mse_scaled": total / len(train_i), "val_ghi_rmse_wm2": rmse,
                           "lr": optimizer.param_groups[0]["lr"]}
                    history.append(rec)
                    if rmse < best:
                        best, best_epoch = rmse, epoch
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                        print(json.dumps(rec), flush=True)
                model.load_state_dict(best_state); model.eval()
                with torch.inference_mode():
                    pp, dd = [], []
                    for start in range(0, len(val_i), args.batch_size):
                        ix = torch.as_tensor(val_i[start:start+args.batch_size], device="cuda")
                        p, _, d = model(X[ix], G[ix], C[ix], CL[ix])
                        pp.append(p.cpu().numpy()); dd.append(d.cpu().numpy())
                pred = np.concatenate(pp).astype(np.float64); residual = np.concatenate(dd).astype(np.float64)
                np.save(run_dir / f"{arm}_val_prediction_ghi_wm2.npy", pred)
                np.save(run_dir / f"{arm}_val_signed_residual_wm2.npy", residual)
                torch.save({"model": best_state, "arm": arm, "seed": seed,
                            "station": station, "best_epoch": best_epoch,
                            "full_A_sha256": sha256(checkpoint),
                            "oof_prediction_sha256": sha256(oof_dir / "A_oof_train_prediction_ghi_wm2.npy")},
                           run_dir / f"{arm}_best.pt")
                payload = {"station": station, "seed": seed, "arm": arm,
                           "n_val": len(val_i), "best_epoch": best_epoch,
                           "selected_step_zero": best_epoch == -1,
                           "step_zero_identity_max_abs_wm2": zero_identity_error,
                           "rmse_wm2": float(np.sqrt(np.mean((pred-truth)**2))),
                           "mae_wm2": float(np.mean(np.abs(pred-truth))),
                           "positive_adjustment_fraction": float(np.mean(residual > 0)),
                           "negative_adjustment_fraction": float(np.mean(residual < 0)),
                           "trainable_parameters": count,
                           "full_A_sha256": sha256(checkpoint),
                           "oof_prediction_sha256": sha256(oof_dir / "A_oof_train_prediction_ghi_wm2.npy"),
                           "oof_features_sha256": sha256(oof_dir / "A_oof_train_encoded_features.npy"),
                           "pack_manifest_sha256": sha256(manifest_path), "test_payload_opened": False}
                (run_dir / f"{arm}_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
                (run_dir / f"{arm}_metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in history))
                top["runs"].append(payload)
                (args.output_root / "experiment_status.json").write_text(json.dumps(top, indent=2) + "\n")
                del model, optimizer, scheduler
                torch.cuda.empty_cache()
            del oof_base, oof_features, a_model
    top["state"] = "COMPLETE"
    (args.output_root / "experiment_status.json").write_text(json.dumps(top, indent=2) + "\n")
    print("OOF_SIGNED_RESIDUAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
