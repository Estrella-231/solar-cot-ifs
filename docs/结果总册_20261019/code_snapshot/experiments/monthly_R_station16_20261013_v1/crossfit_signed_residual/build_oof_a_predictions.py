#!/usr/bin/env python3
"""Build blocked, date-grouped OOF A forecasts/features from train rows only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from head_candidate_v0 import LocalGHIHead


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def bjt_date(utc_text: str) -> str:
    dt = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    return (dt + timedelta(hours=8)).date().isoformat()


def load_a_state(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("model", obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--stations", nargs="+", choices=["sili", "zhujia"],
                    default=["sili", "zhujia"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-max-batches", type=int, default=0,
                    help="nonzero limits each epoch to this many batches; smoke only")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device.startswith("cuda"):
        # Avoid the known incompatible cuDNN shared-library pair on this site.
        torch.backends.cudnn.enabled = False
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_root}")

    manifest_path = args.pack / "pack_manifest.json"
    manifest = json.loads(manifest_path.read_text())
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
    if agri.shape != (n, 13, 3, 3) or valid.shape != agri.shape or geometry.shape != (n, 2, 3, 3):
        raise ValueError("pack feature shape contract failed")
    means = np.asarray(manifest["AGRI_norm_mean"], np.float32)[None, :, None, None]
    stds = np.asarray(manifest["AGRI_norm_std"], np.float32)[None, :, None, None]
    Xn = np.where(np.asarray(valid, dtype=bool),
                  (np.asarray(agri, dtype=np.float32) - means) / stds, 0.0).astype(np.float32)
    Gn = np.array(geometry, dtype=np.float32, copy=True)
    target = np.full(n, np.nan, dtype=np.float32)
    clear_all = np.zeros(n, dtype=np.float32)
    for i, row in enumerate(rows):
        clear_all[i] = np.float32(row["clear"])
        if row["split"] == "train":
            target[i] = np.float32(row["ghi"])
    X = torch.from_numpy(Xn).to(args.device)
    G = torch.from_numpy(Gn).to(args.device)
    Y = torch.from_numpy(np.nan_to_num(target, nan=0.0)).to(args.device)
    CL = torch.from_numpy(clear_all).to(args.device)
    args.output_root.mkdir(parents=True, exist_ok=False)
    top = {
        "state": "RUNNING", "split_id": manifest["split_id"],
        "pack_manifest_sha256": sha256(manifest_path),
        "training_source_sha256": sha256(Path(__file__)),
        "head_source_sha256": sha256(Path(__file__).parents[1] / "head_candidate_v0.py"),
        "date_timezone": "BJT = target_time_utc + 8h",
        "folds": args.folds, "epochs": args.epochs,
        "batch_size": args.batch_size, "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "schedule": "CosineAnnealingLR(eta_min=1e-5)",
        "selection": "fixed final epoch; no validation-based fold selection",
        "date_buffer_days": 1, "test_payload_opened": False, "runs": []}
    (args.output_root / "oof_status.json").write_text(json.dumps(top, indent=2) + "\n")

    for station in args.stations:
        station_idx = np.asarray([i for i, r in enumerate(rows)
                                  if r["station"] == station and r["split"] == "train"], dtype=np.int64)
        if len(station_idx) == 0:
            raise ValueError(f"no train rows for {station}")
        dates = np.asarray([bjt_date(rows[i]["target_time_utc"]) for i in station_idx])
        unique_dates = np.asarray(sorted(set(dates.tolist())))
        if len(unique_dates) < args.folds * 2:
            raise ValueError(f"too few distinct train dates for {args.folds} folds: {len(unique_dates)}")
        date_folds = [list(chunk) for chunk in np.array_split(unique_dates, args.folds)]
        date_to_fold = {d: f for f, chunk in enumerate(date_folds) for d in chunk}
        fold_of_row = np.asarray([date_to_fold[d] for d in dates], dtype=np.int16)
        date_pos = {d: p for p, d in enumerate(unique_dates.tolist())}
        for seed in args.seeds:
            pred = np.full(len(station_idx), np.nan, dtype=np.float32)
            enc = np.full((len(station_idx), 32, 3, 3), np.nan, dtype=np.float32)
            folds_meta = []
            for fold, held_dates in enumerate(date_folds):
                held_dates_set = set(held_dates)
                low, high = date_pos[held_dates[0]], date_pos[held_dates[-1]]
                buffer_dates = set(unique_dates[max(0, low-1):low].tolist()) | set(unique_dates[high+1:high+2].tolist())
                is_hold = fold_of_row == fold
                is_buffer = np.asarray([d in buffer_dates for d in dates])
                train_local = np.flatnonzero(~is_hold & ~is_buffer)
                hold_local = np.flatnonzero(is_hold)
                if not len(train_local) or not len(hold_local):
                    raise ValueError(f"empty train/hold fold for {station}/{seed}/{fold}")
                fold_seed = int(seed + 1009 * (fold + 1))
                torch.manual_seed(fold_seed)
                np.random.seed(fold_seed)
                model = LocalGHIHead("A", geometry_channels=2).to(args.device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
                rng = np.random.default_rng(fold_seed)
                started = time.time()
                for epoch in range(args.epochs):
                    model.train()
                    order = rng.permutation(train_local)
                    loss_sum = 0.0
                    seen = 0
                    for batch_no, start in enumerate(range(0, len(order), args.batch_size)):
                        if args.smoke_max_batches and batch_no >= args.smoke_max_batches:
                            break
                        local = order[start:start + args.batch_size]
                        global_ix = station_idx[local]
                        ix = torch.as_tensor(global_ix, device=args.device)
                        out = model(X[ix], G[ix])
                        loss = ((out * CL[ix] - Y[ix]) / 1000.0).square().mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        loss_sum += float(loss.detach()) * len(local)
                        seen += len(local)
                    sched.step()
                    if epoch in (0, args.epochs - 1) or (epoch + 1) % 10 == 0:
                        print(json.dumps({"station": station, "seed": seed, "fold": fold,
                                          "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
                                          "rows": len(train_local), "held_rows": len(hold_local)}), flush=True)
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(hold_local), args.batch_size):
                        loc = hold_local[start:start + args.batch_size]
                        global_ix = station_idx[loc]
                        ix = torch.as_tensor(global_ix, device=args.device)
                        kt = model(X[ix], G[ix])
                        p = (kt * CL[ix]).cpu().numpy().astype(np.float32)
                        f = model.agri_encoder(torch.cat((X[ix], G[ix]), dim=1)).cpu().numpy().astype(np.float32)
                        pred[loc] = p
                        enc[loc] = f
                fold_truth = target[station_idx[hold_local]].astype(np.float64)
                rmse = float(np.sqrt(np.mean((pred[hold_local].astype(np.float64) - fold_truth) ** 2)))
                folds_meta.append({"fold": fold, "heldout_dates": held_dates,
                                   "buffer_dates_removed_from_fit": sorted(buffer_dates),
                                   "n_fit": int(len(train_local)), "n_holdout": int(len(hold_local)),
                                   "heldout_rmse_wm2": rmse, "fold_seed": fold_seed,
                                   "elapsed_s": time.time() - started})
                print(f"OOF_FOLD_COMPLETE station={station} seed={seed} fold={fold} rmse={rmse:.4f}", flush=True)
                del model, opt, sched
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if not np.isfinite(pred).all() or not np.isfinite(enc).all():
                raise AssertionError("OOF outputs do not cover every train row exactly once")
            out = args.output_root / f"seed{seed}" / station
            out.mkdir(parents=True, exist_ok=False)
            np.save(out / "train_pack_row_indices.npy", station_idx)
            np.save(out / "A_oof_train_prediction_ghi_wm2.npy", pred)
            np.save(out / "A_oof_train_encoded_features.npy", enc)
            np.save(out / "oof_fold_id.npy", fold_of_row)
            with (out / "train_rows.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["pack_row_index", "station", "split", "target_time_utc", "bjt_date", "fold"])
                for loc, ix in enumerate(station_idx):
                    r = rows[ix]; w.writerow([int(ix), station, "train", r["target_time_utc"], dates[loc], int(fold_of_row[loc])])
            run = {"station": station, "seed": seed, "n_train_rows": int(len(station_idx)),
                   "unique_train_dates": int(len(unique_dates)), "folds": folds_meta,
                   "oof_rmse_wm2": float(np.sqrt(np.mean((pred.astype(np.float64) - target[station_idx].astype(np.float64)) ** 2))),
                   "oof_prediction_sha256": sha256(out / "A_oof_train_prediction_ghi_wm2.npy"),
                   "oof_features_sha256": sha256(out / "A_oof_train_encoded_features.npy"),
                   "fold_id_sha256": sha256(out / "oof_fold_id.npy"), "test_payload_opened": False}
            (out / "run_metadata.json").write_text(json.dumps(run, indent=2) + "\n")
            top["runs"].append(run)
            (args.output_root / "oof_status.json").write_text(json.dumps(top, indent=2) + "\n")
    top["state"] = "COMPLETE"
    (args.output_root / "oof_status.json").write_text(json.dumps(top, indent=2) + "\n")
    print("OOF_A_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
