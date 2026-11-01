#!/usr/bin/env python3
"""C1: train-date OOF selection; frozen R; fixed B1 architecture and budget."""
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_folds(clp_path, manifest_path):
    with np.load(clp_path, allow_pickle=False) as z:
        index, split, station, utc = (z[k].copy() for k in
                                     ("sample_indices", "split", "station", "target_time_utc"))
        mask, clp = z["common_mask"][:, 0].astype(bool), z["clp"][:, 0].copy()
    if set(split.astype(str)) != {"train", "val"}:
        raise ValueError("C1 accepts the frozen train/val sidecar only")
    if not np.array_equal(index, np.arange(len(index))):
        raise ValueError("noncanonical sample indices")
    if len(set(zip(station.tolist(), utc.tolist()))) != len(index):
        raise ValueError("duplicate station/UTC keys")
    train, val = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    if (len(train), len(val)) != (15414, 1398):
        raise ValueError("changed frozen cohort")
    if not np.isin(clp[mask], (0, 1, 2)).all() or not np.all(mask.reshape(len(index), -1).any(1)):
        raise ValueError("invalid CLP/mask support")
    dates = np.array([(datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S") +
                       timedelta(hours=8)).date().isoformat() for t in utc])
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(train):
        raise ValueError("OOF manifest count")
    fold = np.full(len(index), -1, np.int64)
    for r in rows:
        i = int(r["sample_index"])
        if i not in train or fold[i] != -1 or int(r["fold"]) not in (0, 1, 2):
            raise ValueError("OOF nontrain/duplicate/invalid fold")
        if (str(station[i]), str(utc[i]), dates[i]) != (
                r["station"], r["target_time_utc"], r["target_date_bjt"]):
            raise ValueError("OOF key/date mismatch")
        if int(r["valid_pixels"]) != int(mask[i].sum()):
            raise ValueError("OOF support mismatch")
        fold[i] = int(r["fold"])
    if (fold[train] < 0).any() or (fold[val] >= 0).any():
        raise ValueError("OOF coverage")
    for date in np.unique(dates[train]):
        if len(np.unique(fold[train[dates[train] == date]])) != 1:
            raise ValueError("same BJT date crosses folds")
    train_dates, train_folds = dates[train], fold[train]
    for month in np.unique(np.asarray([d[:7] for d in train_dates])):
        month_mask = np.asarray([d[:7] == month for d in train_dates])
        month_dates = np.unique(train_dates[month_mask])
        month_position = {str(d): i for i, d in enumerate(month_dates)}
        date_to_fold = {str(d): int(f) for d, f in zip(train_dates[month_mask], train_folds[month_mask])}
        for k in range(3):
            held = [str(d) for d in month_dates if date_to_fold[str(d)] == k]
            positions = sorted(month_position[d] for d in held)
            if positions and positions[-1] - positions[0] + 1 != len(positions):
                raise ValueError("each monthly fold must be a contiguous date block")
    folds, details = [], []
    for k in range(3):
        hold = train[fold[train] == k]
        excluded = set()
        for d in np.unique(dates[hold]):
            day = datetime.strptime(str(d), "%Y-%m-%d")
            excluded.update((day + timedelta(days=v)).date().isoformat() for v in (-1, 0, 1))
        fit = train[~np.isin(dates[train], sorted(excluded))]
        if len(fit) < 0.5 * len(train) or set(station[fit]) != {"sili", "zhujia"}:
            raise ValueError("insufficient buffered fold training support")
        folds.append((fit, hold))
        details.append(dict(fold=k, fit_rows=len(fit), heldout_rows=len(hold),
                            fit_dates=len(np.unique(dates[fit])), heldout_dates=len(np.unique(dates[hold])),
                            fit_sample_indices=fit.tolist(), heldout_sample_indices=hold.tolist(),
                            train_updates=30 * math.ceil(len(fit) / 512)))
    audit = dict(state="PASS_TRAIN_DATE_FOLDS", train_rows=len(train), val_rows=len(val),
                 train_dates=len(np.unique(dates[train])), buffer_days=1, folds=details,
                 clp_sha256=sha(clp_path), fold_manifest_sha256=sha(manifest_path),
                 validation_used_to_select_threshold=False, test_payloads_read=0)
    return audit, folds, dict(train=train, val=val, station=station, utc=utc, dates=dates,
                             fold=fold, mask=mask, clp=clp)


def main():
    p = argparse.ArgumentParser()
    for key in ("clp", "folds", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    for key in ("pack", "normalization", "r-checkpoint", "r-source", "r-validation-prediction", "old-b1"):
        p.add_argument("--" + key, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, folds, d = audit_folds(args.clp, args.folds)
    args.output.mkdir(parents=True)
    dump(args.output / "fold_audit.json", audit)
    if args.audit_only:
        print(json.dumps({k: v for k, v in audit.items() if k != "folds"}))
        return
    import torch
    from torch import nn
    from train_cloud_gate import CloudGate, seed_all
    from conservative_gate_metrics import select_threshold, evaluate_threshold
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one scheduler-assigned CUDA device required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    with (args.pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["state"] == "SUPERVISED"]
    if [(r["station"], r["target_time_utc"]) for r in rows] != list(zip(d["station"], d["utc"])):
        raise ValueError("pack and sidecar keys disagree")
    if [r["split"] for r in rows] != ["train" if i in set(d["train"]) else "val" for i in range(len(rows))]:
        raise ValueError("pack split differs")
    x = np.load(args.pack / "x_raw.npy", allow_pickle=False)
    reference = np.load(args.pack / "cot.npy", allow_pickle=False)[:, 0]
    if x.shape != (len(rows), 16, 16, 16):
        raise ValueError("input shape")
    if not np.array_equal(np.load(args.pack / "mask.npy", allow_pickle=False)[:, 0].astype(bool), d["mask"]):
        raise ValueError("mask differs from frozen R support")
    norm = json.loads(args.normalization.read_text())
    mean, std = (np.asarray(norm[k], np.float32)[None, :, None, None] for k in ("mean", "std"))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid normalization")
    x = (x - mean) / std
    x[~np.isfinite(x)] = 0
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy((d["clp"] >= 1).astype(np.float32)[:, None]).to(device)
    mt = torch.from_numpy(d["mask"][:, None]).to(device)
    spec = importlib.util.spec_from_file_location("frozen_monthly_R", args.r_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = module.COTUNet().to(device)
    checkpoint = torch.load(args.r_checkpoint, map_location=device, weights_only=False)
    rmeta = checkpoint["metadata"]
    if rmeta["normalization_sha256"] != sha(args.normalization):
        raise ValueError("R checkpoint normalization binding")
    if rmeta["pack_status_sha256"] != sha(args.pack / "pack_status.json"):
        raise ValueError("R checkpoint pack binding")
    if rmeta["training_script_sha256"] != sha(args.r_source):
        raise ValueError("R checkpoint source-script binding")
    for name in ("x_raw.npy", "cot.npy", "mask.npy", "rows.csv"):
        if rmeta["pack_arrays"][name] != sha(args.pack / name):
            raise ValueError("R checkpoint pack-array binding: " + name)
    r_artifacts = json.loads((args.r_checkpoint.parent / "artifact_sha256.json").read_text())
    for name, path in (("best.pt", args.r_checkpoint),
                       ("normalization.json", args.normalization),
                       ("validation_predictions.npz", args.r_validation_prediction)):
        if r_artifacts[name] != sha(path):
            raise ValueError("R artifact-manifest hash mismatch: " + name)
    r.load_state_dict(checkpoint["model"], strict=True)
    r.eval().requires_grad_(False)
    rlog = np.empty_like(reference, dtype=np.float32)
    r_batch = int(checkpoint["metadata"]["batch"])
    if r_batch < 1:
        raise ValueError("invalid frozen R inference batch in checkpoint metadata")
    with np.load(args.r_validation_prediction, allow_pickle=False) as z:
        if not np.array_equal(z["sample_indices"], d["val"]):
            raise ValueError("saved R validation prediction indices")
        if not np.array_equal(z["reference_cot"][:, 0], reference[d["val"]]):
            raise ValueError("saved R validation references differ from frozen pack")
        if not np.array_equal(z["mask"][:, 0].astype(bool), d["mask"][d["val"]]):
            raise ValueError("saved R validation support differs from frozen pack")
        val_rlog = z["pred_log1p_cot"][:, 0].copy()
    if not np.isfinite(val_rlog).all():
        raise ValueError("nonfinite saved R validation prediction")
    with torch.no_grad():
        # The checkpoint-bound saved validation predictions are authoritative.
        # Recompute only train R outputs for OOF threshold selection; a fresh
        # GPU pass over val can differ slightly by cuDNN implementation/shape.
        for lo in range(0, len(d["train"]), r_batch):
            batch_indices = d["train"][lo:lo + r_batch]
            batch_tensor = xt[torch.as_tensor(batch_indices, device=device)]
            with torch.autocast("cuda", dtype=torch.float16):
                z = r(batch_tensor)
            rlog[batch_indices] = z.float().cpu().numpy()[:, 0]
    rlog[d["val"]] = val_rlog
    if not np.isfinite(rlog[d["train"]]).all():
        raise ValueError("nonfinite recomputed train R prediction")
    del r, checkpoint
    torch.cuda.empty_cache()
    raw_cot = np.maximum(np.expm1(rlog.astype(np.float64)), 0.0)
    dump(args.output / "source_audit.json", dict(
        R_frozen=True, R_validation_source="checkpoint-bound saved validation predictions",
        R_validation_predictions_sha256=sha(args.r_validation_prediction),
        R_train_inference_batch=r_batch,
        hashes={name: sha(path) for name, path in {
            "R": args.r_checkpoint, "R_source": args.r_source, "normalization": args.normalization,
            "pack_status": args.pack / "pack_status.json", "clp": args.clp, "folds": args.folds,
            "runner": Path(__file__), "metrics": Path(__file__).with_name("conservative_gate_metrics.py"),
            "classifier": Path(__file__).with_name("train_cloud_gate.py")}.items()},
        seed=42, epochs=30, batch_size=512, R_inference_batch=r_batch,
        model="unchanged B1 CloudGate",
        device=torch.cuda.get_device_name(0), test_payloads_read=0))

    def update(model, opt, ix):
        logits = model(xt[ix])
        error = nn.functional.binary_cross_entropy_with_logits(logits, yt[ix], reduction="none")
        masked = torch.where(mt[ix], error, torch.zeros_like(error))
        loss = (masked.sum((1, 2, 3)) / mt[ix].sum((1, 2, 3)).clamp_min(1)).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())

    seed_all(42)
    profile_model = CloudGate().to(device)
    opt = torch.optim.AdamW(profile_model.parameters(), lr=1e-3, weight_decay=1e-4)
    ix = torch.as_tensor(folds[0][0][:512], device=device)
    for _ in range(3):
        update(profile_model, opt, ix)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for _ in range(20):
        update(profile_model, opt, ix)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    dump(args.output / "profile.json", dict(batch_size=512, steps=20, seconds=seconds,
         samples_per_second=20 * 512 / seconds, peak_bytes=torch.cuda.max_memory_allocated(),
         effective_batch_fixed=True, single_gpu=True))
    del profile_model, opt
    torch.cuda.empty_cache()

    def fit_predict(name, fit, predict):
        seed_all(42)
        model = CloudGate().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
        fit_t = torch.as_tensor(fit, device=device)
        updates = 0
        start = time.monotonic()
        with (args.output / (name + "_epochs.jsonl")).open("w") as log:
            for ep in range(1, 31):
                model.train()
                perm = fit_t[torch.randperm(len(fit), device=device)]
                losses = []
                for lo in range(0, len(perm), 512):
                    losses.append(update(model, opt, perm[lo:lo + 512]))
                    updates += 1
                schedule.step()
                rec = dict(model=name, epoch=ep, train_BCE=float(np.mean(losses)),
                           updates=updates, elapsed_seconds=time.monotonic() - start)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if ep in (1, 10, 20, 30):
                    print(json.dumps(rec), flush=True)
        if updates != 30 * math.ceil(len(fit) / 512):
            raise ValueError("update count")
        torch.save(dict(model=model.state_dict(), epoch=30, seed=42, fit_sample_indices=fit,
                        selection="fixed last epoch; no heldout/validation selection"), args.output / (name + "_last.pt"))
        model.eval()
        prob = np.empty((len(predict), 16, 16), np.float32)
        with torch.no_grad():
            for lo in range(0, len(predict), 512):
                pred = torch.sigmoid(model(xt[torch.as_tensor(predict[lo:lo + 512], device=device)]))
                prob[lo:lo + len(pred)] = pred.cpu().numpy()[:, 0]
        del model, opt
        torch.cuda.empty_cache()
        return prob

    prob_oof = np.full(reference.shape, np.nan, np.float32)
    for k, (fit, hold) in enumerate(folds):
        prob_oof[hold] = fit_predict("fold" + str(k), fit, hold)
    tr, va = d["train"], d["val"]
    selection = select_threshold(prob_oof[tr], raw_cot[tr], reference[tr], d["clp"][tr],
                                 d["mask"][tr], d["station"][tr], split="train_oof")
    dump(args.output / "threshold_frozen.json", selection)
    if selection["state"] != "SELECTED":
        dump(args.output / "result.json", dict(state="STOP_NO_FEASIBLE_TRAIN_OOF_THRESHOLD",
                                             selection=selection, test_payloads_read=0))
        return
    # Full-data fit and all validation comparisons occur only after threshold freeze.
    pval = fit_predict("full_train", tr, va)
    with np.load(args.old_b1, allow_pickle=False) as z:
        for key, expected in (("sample_indices", va), ("station", d["station"][va]),
                              ("target_time_utc", d["utc"][va]), ("valid", d["mask"][va])):
            if not np.array_equal(z[key], expected):
                raise ValueError("old B1 comparison key/support mismatch: " + key)
        old_prob = z["prob_cloud"].copy()
    np.savez_compressed(args.output / "train_oof_predictions.npz",
        prob_cloud=prob_oof[tr], raw_cot=raw_cot[tr], reference_cot=reference[tr],
        clp=d["clp"][tr], valid=d["mask"][tr], station=d["station"][tr], sample_indices=tr,
        target_time_utc=d["utc"][tr], fold=d["fold"][tr])
    np.savez_compressed(args.output / "validation_predictions.npz",
        prob_cloud=pval, old_B1_prob_cloud=old_prob, raw_cot=raw_cot[va],
        reference_cot=reference[va], clp=d["clp"][va], valid=d["mask"][va],
        station=d["station"][va], sample_indices=va, target_time_utc=d["utc"][va])
    comparisons = {}
    for name, prob, threshold in (("old_B1_0p5", old_prob, .5),
                                  ("full_last_0p5", pval, .5),
                                  ("full_last_conservative", pval, selection["threshold"])):
        comparisons[name] = evaluate_threshold(prob, raw_cot[va], reference[va], d["clp"][va],
                  d["mask"][va], d["station"][va], threshold=threshold, split="validation")
    dump(args.output / "result.json", dict(state="COMPLETE_C1_FIXED_TRAINING_AND_VALIDATION",
         threshold=selection["threshold"], comparisons=comparisons,
         output_hashes={f.name: sha(f) for f in args.output.glob("*") if f.is_file()},
         test_payloads_read=0, GHI_training_started=False))
    print("C1_FIXED_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
