#!/usr/bin/env python3
"""Train an independent Hubei CLP classifier and date-grouped train OOF bank.

Input normalization is reconstructed from the audited train-only R input pack;
fold-specific normalization is refit on each fold's training dates. The model
is randomly initialized in Hubei and never loads Hunan weights.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class CloudClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def align_clp_sidecar_to_rows(rows, sidecar_keys, sidecar_splits, clp, label_valid):
    row_keys = np.asarray([r["time_bjt"] for r in rows], dtype=str)
    row_splits = np.asarray([r["split"] for r in rows], dtype=str)
    sidecar_keys = np.asarray(sidecar_keys, dtype=str)
    sidecar_splits = np.asarray(sidecar_splits, dtype=str)
    clp = np.asarray(clp, dtype=np.uint8)
    label_valid = np.asarray(label_valid, dtype=bool)
    if len(set(row_keys.tolist())) != len(row_keys):
        raise ValueError("duplicate BJT keys in R pack rows")
    if len(set(sidecar_keys.tolist())) != len(sidecar_keys):
        raise ValueError("duplicate BJT keys in CPP/CLP sidecar")
    if len(sidecar_keys) != len(row_keys):
        raise ValueError("CPP/CLP sidecar and R pack row counts differ")
    if set(sidecar_keys.tolist()) != set(row_keys.tolist()):
        missing = sorted(set(row_keys.tolist()) - set(sidecar_keys.tolist()))
        extra = sorted(set(sidecar_keys.tolist()) - set(row_keys.tolist()))
        raise ValueError(f"CPP/CLP key-set mismatch: missing={len(missing)} extra={len(extra)}")
    sidecar_index = {key: i for i, key in enumerate(sidecar_keys.tolist())}
    sidecar_order = np.asarray([sidecar_index[key] for key in row_keys.tolist()], dtype=np.int64)
    aligned_splits = sidecar_splits[sidecar_order]
    if not np.array_equal(aligned_splits, row_splits):
        bad = np.flatnonzero(aligned_splits != row_splits)
        i = int(bad[0])
        raise ValueError(f"CPP/CLP keyed split mismatch at {row_keys[i]}: {aligned_splits[i]} != {row_splits[i]}")
    if clp.shape[0] != len(sidecar_keys) or label_valid.shape != clp.shape:
        raise ValueError("CLP label shape mismatch")
    # Different source files may use different row order. Join on the unique
    # BJT key and return arrays in exactly the R-pack order used by xraw.
    reordered = int(np.count_nonzero(sidecar_order != np.arange(len(row_keys))))
    return row_keys, row_splits, clp[sidecar_order], label_valid[sidecar_order], reordered


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def masked_loss(logits, target, valid):
    e = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    e = torch.where(valid, e, torch.zeros_like(e))
    denom = valid.sum(dim=(1, 2, 3)).clamp_min(1)
    return (e.sum(dim=(1, 2, 3)) / denom).mean()


def score(prob, y, valid):
    p = np.asarray(prob, dtype=np.float64)[valid]
    yy = np.asarray(y, dtype=bool)[valid]
    if not len(p):
        return {"n": 0}
    pred = p >= 0.5
    tp = int(np.count_nonzero(pred & yy)); tn = int(np.count_nonzero(~pred & ~yy))
    fp = int(np.count_nonzero(pred & ~yy)); fn = int(np.count_nonzero(~pred & yy))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {"n": len(p), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "POD": pod, "FAR": fp / (tp + fp) if tp + fp else float("nan"),
            "CSI": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "KSS": pod + tnr - 1 if np.isfinite(pod) and np.isfinite(tnr) else float("nan"),
            "Brier": float(np.mean((p - yy.astype(np.float64)) ** 2))}


def fit_stats(xraw, idx):
    sub = np.asarray(xraw[idx], dtype=np.float64)
    mean = sub.mean(axis=(0, 2, 3))
    std = sub.std(axis=(0, 2, 3))
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def train_model(xraw, y, valid, train_ids, device, seed, epochs, batch, outdir, tag, mean=None, std=None):
    if len(train_ids) == 0:
        raise ValueError(f"empty training subset for {tag}")
    if mean is None:
        mean, std = fit_stats(xraw, train_ids)
    # CPU normalization to keep only the active train split on GPU.
    xn = (xraw - mean[None, :, None, None]) / std[None, :, None, None]
    xn[~np.isfinite(xn)] = 0
    tx = torch.as_tensor(xn, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y[:, None], dtype=torch.float32, device=device)
    tv = torch.as_tensor(valid[:, None], dtype=torch.bool, device=device)
    ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    seed_all(seed)
    model = CloudClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = ids[torch.randperm(len(ids), device=device)]
        total, n = 0.0, 0
        for b in perm.split(batch):
            loss = masked_loss(model(tx[b]), ty[b], tv[b])
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite BCE in {tag} epoch {ep}")
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step(); total += float(loss.detach()); n += 1
        sched.step()
        record = {"epoch": ep, "train_bce_mean_per_sample": total / max(n, 1), "lr": opt.param_groups[0]["lr"]}
        history.append(record)
        if ep in (1, 5, 10, 20, 30, epochs):
            print(f"{tag} {json.dumps(record)}", flush=True)
    model.eval()
    with torch.no_grad():
        probs = []
        all_ids = torch.arange(len(xraw), dtype=torch.long, device=device)
        for b in all_ids.split(batch):
            probs.append(torch.sigmoid(model(tx[b])).cpu().numpy()[:, 0])
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs,
                "mean": mean, "std": std, "architecture": "Conv2d 16->16->32->16->1, 3x3 SiLU x3, 1x1 logit"},
               outdir / f"{tag}.pt")
    (outdir / f"{tag}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    del tx, ty, tv, ids, model, opt, sched
    torch.cuda.empty_cache()
    return np.concatenate(probs, axis=0).astype(np.float32), mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, type=Path, help="train/val Hubei R pack; input features only")
    p.add_argument("--cpp-sidecar", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--cossoz-min", type=float, default=0.17364817766693041)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hubei classifier training requires allocated CUDA; no CPU fallback")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"refusing nonempty output {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    status = json.loads((a.pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") not in ("HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED",
                                      "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED"):
        raise ValueError("unexpected Hubei input pack status")
    norm = json.loads((a.pack / "norm.json").read_text(encoding="utf-8"))
    if norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("input normalization is not train-only")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8-sig")))
    if not rows or any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("classifier input rows must be train/val only")
    x_norm = np.load(a.pack / "x_norm.npy", mmap_mode="r", allow_pickle=False)
    if x_norm.shape != (len(rows), 16, 16, 16) or not np.isfinite(x_norm).all():
        raise ValueError("unexpected classifier x_norm shape or nonfinite values")
    old_mean = np.asarray(norm["mean"], dtype=np.float32)
    old_std = np.asarray(norm["std"], dtype=np.float32)
    xraw = np.asarray(x_norm, dtype=np.float32) * old_std[None, :, None, None] + old_mean[None, :, None, None]
    xraw[~np.isfinite(xraw)] = 0
    with np.load(a.cpp_sidecar, allow_pickle=False) as z:
        keys, splits, clp, label_valid, reordered_sidecar_rows = align_clp_sidecar_to_rows(
            rows, z["time_bjt"], z["split"], z["clp"], z["clp_valid_mask"])
    if clp.shape != (len(rows), 16, 16) or label_valid.shape != clp.shape:
        raise ValueError("CLP label shape mismatch")
    soz = xraw[:, 13]
    day = xraw[:, 15]
    if np.any((day < -1e-3) | (day > 1.001)) or np.any(np.abs(soz) > 1.01):
        raise ValueError("decoded AGRI solar channels are outside expected physical ranges")
    valid = label_valid & (day > 0.5) & (soz > a.cossoz_min)
    y = (clp >= 1).astype(np.float32)
    train = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"], dtype=np.int64)
    val = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val"], dtype=np.int64)
    train_labeled = train[np.any(valid[train], axis=(1, 2))]
    if len(train_labeled) < 50 or not len(val):
        raise ValueError("insufficient labeled train/val support")
    train_dates = np.asarray([rows[i]["time_bjt"][:8] for i in train], dtype=str)
    unique_dates = np.unique(train_dates)
    if len(unique_dates) < a.folds:
        raise ValueError("not enough distinct train dates for date-grouped OOF")
    rng = np.random.default_rng(a.seed)
    shuffled = unique_dates[rng.permutation(len(unique_dates))]
    fold_date_groups = [np.asarray(x) for x in np.array_split(shuffled, a.folds)]
    oof = np.full((len(rows), 16, 16), np.nan, dtype=np.float32)
    fold_receipts = []
    start = time.monotonic()
    device = torch.device("cuda:0")
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = True

    # Date-held-out probabilities for every train key. Folds are grouped by
    # calendar date; preprocessing statistics are refit without the held dates.
    for fold, held_dates in enumerate(fold_date_groups):
        is_held = (np.asarray([rows[i]["time_bjt"][:8] for i in range(len(rows))])[:, None] == held_dates[None, :]).any(axis=1)
        held = np.flatnonzero(is_held & (np.arange(len(rows)) < len(rows)))
        held = np.intersect1d(held, train)
        fold_train_rows = train[~np.isin(train, held)]
        fit_ids = train_labeled[~np.isin(train_labeled, held)]
        if len(held) == 0 or len(fit_ids) == 0 or len(fold_train_rows) == 0:
            raise ValueError(f"empty date fold {fold}")
        mean, std = fit_stats(xraw, fold_train_rows)
        fold_dir = a.output / "folds"
        probs, _, _ = train_model(xraw, y, valid, fit_ids, device, a.seed + fold,
                                  a.epochs, a.batch, fold_dir, f"fold{fold}", mean, std)
        oof[held] = probs[held]
        fold_receipts.append({"fold": fold, "held_dates": held_dates.tolist(),
                              "held_train_rows": int(len(held)), "labeled_fit_rows": int(len(fit_ids)),
                              "held_valid_clp_pixels": int(valid[held].sum())})

    # Full-train model used only for val and rows without a held-out OOF model.
    mean_full, std_full = fit_stats(xraw, train)
    full_probs, mean_full, std_full = train_model(xraw, y, valid, train_labeled, device,
                                                   a.seed, a.epochs, a.batch, a.output,
                                                   "full_train", mean_full, std_full)
    oof[val] = full_probs[val]
    # Train rows absent from any OOF fold are forbidden rather than silently
    # filled from an in-sample model; the fixed split should make this empty.
    if not np.isfinite(oof[train]).all() or not np.isfinite(oof[val]).all():
        raise ValueError("OOF/full-train bank has unfilled train/validation keys")

    train_metrics = score(oof[train], y[train], valid[train])
    val_metrics = score(oof[val], y[val], valid[val])
    np.savez_compressed(a.output / "cloud_probability_trainval.npz",
                        time_bjt=keys, split=splits, p_cloud=oof.astype(np.float32),
                        p_cloud_full_train=full_probs.astype(np.float32),
                        clp=clp, valid=valid.astype(np.uint8),
                        test_payloads_read=np.asarray(0, dtype=np.int8))
    report = {
        "state": "COMPLETE_HUBEI_CLOUD_CLASSIFIER_OOF_VALIDATION_ONLY_TEST_CLOSED",
        "model": "Hubei random initialization; Conv2d 16->16->32->16->1; 3x3 SiLU x3; 1x1 logit",
        "source_pack_status": status.get("state"), "source_pack_status_sha256": sha(a.pack / "pack_status.json"),
        "source_x_norm_sha256": sha(a.pack / "x_norm.npy"), "train_only_normalization_sha256": sha(a.pack / "norm.json"),
        "correct_utc_cpp_sidecar_sha256": sha(a.cpp_sidecar),
        "row_key_alignment": "exact unique BJT key-set join; keyed split equality; CLP labels reordered to R-pack row order",
        "sidecar_rows_reordered": reordered_sidecar_rows,
        "rows": len(rows), "train_rows": len(train), "validation_rows": len(val),
        "train_labeled_rows": int(len(train_labeled)), "folds": fold_receipts,
        "seed": a.seed, "epochs_per_model": a.epochs, "batch": a.batch,
        "OOF_metric_scope": "train CLP-labeled pixels only; date held-out predictions",
        "validation_metric_scope": "validation CLP-labeled pixels; fixed full-train model; no val-based checkpoint selection",
        "threshold_for_diagnostic_metrics_only": 0.5,
        "train_metrics": train_metrics, "validation_metrics": val_metrics,
        "class_counts_train": {str(k): int(np.count_nonzero(valid[train] & (clp[train] == k))) for k in (0, 1, 2)},
        "class_counts_val": {str(k): int(np.count_nonzero(valid[val] & (clp[val] == k))) for k in (0, 1, 2)},
        "normalization": "fold statistics fitted on each fold's train inputs; full model statistics fit on all train inputs only",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "elapsed_seconds": time.monotonic() - start,
        "reference_caveat": "CPP CLP is a retrieval reference, not independent cloud truth",
    }
    (a.output / "classifier_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
