"""Staged COT pilot: fit a near-zero/cloud classifier, then class regressors.

Uses only the verified real-AGRI train/validation pack. The frozen R feature
extractor and the selected classifier are frozen before conditional regression.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_cot_repaired import COTUNet


FEATURES = [f"C{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)]
FEATURES += ["cosSOZ", "cosRAA", "day_mask"]
THRESHOLD = 0.1
LOG_THRESHOLD = math.log1p(THRESHOLD)
BASELINE = {
    "clear_cot_lt0p1_rmse": 0.7711762542248346,
    "cot_rmse": 4.246164397624913,
    "high_cot_ge30_rmse": 11.949837608746112,
    "station_5x5_rmse": 3.943565921165888,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


class CloudClassifier(nn.Module):
    def __init__(self, cloud_prior):
        super().__init__()
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.fill_(math.log(cloud_prior / (1 - cloud_prior)))

    def forward(self, features):
        return self.head(features)


class ConditionalRegressors(nn.Module):
    def __init__(self, frozen_r_head):
        super().__init__()
        self.low_head = nn.Conv2d(16, 1, 1)
        self.cloud_head = nn.Conv2d(16, 1, 1)
        with torch.no_grad():
            self.low_head.weight.copy_(frozen_r_head.weight)
            self.low_head.bias.copy_(frozen_r_head.bias)
            self.cloud_head.weight.copy_(frozen_r_head.weight)
            self.cloud_head.bias.copy_(frozen_r_head.bias)

    def forward(self, features):
        low_log = LOG_THRESHOLD * torch.sigmoid(self.low_head(features))
        cloud_log = LOG_THRESHOLD + F.softplus(self.cloud_head(features))
        return torch.expm1(low_log), torch.expm1(cloud_log), low_log, cloud_log


def evaluate_arrays(pred, prob, ref, mask):
    error = pred - ref
    low = ref < THRESHOLD
    cloud = ~low
    thick = ref >= 30.0
    center = np.zeros((1, 1, 16, 16), dtype=bool)
    center[:, :, 6:11, 6:11] = True
    region = np.broadcast_to(center, mask.shape)
    groups = {"all": mask, "clear_cot_lt0p1": mask & low,
              "high_cot_ge30": mask & thick, "station_5x5": mask & region}
    counts = {name: int(which.sum()) for name, which in groups.items()}
    if not all(counts.values()):
        raise RuntimeError("empty validation stratum")
    result = {}
    for name, which in groups.items():
        values = error[which]
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        result[f"{name}_bias"] = float(np.mean(values, dtype=np.float64))
    pred_cloud = prob >= 0.5
    result.update({
        "near_zero_false_cloud_fraction": float((pred_cloud & mask & low).sum() / counts["clear_cot_lt0p1"]),
        "near_zero_predicted_ge0p1_fraction": float(((pred >= THRESHOLD) & mask & low).sum() / counts["clear_cot_lt0p1"]),
        "cloud_class_accuracy": float(((pred_cloud == cloud) & mask).sum() / counts["all"]),
        "cloud_class_precision": float(((pred_cloud & cloud) & mask).sum() / max(1, (pred_cloud & mask).sum())),
        "cloud_class_recall": float(((pred_cloud & cloud) & mask).sum() / max(1, (cloud & mask).sum())),
        "cloud_class_brier": float(np.square(prob[mask] - cloud[mask]).mean(dtype=np.float64)),
        "valid_pixels": counts,
    })
    result["balanced_relative_rmse"] = float(np.mean([
        result["clear_cot_lt0p1_rmse"] / BASELINE["clear_cot_lt0p1_rmse"],
        result["all_rmse"] / BASELINE["cot_rmse"],
        result["high_cot_ge30_rmse"] / BASELINE["high_cot_ge30_rmse"],
        result["station_5x5_rmse"] / BASELINE["station_5x5_rmse"],
    ]))
    return result


@torch.no_grad()
def classifier_probs(classifier, features, indexes, batch):
    classifier.eval()
    out = []
    for ids in indexes.split(batch):
        out.append(torch.sigmoid(classifier(features[ids]).float()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate_full(classifier, regressors, features, target, valid, indexes, batch, save_path=None):
    classifier.eval()
    regressors.eval()
    pred_rows, prob_rows = [], []
    for ids in indexes.split(batch):
        feat = features[ids]
        prob = torch.sigmoid(classifier(feat).float())
        low_cot, cloud_cot, _, _ = regressors(feat)
        pred = torch.where(prob >= 0.5, cloud_cot, low_cot).float()
        pred_rows.append(pred.cpu().numpy())
        prob_rows.append(prob.cpu().numpy())
    pred = np.concatenate(pred_rows).astype(np.float32)
    prob = np.concatenate(prob_rows).astype(np.float32)
    row_ids = indexes.cpu().numpy()
    ref, mask = target[row_ids], valid[row_ids]
    metrics = evaluate_arrays(pred, prob, ref, mask)
    if save_path is not None:
        np.savez_compressed(save_path, row_indices=row_ids, pred_cot=pred, cloud_probability=prob,
                            reference_cot=ref.astype(np.float32), mask=mask)
    return metrics


def cpu_smoke():
    torch.manual_seed(42)
    backbone = COTUNet().cpu().eval()
    x = torch.randn(2, 16, 16, 16)
    with torch.no_grad():
        features = backbone.forward_features(x)
    classifier = CloudClassifier(0.6).cpu().train()
    regressors = ConditionalRegressors(backbone.head).cpu().train()
    logits = classifier(features)
    low, cloud, low_log, cloud_log = regressors(features)
    assert logits.shape == low.shape == cloud.shape == (2, 1, 16, 16)
    assert bool(((low >= 0) & (low <= THRESHOLD)).all()) and bool((cloud >= THRESHOLD).all())
    assert torch.allclose(torch.log1p(low), low_log, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.log1p(cloud), cloud_log, rtol=1e-5, atol=1e-6)
    target = torch.rand_like(low) * 4.0
    valid = torch.ones_like(low, dtype=torch.bool)
    loss_cls = masked_mean(F.binary_cross_entropy_with_logits(
        logits, (target >= THRESHOLD).float(), reduction="none"), valid)
    loss_reg = masked_mean(F.huber_loss(torch.log1p(cloud), torch.log1p(target), reduction="none"), valid)
    (loss_cls + loss_reg).backward()
    assert torch.isfinite(loss_cls + loss_reg)
    print("COT_STAGED_HURDLE_CPU_SMOKE_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--regressor-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    if args.root is None or args.output is None:
        parser.error("--root and --output are required for training")
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite staged hurdle pilot output")
    if not (1 <= args.epochs <= 50 and 1 <= args.batch <= 4096):
        raise ValueError("pilot hyperparameter outside preregistered range")
    pack = root / "data/cot_repaired_pack_20260907_v1"
    base = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    if sha(base["checkpoint"]) != base["checkpoint_sha256"]:
        raise RuntimeError("frozen R checkpoint hash mismatch")
    checkpoint = torch.load(base["checkpoint"], map_location="cpu", weights_only=False)
    norm = json.loads((pack / "norm.json").read_text())
    if norm != checkpoint["metadata"]["norm"] or norm["feature_names"] != FEATURES:
        raise RuntimeError("frozen R normalization/channel contract mismatch")
    pack_status = json.loads((pack / "pack_status.json").read_text())
    if pack_status["state"] != "COMPLETE" or pack_status["test_payloads_read"] != 0:
        raise RuntimeError("train/validation pack incomplete or test payload accessed")
    pack_hashes = json.loads((pack / "SHA256.json").read_text())
    for name in ("x_raw.npy", "target.npy", "mask.npy", "rows.csv", "norm.json"):
        if sha(pack / name) != pack_hashes[name]:
            raise RuntimeError("COT pack SHA mismatch: " + name)
    with (pack / "rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 18514 or any(row["split"] not in ("train", "validation") for row in rows):
        raise RuntimeError("unexpected split; test rows are forbidden")
    train_index = np.array([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    val_index = np.array([i for i, row in enumerate(rows) if row["split"] == "validation"], dtype=np.int64)
    if len(train_index) != 11978 or len(val_index) != 6536:
        raise RuntimeError("train/validation sample counts drifted")
    raw = np.load(pack / "x_raw.npy")
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    x_np = (raw - mean) / std
    nonfinite = int((~np.isfinite(x_np)).sum())
    x_np[~np.isfinite(x_np)] = 0
    target = np.load(pack / "target.npy") * np.float32(100)
    valid = np.load(pack / "mask.npy")
    if target.shape != (18514, 1, 16, 16) or valid.shape != target.shape:
        raise RuntimeError("target/mask shape mismatch")
    if not np.isfinite(target).all() or (target < 0).any() or (target > 100).any():
        raise RuntimeError("invalid CPP COT target")
    if not valid.reshape(len(valid), -1).any(axis=1).all():
        raise RuntimeError("empty valid mask")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("PBS GPU not visible")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x = torch.from_numpy(x_np).to(device)
    y_cot = torch.from_numpy(target).to(device)
    y_log = torch.log1p(y_cot)
    mask = torch.from_numpy(valid).to(device)
    train_ids = torch.from_numpy(train_index).to(device)
    val_ids = torch.from_numpy(val_index).to(device)
    train_valid = valid[train_index]
    train_cloud = target[train_index] >= THRESHOLD
    cloud_fraction = float((train_cloud & train_valid).sum() / train_valid.sum())
    if not 0 < cloud_fraction < 1:
        raise RuntimeError("training split lacks one COT class")
    output.mkdir(parents=True)
    contract = {
        "state": "PILOT_RUNNING", "test_used": False, "seed": args.seed,
        "source_script_sha256": sha(__file__), "frozen_checkpoint_sha256": base["checkpoint_sha256"],
        "pack_hashes": pack_hashes, "train_rows": len(train_index), "validation_rows": len(val_index),
        "train_valid_pixels": int(train_valid.sum()), "train_cloud_fraction": cloud_fraction,
        "nonfinite_input_values_imputed": nonfinite, "class_threshold_cpp_cot": THRESHOLD,
        "threshold_definition": "CPP COT<0.1 is near-zero; COT>=0.1 is cloud class, not a meteorological clear-sky label",
        "stages": ["freeze original R feature extractor and train classifier on train labels only",
                   "freeze selected classifier and feature extractor; train separate low/cloud regressors on matching train classes"],
        "inference": "hard class decision at fixed p_cloud>=0.5 selects low or cloud regression head",
        "regressor_scale": "log1p(COT); low output constrained [0,log1p(0.1)]; cloud output constrained [log1p(0.1),infinity)",
        "classifier_epochs": args.epochs, "regressor_epochs": args.epochs, "batch": args.batch,
        "classifier_optimizer": f"AdamW lr={args.classifier_lr} weight_decay=1e-4; masked natural-frequency BCE",
        "regressor_optimizer": f"AdamW lr={args.regressor_lr} weight_decay=1e-4; equal-class masked Huber delta=0.1",
        "selection": "classifier: lowest validation Brier; regressors: same balanced four-stratum normalized RMSE for final hard-gated prediction",
        "retention_limits": BASELINE,
        "input_domain": "observed real AGRI train/validation only; no forecast AGRI",
        "gpu": torch.cuda.get_device_name(0), "pbs_job_id": os.environ.get("PBS_JOBID"),
    }
    save_json(output / "contract.json", contract)

    # Recompute the frozen-R baseline on the exact validation cohort before training.
    backbone = COTUNet().to(device).eval()
    backbone.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        frozen_pred_rows = []
        for ids in val_ids.split(args.batch):
            frozen_pred_rows.append(torch.expm1(backbone(x[ids]).float()).cpu().numpy())
        frozen_pred = np.concatenate(frozen_pred_rows)
    base_metrics = evaluate_arrays(frozen_pred, (frozen_pred >= THRESHOLD).astype(np.float32),
                                   target[val_index], valid[val_index])
    baseline_names = {"clear_cot_lt0p1_rmse": "clear_cot_lt0p1_rmse", "cot_rmse": "all_rmse",
                      "high_cot_ge30_rmse": "high_cot_ge30_rmse", "station_5x5_rmse": "station_5x5_rmse"}
    for key, expected in BASELINE.items():
        if abs(base_metrics[baseline_names[key]] - expected) > 2e-4:
            raise RuntimeError(f"frozen R baseline drift in {key}")
    np.savez_compressed(output / "frozen_R_validation_predictions.npz", row_indices=val_index,
                        pred_cot=frozen_pred.astype(np.float32), reference_cot=target[val_index],
                        mask=valid[val_index])
    save_json(output / "frozen_R_metrics.json", base_metrics)

    # Cache frozen R features once; no gradient reaches the retrieval backbone.
    feature_rows = []
    with torch.no_grad():
        for ids in torch.arange(len(rows), device=device).split(args.batch):
            feature_rows.append(backbone.forward_features(x[ids]).float())
    features = torch.cat(feature_rows)
    frozen_head = backbone.head.cpu()
    del backbone, x, x_np, raw, feature_rows
    torch.cuda.empty_cache()

    # Stage 1: train the classifier independently; select by proper Brier score.
    seed_all(args.seed)
    classifier = CloudClassifier(cloud_fraction).to(device)
    classifier_opt = torch.optim.AdamW(classifier.parameters(), lr=args.classifier_lr, weight_decay=1e-4)
    cloud_target = (y_cot >= THRESHOLD).to(torch.float32)
    val_cloud_np = (target[val_index] >= THRESHOLD)
    val_mask_np = valid[val_index]
    best_brier, best_classifier_state, best_classifier_epoch = float("inf"), None, -1
    clf_dir = output / "stage1_classifier"
    clf_dir.mkdir()
    torch.save({"state_dict": classifier.state_dict(), "epoch": -1}, clf_dir / "initial.pt")
    classifier_epochs = []
    for epoch in range(args.epochs):
        classifier.train()
        permutation = train_ids[torch.randperm(len(train_ids), device=device)]
        loss_sum, seen = 0.0, 0
        started = time.monotonic()
        for ids in permutation.split(args.batch):
            classifier_opt.zero_grad(set_to_none=True)
            logits = classifier(features[ids]).float()
            values = F.binary_cross_entropy_with_logits(logits, cloud_target[ids], reduction="none")
            loss = masked_mean(values, mask[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0, error_if_nonfinite=True)
            classifier_opt.step()
            n = int(mask[ids].sum())
            loss_sum += float(loss.detach()) * n
            seen += n
        probs = classifier_probs(classifier, features, val_ids, args.batch)
        brier = float(np.square(probs - val_cloud_np).flatten()[val_mask_np.flatten()].mean(dtype=np.float64))
        if brier < best_brier:
            best_brier, best_classifier_epoch = brier, epoch
            best_classifier_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            torch.save({"state_dict": best_classifier_state, "epoch": epoch, "brier": brier}, clf_dir / "best.pt")
            np.savez_compressed(clf_dir / "best_validation_probabilities.npz", row_indices=val_index,
                                cloud_probability=probs.astype(np.float32), cloud_reference=val_cloud_np,
                                mask=val_mask_np)
        row = {"epoch": epoch, "train_bce": loss_sum / max(1, seen), "validation_brier": brier,
               "best_epoch": best_classifier_epoch, "seconds": time.monotonic() - started}
        classifier_epochs.append(row)
        with (clf_dir / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print("CLASSIFIER_EPOCH", json.dumps(row), flush=True)
    classifier.load_state_dict(best_classifier_state, strict=True)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    classifier.eval()
    del classifier_opt

    # Stage 2: regress each class separately with frozen classifier and features.
    seed_all(args.seed)
    regressors = ConditionalRegressors(frozen_head).to(device)
    reg_opt = torch.optim.AdamW(regressors.parameters(), lr=args.regressor_lr, weight_decay=1e-4)
    low_class = y_cot < THRESHOLD
    cloud_class = ~low_class
    val_probs = classifier_probs(classifier, features, val_ids, args.batch)
    fixed_prob_by_row = torch.zeros((len(rows), 1, 16, 16), dtype=torch.float32, device=device)
    fixed_prob_by_row[val_ids] = torch.from_numpy(val_probs).to(device)
    reg_dir = output / "stage2_conditional_regressors"
    reg_dir.mkdir()
    initial_metrics = evaluate_full(classifier, regressors, features, target, valid, val_ids,
                                   args.batch, reg_dir / "initial_validation_predictions.npz")
    save_json(reg_dir / "initial_metrics.json", initial_metrics)
    torch.save({"state_dict": regressors.state_dict(), "epoch": -1,
                "validation_metrics": initial_metrics}, reg_dir / "initial.pt")
    best_score, best_metrics, best_epoch = initial_metrics["balanced_relative_rmse"], initial_metrics, -1
    for epoch in range(args.epochs):
        regressors.train()
        permutation = train_ids[torch.randperm(len(train_ids), device=device)]
        loss_sum, seen = 0.0, 0
        started = time.monotonic()
        for ids in permutation.split(args.batch):
            feat = features[ids]
            low_cot, cloud_cot, low_log, cloud_log = regressors(feat)
            low_loss = masked_mean(F.huber_loss(low_log.float(), y_log[ids], reduction="none", delta=0.1),
                                   mask[ids] & low_class[ids])
            cloud_loss = masked_mean(F.huber_loss(cloud_log.float(), y_log[ids], reduction="none", delta=0.1),
                                     mask[ids] & cloud_class[ids])
            loss = 0.5 * (low_loss + cloud_loss)
            reg_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regressors.parameters(), 5.0, error_if_nonfinite=True)
            reg_opt.step()
            n = int(mask[ids].sum())
            loss_sum += float(loss.detach()) * n
            seen += n
        metrics = evaluate_full(classifier, regressors, features, target, valid, val_ids, args.batch)
        score = metrics["balanced_relative_rmse"]
        if score < best_score:
            best_score, best_metrics, best_epoch = score, metrics, epoch
            torch.save({"state_dict": regressors.state_dict(), "epoch": epoch,
                        "validation_metrics": metrics}, reg_dir / "best.pt")
            evaluate_full(classifier, regressors, features, target, valid, val_ids, args.batch,
                          reg_dir / "best_validation_predictions.npz")
        row = {"epoch": epoch, "train_conditional_huber": loss_sum / max(1, seen),
               **metrics, "best_epoch": best_epoch, "seconds": time.monotonic() - started}
        with (reg_dir / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print("REGRESSOR_EPOCH", json.dumps(row), flush=True)
    save_json(reg_dir / "best_metrics.json", {"best_epoch": best_epoch, **best_metrics})

    retention = {
        "clear_cot_lt0p1_rmse": best_metrics["clear_cot_lt0p1_rmse"] <= BASELINE["clear_cot_lt0p1_rmse"],
        "cot_rmse": best_metrics["all_rmse"] <= BASELINE["cot_rmse"],
        "high_cot_ge30_rmse": best_metrics["high_cot_ge30_rmse"] <= BASELINE["high_cot_ge30_rmse"],
        "station_5x5_rmse": best_metrics["station_5x5_rmse"] <= BASELINE["station_5x5_rmse"],
    }
    result = {
        "state": "COMPLETE_STAGED_VALIDATION_PILOT", "test_used": False,
        "classifier_best_epoch": best_classifier_epoch, "classifier_best_validation_brier": best_brier,
        "frozen_R": base_metrics, "staged_candidate": {"best_epoch": best_epoch, **best_metrics},
        "candidate_passes_all_retention_limits": all(retention.values()) and
            best_metrics["clear_cot_lt0p1_rmse"] < BASELINE["clear_cot_lt0p1_rmse"],
        "candidate_retention_checks_vs_frozen_R": retention,
        "note": "Validation-only staged classifier then class regressors; real-AGRI COT retrieval against CPP reference; no GHI claim and no test used.",
    }
    save_json(output / "complete.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
