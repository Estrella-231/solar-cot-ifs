"""Train a near-zero/cloud classifier with class-conditional COT regressors.

Only the verified real-AGRI train/validation pack is read. The frozen original
R is retained as the initialization and validation benchmark; outputs are
written to a new, non-overwriting pilot directory.
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
BASELINE = {
    "clear_cot_lt0p1_rmse": 0.7711762542248346,
    "cot_rmse": 4.246164397624913,
    "high_cot_ge30_rmse": 11.949837608746112,
    "station_5x5_rmse": 3.943565921165888,
}
STRATUM = 0.1
LOG_STRATUM = math.log1p(STRATUM)


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


def masked_huber(pred, target, mask):
    values = F.huber_loss(pred.float(), target.float(), delta=0.1, reduction="none")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


class HurdleCOT(nn.Module):
    """Shared COT U-Net features plus class gate and low/high COT experts."""
    def __init__(self, initial_state, cloud_prior):
        super().__init__()
        self.backbone = COTUNet()
        self.backbone.load_state_dict(initial_state, strict=True)
        old_head = self.backbone.head
        self.low_head = nn.Conv2d(16, 1, 1)
        self.cloud_head = nn.Conv2d(16, 1, 1)
        self.class_head = nn.Conv2d(16, 1, 1)
        with torch.no_grad():
            self.low_head.weight.copy_(old_head.weight)
            self.low_head.bias.copy_(old_head.bias)
            self.cloud_head.weight.copy_(old_head.weight)
            self.cloud_head.bias.copy_(old_head.bias)
            nn.init.zeros_(self.class_head.weight)
            prior_logit = math.log(cloud_prior / (1.0 - cloud_prior))
            self.class_head.bias.fill_(prior_logit)
        self.backbone.head = nn.Identity()

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cloud_logit = self.class_head(features)
        cloud_prob = torch.sigmoid(cloud_logit)
        # Frozen R predicts log1p(COT) before Softplus. Keep the conditional
        # experts in that scale so copied R weights retain their meaning.
        low_log = LOG_STRATUM * torch.sigmoid(self.low_head(features))
        cloud_log = LOG_STRATUM + F.softplus(self.cloud_head(features))
        low_cot = torch.expm1(low_log)
        cloud_cot = torch.expm1(cloud_log)
        expected_cot = (1.0 - cloud_prob) * low_cot + cloud_prob * cloud_cot
        return cloud_logit, cloud_prob, low_cot, cloud_cot, expected_cot


def hurdle_loss(model, x, target_log, target_cot, valid, class_weight):
    logits, _, low_cot, cloud_cot, _ = model(x)
    cloud = target_cot >= STRATUM
    low = ~cloud
    class_values = F.binary_cross_entropy_with_logits(
        logits.float(), cloud.to(logits.dtype), reduction="none")
    class_mask = valid.to(class_values.dtype)
    class_loss = (class_values * class_mask).sum() / class_mask.sum().clamp_min(1)
    low_mask = valid & low
    cloud_mask = valid & cloud
    low_loss = masked_huber(torch.log1p(low_cot), target_log, low_mask)
    cloud_loss = masked_huber(torch.log1p(cloud_cot), target_log, cloud_mask)
    # Give each conditional regressor equal influence despite class imbalance.
    regression_loss = 0.5 * (low_loss + cloud_loss)
    return regression_loss + class_weight * class_loss, {
        "loss_class": float(class_loss.detach()),
        "loss_low_regression": float(low_loss.detach()),
        "loss_cloud_regression": float(cloud_loss.detach()),
    }


@torch.no_grad()
def predict(model, x, indexes, batch, hurdle):
    model.eval()
    predictions, probabilities = [], []
    for ids in indexes.split(batch):
        if hurdle:
            _, prob, _, _, pred = model(x[ids])
            probabilities.append(prob.float().cpu().numpy())
        else:
            pred = torch.expm1(model(x[ids]).float()).clamp_min(0)
            probabilities.append((pred >= STRATUM).float().cpu().numpy())
        if not torch.isfinite(pred).all():
            raise RuntimeError("nonfinite validation prediction")
        predictions.append(pred.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


@torch.no_grad()
def evaluate(model, x, target_cot, valid, indexes, batch, hurdle, save_path=None):
    pred, prob = predict(model, x, indexes, batch, hurdle)
    row_ids = indexes.cpu().numpy()
    ref = target_cot[row_ids]
    mask = valid[row_ids]
    err = pred - ref
    low = ref < STRATUM
    cloud = ~low
    thick = ref >= 30.0
    center = np.zeros((1, 1, 16, 16), dtype=bool)
    center[:, :, 6:11, 6:11] = True
    region = np.broadcast_to(center, mask.shape)
    groups = {"all": mask, "clear_cot_lt0p1": mask & low,
              "high_cot_ge30": mask & thick, "station_5x5": mask & region}
    counts = {k: int(v.sum()) for k, v in groups.items()}
    if not all(counts.values()):
        raise RuntimeError("empty validation stratum")
    result = {}
    for name, selector in groups.items():
        values = err[selector]
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(values ** 2, dtype=np.float64)))
        result[f"{name}_bias"] = float(np.mean(values, dtype=np.float64))
    clear_pred_cloud = (prob >= 0.5)
    y_cloud = cloud
    result.update({
        "clear_cot_lt0p1_false_cloud_fraction": float((clear_pred_cloud & mask & low).sum() / counts["clear_cot_lt0p1"]),
        "clear_cot_lt0p1_bias": result["clear_cot_lt0p1_bias"],
        "cloud_class_accuracy": float(((clear_pred_cloud == y_cloud) & mask).sum() / counts["all"]),
        "cloud_class_precision": float(((clear_pred_cloud & y_cloud) & mask).sum() / max(1, (clear_pred_cloud & mask).sum())),
        "cloud_class_recall": float(((clear_pred_cloud & y_cloud) & mask).sum() / max(1, (y_cloud & mask).sum())),
        "cloud_class_brier": float(np.square(prob[mask] - y_cloud[mask]).mean(dtype=np.float64)),
        "near_zero_reference_pixels_predicted_ge0p1_fraction": float(((pred >= STRATUM) & mask & low).sum() / counts["clear_cot_lt0p1"]),
        "valid_pixels": counts,
    })
    result["balanced_relative_rmse"] = float(np.mean([
        result[f"clear_cot_lt0p1_rmse"] / BASELINE["clear_cot_lt0p1_rmse"],
        result["all_rmse"] / BASELINE["cot_rmse"],
        result["high_cot_ge30_rmse"] / BASELINE["high_cot_ge30_rmse"],
        result["station_5x5_rmse"] / BASELINE["station_5x5_rmse"],
    ]))
    if save_path is not None:
        np.savez_compressed(save_path, row_indices=indexes.cpu().numpy(),
                            pred_cot=pred.astype(np.float32), cloud_probability=prob.astype(np.float32),
                            reference_cot=ref.astype(np.float32), mask=mask)
    return result


def cpu_smoke():
    torch.manual_seed(42)
    source = COTUNet().cpu()
    seed = source.state_dict()
    model = HurdleCOT(seed, 0.55).cpu().train()
    x = torch.randn(2, 16, 16, 16)
    target = torch.rand(2, 1, 16, 16) * 4.0
    valid = torch.ones_like(target, dtype=torch.bool)
    logits, prob, low, cloud, mix = model(x)
    assert logits.shape == prob.shape == low.shape == cloud.shape == mix.shape == (2, 1, 16, 16)
    assert bool(((low >= 0) & (low <= STRATUM)).all())
    assert bool((cloud >= STRATUM).all())
    with torch.no_grad():
        expected_cloud = torch.expm1(source(x) + LOG_STRATUM)
    assert torch.allclose(cloud, expected_cloud, rtol=1e-5, atol=1e-6)
    loss, _ = hurdle_loss(model, x, torch.log1p(target), target, valid, 0.05)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print("COT_HURDLE_CPU_SMOKE_PASS", f"loss={float(loss):.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--class-weight", type=float, default=0.05)
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
        raise RuntimeError("refuse to overwrite hurdle pilot output")
    if not (1 <= args.epochs <= 50 and 1 <= args.batch <= 4096 and 0 < args.lr <= 1e-3
            and 0 < args.class_weight <= 1):
        raise ValueError("pilot hyperparameter outside preregistered range")
    pack = root / "data/cot_repaired_pack_20260907_v1"
    base = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    if sha(base["checkpoint"]) != base["checkpoint_sha256"]:
        raise RuntimeError("frozen R checkpoint hash mismatch")
    checkpoint = torch.load(base["checkpoint"], map_location="cpu", weights_only=False)
    norm = json.loads((pack / "norm.json").read_text())
    if norm != checkpoint["metadata"]["norm"] or norm["feature_names"] != FEATURES:
        raise RuntimeError("frozen R normalization/channel contract mismatch")
    status = json.loads((pack / "pack_status.json").read_text())
    if status["state"] != "COMPLETE" or status["test_payloads_read"] != 0:
        raise RuntimeError("train/validation pack incomplete or test payload accessed")
    pack_hashes = json.loads((pack / "SHA256.json").read_text())
    for name in ("x_raw.npy", "target.npy", "mask.npy", "rows.csv", "norm.json"):
        if sha(pack / name) != pack_hashes[name]:
            raise RuntimeError("COT pack SHA mismatch: " + name)
    with (pack / "rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 18514 or any(row["split"] not in ("train", "validation") for row in rows):
        raise RuntimeError("unexpected pack split; test rows are forbidden")
    train_index = np.array([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    val_index = np.array([i for i, row in enumerate(rows) if row["split"] == "validation"], dtype=np.int64)
    if len(train_index) != 11978 or len(val_index) != 6536:
        raise RuntimeError("train/validation sample count drift")
    raw = np.load(pack / "x_raw.npy")
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    x_np = (raw - mean) / std
    nonfinite = int((~np.isfinite(x_np)).sum())
    x_np[~np.isfinite(x_np)] = 0
    target = np.load(pack / "target.npy") * np.float32(100)
    valid = np.load(pack / "mask.npy")
    if target.shape != (18514, 1, 16, 16) or valid.shape != target.shape:
        raise RuntimeError("target/mask shape drift")
    if not np.isfinite(target).all() or (target < 0).any() or (target > 100).any():
        raise RuntimeError("invalid CPP reference COT")
    if not valid.reshape(len(valid), -1).any(axis=1).all():
        raise RuntimeError("empty reference mask")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("PBS GPU not visible")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x = torch.from_numpy(x_np).to(device)
    target_cot = torch.from_numpy(target).to(device)
    target_log = torch.log1p(target_cot)
    mask = torch.from_numpy(valid).to(device)
    train_ids = torch.from_numpy(train_index).to(device)
    val_ids = torch.from_numpy(val_index).to(device)
    train_valid = valid[train_index]
    train_cloud = target[train_index] >= STRATUM
    cloud_fraction = float((train_cloud & train_valid).sum() / train_valid.sum())
    if cloud_fraction <= 0 or cloud_fraction >= 1:
        raise RuntimeError("training split lacks one of the two COT classes")
    output.mkdir(parents=True)
    contract = {
        "state": "PILOT_RUNNING", "test_used": False, "seed": args.seed,
        "initial_checkpoint_sha256": base["checkpoint_sha256"],
        "source_script_sha256": sha(__file__), "pack_hashes": pack_hashes,
        "train_rows": len(train_index), "validation_rows": len(val_index),
        "train_valid_pixels": int(train_valid.sum()), "train_cloud_fraction": cloud_fraction,
        "nonfinite_input_values_imputed": nonfinite, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr, "optimizer": "AdamW weight_decay=1e-4",
        "precision": "FP32", "class_threshold_cot": STRATUM,
        "semantics": "CPP-reference COT<0.1 is near-zero class; COT>=0.1 is cloud class, not a meteorological clear-sky label",
        "hurdle_output": "p(nearzero)*low_regression + p(cloud)*cloud_regression in physical COT units",
        "conditional_regression": "low expert constrained to [0,0.1]; cloud expert constrained to [0.1,infinity); each trained only on its train class",
        "loss": "0.5*(low-class mean masked Huber + cloud-class mean masked Huber) on log1p(COT) + class_weight*valid-pixel BCE",
        "class_weight": args.class_weight,
        "selection": "same four-stratum normalized RMSE score for control and hurdle: mean of nearzero, full-grid, COT>=30, station-5x5 RMSE divided by frozen-R values",
        "retention_limits": BASELINE,
        "input_domain": "observed real AGRI for train and validation; no forecast AGRI",
        "gpu": torch.cuda.get_device_name(0), "pbs_job_id": os.environ.get("PBS_JOBID"),
    }
    save_json(output / "contract.json", contract)
    frozen_model = COTUNet().to(device)
    frozen_model.load_state_dict(checkpoint["model"], strict=True)
    frozen_metrics = evaluate(frozen_model, x, target, valid, val_ids, args.batch, False,
                              output / "frozen_R_validation_predictions.npz")
    frozen_names = {"clear_cot_lt0p1_rmse": "clear_cot_lt0p1_rmse", "cot_rmse": "all_rmse",
                    "high_cot_ge30_rmse": "high_cot_ge30_rmse", "station_5x5_rmse": "station_5x5_rmse"}
    for key, expected in BASELINE.items():
        observed = frozen_metrics[frozen_names[key]]
        if abs(observed - expected) > 2e-4:
            raise RuntimeError(f"frozen R validation baseline drift for {key}: {observed} vs {expected}")
    save_json(output / "frozen_R_metrics.json", frozen_metrics)
    del frozen_model
    torch.cuda.empty_cache()

    results = {}
    for arm in ("single_head_control", "hurdle_classifier_two_regressors"):
        seed_all(args.seed)
        arm_dir = output / arm
        arm_dir.mkdir()
        if arm == "single_head_control":
            model = COTUNet().to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            model = HurdleCOT(checkpoint["model"], cloud_fraction).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        hurdle = arm != "single_head_control"
        initial = evaluate(model, x, target, valid, val_ids, args.batch, hurdle,
                           arm_dir / "initial_validation_predictions.npz")
        best_score = initial["balanced_relative_rmse"]
        best_epoch, best_metrics = -1, initial
        save_json(arm_dir / "initial_metrics.json", initial)
        torch.save({"model": model.state_dict(), "arm": arm, "epoch": -1,
                    "contract": contract, "validation_metrics": initial}, arm_dir / "initial.pt")
        for epoch in range(args.epochs):
            model.train()
            permutation = train_ids[torch.randperm(len(train_ids), device=device)]
            objective_sum, seen_pixels = 0.0, 0
            epoch_losses = {k: 0.0 for k in ("loss_class", "loss_low_regression", "loss_cloud_regression")}
            started = time.monotonic()
            for ids in permutation.split(args.batch):
                optimizer.zero_grad(set_to_none=True)
                if hurdle:
                    loss, parts = hurdle_loss(model, x[ids], target_log[ids], target_cot[ids],
                                              mask[ids], args.class_weight)
                    for key, value in parts.items():
                        epoch_losses[key] += value * len(ids)
                else:
                    log_pred = model(x[ids]).float()
                    loss = masked_huber(log_pred, target_log[ids], mask[ids])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss: {arm}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                n = int(mask[ids].sum())
                objective_sum += float(loss.detach()) * n
                seen_pixels += n
            metrics = evaluate(model, x, target, valid, val_ids, args.batch, hurdle)
            score = metrics["balanced_relative_rmse"]
            if score < best_score:
                best_score, best_epoch, best_metrics = score, epoch, metrics
                torch.save({"model": model.state_dict(), "arm": arm, "epoch": epoch,
                            "contract": contract, "validation_metrics": metrics}, arm_dir / "best.pt")
                evaluate(model, x, target, valid, val_ids, args.batch, hurdle,
                         arm_dir / "best_validation_predictions.npz")
            record = {"epoch": epoch, "train_objective": objective_sum / max(1, seen_pixels),
                      **metrics, "best_epoch": best_epoch, "seconds": time.monotonic() - started}
            if hurdle:
                record.update({key: value / len(permutation) for key, value in epoch_losses.items()})
            with (arm_dir / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print("EPOCH", arm, json.dumps(record), flush=True)
        save_json(arm_dir / "best_metrics.json", {"best_epoch": best_epoch, **best_metrics})
        results[arm] = {"best_epoch": best_epoch, **best_metrics}
        del model, optimizer
        torch.cuda.empty_cache()

    candidate = results["hurdle_classifier_two_regressors"]
    retention = {
        "clear_cot_lt0p1_rmse": candidate["clear_cot_lt0p1_rmse"] <= BASELINE["clear_cot_lt0p1_rmse"],
        "cot_rmse": candidate["all_rmse"] <= BASELINE["cot_rmse"],
        "high_cot_ge30_rmse": candidate["high_cot_ge30_rmse"] <= BASELINE["high_cot_ge30_rmse"],
        "station_5x5_rmse": candidate["station_5x5_rmse"] <= BASELINE["station_5x5_rmse"],
    }
    result = {
        "state": "COMPLETE_PAIRED_VALIDATION_PILOT", "test_used": False,
        "frozen_R_initial": frozen_metrics,
        "single_head_control": results["single_head_control"],
        "hurdle_classifier_two_regressors": candidate,
        "candidate_passes_all_retention_limits": all(retention.values()) and
            candidate["clear_cot_lt0p1_rmse"] < BASELINE["clear_cot_lt0p1_rmse"],
        "candidate_retention_checks_vs_frozen_R": retention,
        "candidate_minus_control": {key: candidate[key] - results["single_head_control"][key]
                                    for key in ("clear_cot_lt0p1_rmse", "all_rmse",
                                                "high_cot_ge30_rmse", "station_5x5_rmse",
                                                "balanced_relative_rmse")},
        "note": "Validation-only real-AGRI retrieval pilot against CPP product reference; no GHI claim and no test data used.",
    }
    save_json(output / "complete.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
