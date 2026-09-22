"""Paired, validation-only continuation pilot for physical COT accuracy.

Both arms start from the same frozen Hunan R checkpoint and use the same
training order, optimizer and validation selector. The sole arm difference is
a small physical-COT SmoothL1 auxiliary term. No test payload is opened.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_cot_repaired import COTUNet


FEATURES = [f"C{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)]
FEATURES += ["cosSOZ", "cosRAA", "day_mask"]
ARMS = ("log_control", "physical_aux")


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
    return (values * mask).sum() / mask.sum().clamp_min(1)


def train_loss(log_pred, log_true, mask, arm, physical_weight):
    log_term = masked_mean(F.huber_loss(log_pred, log_true, delta=0.1, reduction="none"), mask)
    if arm == "log_control":
        return log_term
    physical_pred = torch.expm1(log_pred)
    physical_true = torch.expm1(log_true)
    physical_term = masked_mean(F.smooth_l1_loss(physical_pred, physical_true,
                                                beta=2.0, reduction="none"), mask)
    return log_term + physical_weight * physical_term


@torch.no_grad()
def evaluate(model, x, y, mask, indexes, batch, save_predictions=False):
    model.eval()
    total = {key: 0.0 for key in ("se", "ae", "bias", "log_huber_sum", "high_se", "high_bias",
                                     "center_se", "clear_se")}
    counts = {key: 0 for key in ("all", "high", "center", "clear")}
    center = torch.zeros((1, 1, 16, 16), dtype=torch.bool, device=x.device)
    center[:, :, 6:11, 6:11] = True
    outputs = []
    for ids in indexes.split(batch):
        log_pred = model(x[ids]).float()
        if not torch.isfinite(log_pred).all():
            raise RuntimeError("nonfinite COT validation prediction")
        reference = torch.expm1(y[ids])
        prediction = torch.expm1(log_pred)
        error = prediction - reference
        valid = mask[ids]
        high = valid & (reference >= 30)
        central = valid & center
        clear = valid & (reference < 0.1)
        n = int(valid.sum())
        counts["all"] += n
        counts["high"] += int(high.sum())
        counts["center"] += int(central.sum())
        counts["clear"] += int(clear.sum())
        total["se"] += float((error.square() * valid).sum().double())
        total["ae"] += float((error.abs() * valid).sum().double())
        total["bias"] += float((error * valid).sum().double())
        total["high_se"] += float((error.square() * high).sum().double())
        total["high_bias"] += float((error * high).sum().double())
        total["center_se"] += float((error.square() * central).sum().double())
        total["clear_se"] += float((error.square() * clear).sum().double())
        total["log_huber_sum"] += float((F.huber_loss(log_pred, y[ids], delta=0.1,
                                                       reduction="none") * valid).sum().double())
        if save_predictions:
            outputs.append(log_pred.cpu().numpy())
    if not all(counts.values()):
        raise RuntimeError("empty validation stratum")
    result = {"cot_rmse": (total["se"] / counts["all"]) ** 0.5,
              "cot_mae": total["ae"] / counts["all"],
              "cot_bias": total["bias"] / counts["all"],
              "log_huber": total["log_huber_sum"] / counts["all"],
              "high_cot_ge30_rmse": (total["high_se"] / counts["high"]) ** 0.5,
              "high_cot_ge30_bias": total["high_bias"] / counts["high"],
              "station_5x5_rmse": (total["center_se"] / counts["center"]) ** 0.5,
              "clear_cot_lt0p1_rmse": (total["clear_se"] / counts["clear"]) ** 0.5,
              "valid_pixels": counts}
    return result, (np.concatenate(outputs) if save_predictions else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--physical-weight", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite pilot output")
    if not (1 <= args.epochs <= 50 and 1 <= args.batch <= 4096 and 0 < args.lr <= 1e-3
            and 0 < args.physical_weight <= 0.02):
        raise ValueError("pilot hyperparameter outside preregistered safety range")
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
        raise RuntimeError("COT train/validation pack incomplete")
    pack_hashes = json.loads((pack / "SHA256.json").read_text())
    for name in ("x_raw.npy", "target.npy", "mask.npy", "rows.csv", "norm.json"):
        if sha(pack / name) != pack_hashes[name]:
            raise RuntimeError("COT pack SHA mismatch: " + name)
    with (pack / "rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 18514 or any(row["split"] not in ("train", "validation") for row in rows):
        raise RuntimeError("unexpected COT pack split")
    train_index = np.array([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    val_index = np.array([i for i, row in enumerate(rows) if row["split"] == "validation"], dtype=np.int64)
    if len(train_index) != 11978 or len(val_index) != 6536:
        raise RuntimeError("COT train/validation sample count drift")
    raw = np.load(pack / "x_raw.npy")
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    normalized = (raw - mean) / std
    nonfinite = int((~np.isfinite(normalized)).sum())
    normalized[~np.isfinite(normalized)] = 0
    target = np.load(pack / "target.npy") * np.float32(100)
    valid = np.load(pack / "mask.npy")
    if target.shape != (18514, 1, 16, 16) or valid.shape != target.shape:
        raise RuntimeError("COT target/mask shape drift")
    if not np.isfinite(target).all() or (target < 0).any() or (target > 100).any():
        raise RuntimeError("invalid physical COT target")
    if not valid.reshape(len(valid), -1).any(axis=1).all():
        raise RuntimeError("empty COT target mask")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("PBS GPU not visible")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(np.log1p(target)).to(device)
    mask = torch.from_numpy(valid).to(device)
    train_ids = torch.from_numpy(train_index).to(device)
    val_ids = torch.from_numpy(val_index).to(device)
    output.mkdir(parents=True)
    contract = {"state": "PILOT_RUNNING", "test_used": False, "seed": args.seed,
                "initial_checkpoint_sha256": base["checkpoint_sha256"],
                "initial_checkpoint": base["checkpoint"],
                "source_script_sha256": sha(__file__), "pack_hashes": pack_hashes,
                "train_rows": len(train_index), "validation_rows": len(val_index),
                "nonfinite_input_values_imputed": nonfinite,
                "arms": list(ARMS), "epochs": args.epochs, "batch": args.batch,
                "lr": args.lr, "physical_weight": args.physical_weight,
                "optimizer": "AdamW weight_decay=1e-4", "precision": "FP32",
                "selection": "lowest same-mask validation physical COT RMSE; baseline epoch -1 retained",
                "physical_aux": "log1p Huber delta=0.1 + weight * physical COT SmoothL1 beta=2",
                "gpu": torch.cuda.get_device_name(0), "pbs_job_id": os.environ.get("PBS_JOBID")}
    save_json(output / "contract.json", contract)
    for arm in ARMS:
        seed_all(args.seed)
        model = COTUNet().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        arm_dir = output / arm
        arm_dir.mkdir()
        best_metrics, best_prediction = evaluate(model, x, y, mask, val_ids, args.batch, True)
        best_score, best_epoch = best_metrics["cot_rmse"], -1
        save_json(arm_dir / "initial_metrics.json", best_metrics)
        np.savez_compressed(arm_dir / "validation_predictions.npz", row_indices=val_index,
                            pred_log1p_cot=best_prediction, reference_cot=target[val_index], mask=valid[val_index])
        for epoch in range(args.epochs):
            model.train()
            permutation = train_ids[torch.randperm(len(train_ids), device=device)]
            weighted_loss, pixels = 0.0, 0
            started = time.monotonic()
            for ids in permutation.split(args.batch):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[ids]).float()
                loss = train_loss(prediction, y[ids], mask[ids], arm, args.physical_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss in {arm} epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                count = int(mask[ids].sum())
                weighted_loss += float(loss) * count
                pixels += count
            metrics, _ = evaluate(model, x, y, mask, val_ids, args.batch)
            improved = metrics["cot_rmse"] < best_score
            if improved:
                best_score, best_epoch, best_metrics = metrics["cot_rmse"], epoch, metrics
                torch.save({"model": model.state_dict(), "arm": arm, "epoch": epoch,
                            "contract": contract, "validation_metrics": metrics}, arm_dir / "best.pt")
                _, best_prediction = evaluate(model, x, y, mask, val_ids, args.batch, True)
                np.savez_compressed(arm_dir / "validation_predictions.npz", row_indices=val_index,
                                    pred_log1p_cot=best_prediction, reference_cot=target[val_index], mask=valid[val_index])
            record = {"epoch": epoch, "train_objective": weighted_loss / pixels,
                      **metrics, "best_epoch": best_epoch, "seconds": time.monotonic() - started}
            with (arm_dir / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print("EPOCH", arm, json.dumps(record), flush=True)
        save_json(arm_dir / "best_metrics.json", {"best_epoch": best_epoch, **best_metrics})
    control = json.loads((output / "log_control/best_metrics.json").read_text())
    candidate = json.loads((output / "physical_aux/best_metrics.json").read_text())
    result = {"state": "COMPLETE_PAIRED_VALIDATION_PILOT", "test_used": False,
              "control": control, "physical_aux": candidate,
              "candidate_minus_control_cot_rmse": candidate["cot_rmse"] - control["cot_rmse"],
              "candidate_minus_control_high_rmse": candidate["high_cot_ge30_rmse"] - control["high_cot_ge30_rmse"],
              "candidate_minus_control_station_5x5_rmse": candidate["station_5x5_rmse"] - control["station_5x5_rmse"],
              "note": "Validation-only selection; not an independent test or GHI claim."}
    save_json(output / "complete.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
