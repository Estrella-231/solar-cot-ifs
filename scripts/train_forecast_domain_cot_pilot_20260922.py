"""Paired R continuation: observed versus SimVP forecast AGRI train inputs.

Both arms use identical Hunan train target keys and CPP masks, initialization,
optimizer, order and forecast-domain validation selector. Test is unopened.
"""
import argparse
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
from train_cot_physical_aux_pilot_20260922 import evaluate, save_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(raw, mean, std):
    x = (np.asarray(raw, dtype=np.float32) - mean) / std
    missing = int((~np.isfinite(x)).sum())
    x[~np.isfinite(x)] = 0
    return torch.from_numpy(np.ascontiguousarray(x)).cuda(), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, pack, output = args.root.resolve(), args.pack.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite forecast-domain R pilot")
    if not torch.cuda.is_available() or not (1 <= args.epochs <= 20 and 1 <= args.batch <= 2048):
        raise RuntimeError("GPU unavailable or pilot bounds invalid")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    if sha(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
        raise RuntimeError("R checkpoint hash mismatch")
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    original_pack = root / "data/cot_repaired_pack_20260907_v1"
    norm = json.loads((original_pack / "norm.json").read_text())
    if norm != checkpoint["metadata"]["norm"] or norm["test_used"] is not False:
        raise RuntimeError("frozen R normalization mismatch")
    packed = json.loads((pack / "complete.json").read_text())
    if packed["state"] != "COMPLETE_FORECAST_DOMAIN_COT_PILOT_PACK" or packed["test_used"] is not False:
        raise RuntimeError("forecast-domain pair pack incomplete")
    source_rows_hash = sha(original_pack / "rows.csv")
    if packed["R_pack_rows_sha256"] != source_rows_hash:
        raise RuntimeError("CPP reference row binding changed")
    for split in ("train", "val"):
        for filename, key in (("forecast_x.npy", "forecast_x_sha256"), ("keys.npy", "keys_sha256")):
            if sha(pack / split / filename) != packed["splits"][split][key]:
                raise RuntimeError("pilot pack hash mismatch: " + split + "/" + filename)
    train_keys = np.load(pack / "train/keys.npy")
    val_keys = np.load(pack / "val/keys.npy")
    if train_keys.shape[1] != 4 or val_keys.shape[1] != 4:
        raise RuntimeError("pilot key shape mismatch")
    if len(train_keys) != packed["splits"]["train"]["pairs"] or len(val_keys) != packed["splits"]["val"]["pairs"]:
        raise RuntimeError("pilot key row count mismatch")
    train_ref = torch.from_numpy(train_keys[:, 3].copy()).long().cuda()
    val_ref = torch.from_numpy(val_keys[:, 3].copy()).long().cuda()
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise RuntimeError("invalid R normalization")
    real_x, real_missing = normalize(np.load(original_pack / "x_raw.npy"), mean, std)
    forecast_train, train_missing = normalize(np.load(pack / "train/forecast_x.npy"), mean, std)
    forecast_val, val_missing = normalize(np.load(pack / "val/forecast_x.npy"), mean, std)
    reference = np.load(original_pack / "target.npy") * np.float32(100)
    valid = np.load(original_pack / "mask.npy")
    y_all = torch.from_numpy(np.log1p(reference)).cuda()
    m_all = torch.from_numpy(valid).cuda()
    y_val = y_all[val_ref]
    m_val = m_all[val_ref]
    val_ids = torch.arange(len(val_keys), device="cuda")
    output.mkdir(parents=True)
    contract = {"state": "RUNNING_FORECAST_DOMAIN_COT_PAIRED_PILOT", "test_used": False,
                "R_initial_checkpoint_sha256": cfg["checkpoint_sha256"],
                "pair_pack_sha256": sha(pack / "complete.json"),
                "script_sha256": sha(__file__), "train_pairs": len(train_keys),
                "validation_pairs": len(val_keys),
                "train_unique_cpp_targets": packed["splits"]["train"]["unique_cpp_targets"],
                "validation_unique_cpp_targets": packed["splits"]["val"]["unique_cpp_targets"],
                "leads_minutes": packed["forecast_leads_minutes"],
                "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                "seed": args.seed, "precision": "FP32",
                "loss": "valid-pixel log1p(COT) Huber delta=0.1",
                "selection": "same forecast-domain validation physical COT RMSE, including initial epoch -1",
                "arms": ["real_train_control", "forecast_train_adapt"],
                "train_inputs": "observed AGRI versus causal SimVP-predicted AGRI; labels and all other settings identical",
                "validation_inputs": "causal SimVP-predicted AGRI for both arms",
                "nonfinite_imputation": {"real": real_missing, "forecast_train": train_missing,
                                         "forecast_validation": val_missing},
                "gpu": torch.cuda.get_device_name(0), "pbs_job_id": os.environ.get("PBS_JOBID")}
    save_json(output / "contract.json", contract)
    summaries = {}
    for arm in contract["arms"]:
        seed_all(args.seed)
        model = COTUNet().float().cuda()
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        arm_dir = output / arm
        arm_dir.mkdir()
        initial, predictions = evaluate(model, forecast_val, y_val, m_val, val_ids, args.batch, True)
        save_json(arm_dir / "initial_metrics.json", initial)
        np.savez_compressed(arm_dir / "validation_predictions.npz", keys=val_keys,
                            pred_log1p_cot=predictions,
                            reference_cot=reference[val_keys[:, 3]], mask=valid[val_keys[:, 3]])
        best_score, best_epoch, best = initial["cot_rmse"], -1, initial
        for epoch in range(args.epochs):
            started = time.monotonic()
            model.train()
            order = torch.randperm(len(train_keys), device="cuda")
            loss_sum, pixel_count = 0.0, 0
            for ids in order.split(args.batch):
                refs = train_ref[ids]
                inputs = real_x[refs] if arm == "real_train_control" else forecast_train[ids]
                labels = y_all[refs]
                mask = m_all[refs]
                optimizer.zero_grad(set_to_none=True)
                pred = model(inputs)
                loss = (F.huber_loss(pred, labels, reduction="none", delta=0.1) * mask).sum() / mask.sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite train loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                n = int(mask.sum())
                loss_sum += float(loss) * n
                pixel_count += n
            metrics, _ = evaluate(model, forecast_val, y_val, m_val, val_ids, args.batch)
            if metrics["cot_rmse"] < best_score:
                best_score, best_epoch, best = metrics["cot_rmse"], epoch, metrics
                torch.save({"model": model.state_dict(), "arm": arm, "epoch": epoch,
                            "contract": contract, "validation_metrics": metrics}, arm_dir / "best.pt")
                _, predictions = evaluate(model, forecast_val, y_val, m_val, val_ids, args.batch, True)
                np.savez_compressed(arm_dir / "validation_predictions.npz", keys=val_keys,
                                    pred_log1p_cot=predictions,
                                    reference_cot=reference[val_keys[:, 3]], mask=valid[val_keys[:, 3]])
            record = {"epoch": epoch, "train_huber": loss_sum / pixel_count,
                      **metrics, "best_epoch": best_epoch, "seconds": time.monotonic() - started}
            with (arm_dir / "epochs.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            print("EPOCH", arm, json.dumps(record), flush=True)
        save_json(arm_dir / "best_metrics.json", {"best_epoch": best_epoch, **best})
        summaries[arm] = {"best_epoch": best_epoch, **best}
    control, adapted = summaries["real_train_control"], summaries["forecast_train_adapt"]
    report = {"state": "COMPLETE_FORECAST_DOMAIN_COT_PAIRED_PILOT", "test_used": False,
              "arms": summaries,
              "forecast_adapt_minus_real_control": {
                  key: adapted[key] - control[key] for key in
                  ("cot_rmse", "cot_mae", "high_cot_ge30_rmse", "station_5x5_rmse",
                   "clear_cot_lt0p1_rmse")},
              "note": f"{len(contract['leads_minutes'])} predeclared leads; validation-only; CPP is retrieval reference, not independent physical truth."}
    save_json(output / "complete.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
