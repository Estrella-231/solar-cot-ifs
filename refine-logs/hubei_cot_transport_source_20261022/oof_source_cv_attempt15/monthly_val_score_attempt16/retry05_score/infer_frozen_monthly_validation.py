#!/usr/bin/env python3
"""Run frozen attempt15 checkpoints on the audited monthly validation cache.

This script performs inference only. It never calls the trainer, optimizer, loss,
or scheduler. CPP targets emitted by the dataset are not passed to the model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


LEADS = tuple(range(15, 241, 15))
HISTORY_OFFSETS = tuple(range(-105, 1, 15))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y%m%d%H%M")


class CausalValidationDataset(Dataset):
    """Monthly-validation histories and deterministic geometry only.

    Unlike the trainer dataset, this class never opens CPP value or validity
    arrays, so future targets cannot enter the inference batches.
    """

    def __init__(self, cache: Path):
        self.cache = cache
        self.arrays = {
            name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("geometry", "history_cot_log", "history_valid")
        }
        rows = read_csv(cache / "bank_rows.csv")
        self.by_time: dict[dt.datetime, int] = {}
        self.split_by_time: dict[dt.datetime, str] = {}
        for row in rows:
            t = parse_time(row["time_bjt"])
            split = row["split"].lower()
            if split not in {"train", "val"}:
                raise ValueError(f"unexpected row split in train/val cache: {split}")
            if t in self.by_time:
                raise ValueError(f"duplicate bank timestamp {t}")
            self.by_time[t] = int(row["index"])
            self.split_by_time[t] = split

        val_pairs = [r for r in read_csv(cache / "forecast_pairs.csv")
                     if r["split"].lower() == "val"]
        pair_by_init: dict[dt.datetime, set[int]] = {}
        for row in val_pairs:
            init_t = parse_time(row["init_time_bjt"])
            target_t = parse_time(row["target_time_bjt"])
            lead = int(row["lead_minutes"])
            if lead not in LEADS or target_t - init_t != dt.timedelta(minutes=lead):
                raise ValueError(f"invalid validation time/lead row: {row}")
            if self.split_by_time.get(init_t) != "val" or self.split_by_time.get(target_t) != "val":
                raise ValueError(f"cross-split validation pair: {row}")
            if self.by_time.get(init_t) != int(row["init_index"]):
                raise ValueError(f"validation init index mismatch: {row}")
            if self.by_time.get(target_t) != int(row["target_index"]):
                raise ValueError(f"validation target index mismatch: {row}")
            pair_by_init.setdefault(init_t, set()).add(lead)

        self.windows: list[tuple[dt.datetime, list[int], list[int]]] = []
        for init_t in sorted(pair_by_init):
            history_times = [init_t + dt.timedelta(minutes=m) for m in HISTORY_OFFSETS]
            if any(self.split_by_time.get(t) != "val" or t not in self.by_time
                   for t in history_times):
                raise ValueError(f"incomplete or cross-split history at {init_t}")
            hist_idx = [self.by_time[t] for t in history_times]
            future_idx = []
            for lead in LEADS:
                target_t = init_t + dt.timedelta(minutes=lead)
                target_i = self.by_time.get(target_t, -1)
                if target_i >= 0 and self.split_by_time.get(target_t) != "val":
                    raise ValueError(f"future geometry crosses split at {init_t} +{lead}")
                future_idx.append(target_i)
            self.windows.append((init_t, hist_idx, future_idx))
        if not self.windows:
            raise ValueError("monthly-validation cache contains no forecast windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        init_t, hist_idx, future_idx = self.windows[item]
        history = np.asarray(self.arrays["history_cot_log"][hist_idx], dtype=np.float32)
        history_valid = np.asarray(self.arrays["history_valid"][hist_idx], dtype=np.uint8)
        geom_hist = np.asarray(self.arrays["geometry"][hist_idx], dtype=np.float32)
        geom_future = np.zeros((16, 3, 16, 16), dtype=np.float32)
        for lead_i, idx in enumerate(future_idx):
            if idx >= 0:
                geom_future[lead_i] = np.asarray(self.arrays["geometry"][idx], dtype=np.float32)
        geometry = np.concatenate((geom_hist, geom_future), axis=0)
        return {
            "history": torch.from_numpy(history),
            "history_valid": torch.from_numpy(history_valid),
            "geometry": torch.from_numpy(geometry),
            "init_time": init_t.strftime("%Y%m%d%H%M"),
        }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--trainer", type=Path, required=True)
    ap.add_argument("--project-scripts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--checkpoint-origin", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cache = args.cache.resolve()
    report_path = cache / "cache_report.json"
    audit_path = cache.parent / "cache_audit" / "cache_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_HUBEI_STATION_PATCH_COT_GEOMETRY_CACHE_TRAINVAL_ONLY":
        raise ValueError(f"unexpected cache state: {report.get('state')}")
    if audit.get("state") != "PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED":
        raise ValueError(f"cache audit failed: {audit.get('state')}")
    for doc in (report, audit):
        if doc.get("test_payloads_read") != 0 or doc.get("buffer_payloads_read") != 0:
            raise ValueError("test/buffer cache gate is not closed")
    if int(report.get("split", {}).get("val", -1)) != 1855:
        raise ValueError(f"unexpected validation frame count: {report.get('split')}")

    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite inference output: {args.output}")
    sys.path.insert(0, str(args.project_scripts.resolve()))
    sys.path.insert(0, str(args.trainer.resolve().parent))
    from retrieved_cot_forecaster_20261013 import RetrievedCOTForecaster

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("PBS inference must use its allocated GPU; CPU fallback is disabled")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dataset = CausalValidationDataset(cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    model = RetrievedCOTForecaster(width=96).to(device)

    checkpoint_origin = args.checkpoint_origin.resolve()
    if sha256(args.checkpoint) != sha256(checkpoint_origin):
        raise ValueError("loaded checkpoint differs from its provenance path")
    training_receipt_path = checkpoint_origin.parent / "training_complete.json"
    training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    expected_seed = int(args.model_name.rsplit("seed", 1)[1])
    if (training_receipt.get("state") != "COMPLETE_TRAINED_HUBEI_INTERNAL_TRAIN_DATE_HOLDOUT_COT_DIAGNOSTIC"
            or training_receipt.get("seed") != expected_seed
            or training_receipt.get("global_step") != 1350
            or training_receipt.get("global_batch") != 32
            or training_receipt.get("world_size") != 1
            or training_receipt.get("train_config", {}).get("width") != 96
            or training_receipt.get("official_monthly_validation_payloads_read", 0) != 0
            or "official monthly validation, test, and buffer not read" not in training_receipt.get("split", "")
            or training_receipt.get("test_payloads_read") != 0
            or training_receipt.get("buffer_payloads_read") != 0):
        raise ValueError(f"attempt15 checkpoint receipt is outside the frozen contract: {training_receipt_path}")
    checkpoint_hash = sha256(args.checkpoint)
    if training_receipt.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"checkpoint hash does not match its training receipt: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model state dictionary")
    model.load_state_dict(state, strict=True)
    model.eval()

    outputs: list[np.ndarray] = []
    keys: list[str] = []
    input_shapes: set[tuple[int, ...]] = set()
    with torch.inference_mode():
        for batch in loader:
            history = batch["history"].to(device, non_blocking=True)
            history_valid = batch["history_valid"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            if tuple(history.shape[1:]) != (8, 1, 16, 16):
                raise ValueError(f"unexpected history shape {tuple(history.shape)}")
            if tuple(history_valid.shape[1:]) != (8, 1, 16, 16):
                raise ValueError(f"unexpected history-valid shape {tuple(history_valid.shape)}")
            if tuple(geometry.shape[1:]) != (24, 3, 16, 16):
                raise ValueError(f"unexpected geometry shape {tuple(geometry.shape)}")
            input_shapes.add(tuple(history.shape[1:]))
            # Deliberately pass only causal history, its mask, and known geometry.
            pred = model(history, history_valid, geometry)
            if pred.ndim != 5 or pred.shape[1:] != (16, 1, 16, 16):
                raise ValueError(f"unexpected prediction shape {tuple(pred.shape)}")
            if not torch.isfinite(pred).all():
                raise ValueError("model emitted non-finite validation prediction")
            outputs.append(pred.float().cpu().numpy())
            keys.extend(batch["init_time"])
    if len(keys) != len(dataset) or len(set(keys)) != len(keys):
        raise ValueError("validation prediction keys are missing or duplicated")
    if len(input_shapes) != 1:
        raise ValueError(f"input shape changed across batches: {input_shapes}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_path = args.output
    np.savez_compressed(pred_path, init_time_bjt=np.asarray(keys),
                        pred_log_scaled=np.concatenate(outputs, axis=0))
    receipt = {
        "state": "COMPLETE_FROZEN_MONTHLY_VALIDATION_INFERENCE",
        "model_name": args.model_name,
        "checkpoint": str(checkpoint_origin),
        "checkpoint_sha256": checkpoint_hash,
        "training_receipt": str(training_receipt_path),
        "training_receipt_sha256": sha256(training_receipt_path),
        "prediction_file": str(pred_path),
        "prediction_sha256": sha256(pred_path),
        "cache_report_sha256": sha256(report_path),
        "cache_audit_sha256": sha256(audit_path),
        "trainer_source_sha256": sha256(args.trainer),
        "project_model_source_sha256": sha256(args.project_scripts / "retrieved_cot_forecaster_20261013.py"),
        "inference_script_sha256": sha256(Path(__file__).resolve()),
        "validation_init_windows": len(dataset),
        "validation_init_keys": len(keys),
        "input_history_shape_per_sample": list(next(iter(input_shapes))),
        "forecast_leads_minutes": list(LEADS),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "official_monthly_validation_cache_used_for_inference": True,
        "future_cpp_values_or_masks_loaded_for_inference": False,
        "model_inputs": ["causal_history_cot_log", "history_valid_mask", "deterministic_solar_geometry"],
        "optimizer_updates": 0,
    }
    receipt_path = pred_path.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
