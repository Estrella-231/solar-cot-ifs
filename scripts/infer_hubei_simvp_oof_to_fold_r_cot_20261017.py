#!/usr/bin/env python3
"""Infer fold-specific forecast COT from a held-out SimVP OOF bank.

This entry point is deliberately restricted to one train-date OOF fold. It
does not read GHI labels or test/buffer payloads and refuses an output path
that already exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


CHANNELS = [
    "C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10", "C11",
    "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask",
]
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bjt(value: str) -> datetime:
    text = value.strip().replace("-", "").replace(":", "").replace(" ", "")
    if len(text) == 14:
        text = text[:12]
    if len(text) != 12 or not text.isdigit():
        raise ValueError(f"Invalid BJT timestamp: {value!r}")
    return datetime.strptime(text, "%Y%m%d%H%M")


def read_day_fold(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    if not rows or not {"date_BJT", "split", "heldout_fold"}.issubset(rows[0]):
        raise ValueError("Unexpected frozen date-fold schema")
    result: dict[str, int] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        day = row["date_BJT"].replace("-", "")
        fold = int(row["heldout_fold"])
        if day in result:
            raise ValueError(f"Duplicate train date in fold assignment: {day}")
        result[day] = fold
    return result


def validate_source(bank: Path, oof_index: Path, fold_assignment: Path, r_pack: Path,
                    r_run: Path, model_source: Path, fold: int) -> tuple[dict, dict, dict, list[dict[str, str]]]:
    report = json_file(bank / "BANK_REPORT.json")
    if report.get("state") not in {"MERGED_SIMVP_R_PATCH_OOF_VAL_BANK", "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK"}:
        raise ValueError("Input is not an audited per-fold SimVP OOF bank")
    split_counts = report.get("split_counts", {})
    if report.get("test_payloads_read") != 0:
        raise ValueError("OOF bank report does not prove test/buffer closure")
    if report.get("test_or_buffer_rows") not in (0, None) or split_counts.get("test", 0) != 0 or split_counts.get("buffer", 0) != 0:
        raise ValueError("OOF bank report has nonzero or unverified test/buffer rows")
    expected_count = report.get("sequences") or report.get("split_counts", {}).get("val")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("OOF bank report has no positive validation sequence count")
    report_index_sha = report.get("canonical_oof_index_sha256") or report.get("index_sha256")
    if not report_index_sha or sha256(oof_index) != report_index_sha:
        raise ValueError("Input bank is not bound to the supplied frozen OOF index")
    expected_sha = report.get("files_sha256", {})
    for name in ("keys.csv", "pred_agri_patch.npy", "future_geometry_patch.npy"):
        path = bank / "val" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing OOF bank source: {path}")
        if name in expected_sha and sha256(path) != expected_sha.get(name):
            raise ValueError(f"OOF bank source checksum mismatch: {name}")
    keys = read_csv(bank / "val" / "keys.csv")
    pred = np.load(bank / "val" / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
    geom = np.load(bank / "val" / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
    if len(keys) != expected_count or pred.shape != (expected_count, 16, 13, 16, 16) or geom.shape != (expected_count, 16, 3, 16, 16):
        raise ValueError("OOF keys/AGRI/geometry shapes disagree")

    folds = read_day_fold(fold_assignment)
    oof_days: set[str] = set()
    for i, row in enumerate(keys):
        if int(row.get("sample_index", "-1")) != i or row.get("split") != "val":
            raise ValueError(f"Noncanonical OOF key at row {i}")
        init = parse_bjt(row["init_time_bjt"])
        day = init.strftime("%Y%m%d")
        if folds.get(day) != fold:
            raise ValueError(f"OOF key {i} is not from frozen heldout fold{fold} date {day}")
        targets = row.get("target_times_bjt", "").split("|")
        if len(targets) != 16:
            raise ValueError(f"OOF key {i} does not have 16 targets")
        for lead_i, target_raw in enumerate(targets):
            target = parse_bjt(target_raw)
            if target != init + timedelta(minutes=15 * (lead_i + 1)):
                raise ValueError(f"Invalid target lead in OOF key {i}")
            if target.strftime("%Y%m%d") != day:
                raise ValueError(f"OOF sequence crosses BJT date at key {i}")
        oof_days.add(day)

    status = json_file(r_run / "status.json")
    metadata = json_file(r_run / "run_metadata.json")
    audit = json_file(r_run / "audit_report.json")
    norm_path = r_pack / "norm.json"
    norm = json_file(norm_path)
    if status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or status.get("test_payloads_read") != 0:
        raise ValueError("Fold-specific R run is not complete/test-closed")
    if audit.get("state") != "HUBEI_OOF_R_FOLD_COMPLETION_AUDIT_PASSED_TEST_CLOSED" or audit.get("test_payloads_read") != 0:
        raise ValueError("Fold-specific R completion audit did not pass")
    if int(audit.get("fold_seed", -1)) != 42 or int(metadata.get("seed", -1)) != 42:
        raise ValueError("Unexpected R seed")
    if metadata.get("normalization") != norm or norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("R normalization provenance mismatch")
    if norm.get("input_channels") != CHANNELS:
        raise ValueError("Unexpected R input channel order")
    if metadata.get("model_source_sha256") != sha256(model_source):
        raise ValueError("R model source hash does not match the trained source")
    for filename, expected in metadata.get("source_pack_hashes", {}).items():
        if sha256(r_pack / filename) != expected:
            raise ValueError(f"Fold-specific R pack hash mismatch: {filename}")
    checkpoint = r_run / "best.pt"
    if sha256(checkpoint) != audit.get("best_checkpoint_sha256"):
        raise ValueError("Fold-specific R best checkpoint hash mismatch")

    r_rows = read_csv(r_pack / "rows.csv")
    train_dates = {parse_bjt(r["time_bjt"]).strftime("%Y%m%d") for r in r_rows if r["split"] == "train"}
    if train_dates & oof_days:
        raise ValueError("R training dates overlap held-out S OOF dates")
    if audit.get("test_payloads_read") != 0:
        raise ValueError("R audit indicates test payload access")
    return report, status, metadata, keys


def normalize_r_inputs(pred_agri: np.ndarray, future_geometry: np.ndarray,
                       mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.concatenate((np.asarray(pred_agri, dtype=np.float32),
                          np.asarray(future_geometry, dtype=np.float32)), axis=2)
    raw = raw.reshape(-1, 16, 16, 16)
    if raw.shape[1:] != (16, 16, 16) or not np.isfinite(raw).all():
        raise ValueError("Invalid forecast-AGRI/geometry input")
    solar = raw[:, 15:16] > 0.5
    x = (raw - mean[None, :, None, None]) / std[None, :, None, None]
    x[:, 15:16] = np.where(solar, x[:, 15:16], 0.0)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite normalized R input")
    return x.astype(np.float32), solar


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", required=True, type=Path)
    p.add_argument("--oof-index", required=True, type=Path)
    p.add_argument("--fold-assignment", required=True, type=Path)
    p.add_argument("--r-pack", required=True, type=Path)
    p.add_argument("--r-run", required=True, type=Path)
    p.add_argument("--model-source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fold", required=True, type=int, choices=range(5))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sequence-batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Smoke-test row limit; 0 means all")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available() or not a.device.startswith("cuda"):
        raise RuntimeError("Inference requires the explicitly selected available GPU; CPU fallback disabled")
    if a.sequence_batch < 1 or a.limit < 0:
        raise ValueError("Invalid sequence-batch/limit")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {a.output}")

    bank_report, r_status, r_meta, keys = validate_source(
        a.forecast_bank, a.oof_index, a.fold_assignment, a.r_pack, a.r_run, a.model_source, a.fold)
    norm = json_file(a.r_pack / "norm.json")
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    if mean.shape != (16,) or std.shape != (16,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid fold-specific R normalization vectors")

    spec = importlib.util.spec_from_file_location("hubei_fold_r_inference", a.model_source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen R model source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device(a.device)
    model = module.COTUNet().to(device).eval()
    state = torch.load(a.r_run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)

    count = len(keys) if a.limit == 0 else min(a.limit, len(keys))
    source = a.forecast_bank / "val"
    pred = np.load(source / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
    geom = np.load(source / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
    a.output.mkdir(parents=True, exist_ok=False)
    out = np.lib.format.open_memmap(a.output / "forecast_cot.npy", mode="w+", dtype=np.float32,
                                   shape=(count, 16, 1, 16, 16))
    with (a.output / "keys.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys[0].keys()))
        writer.writeheader()
        writer.writerows(keys[:count])

    day_off = 0
    pixel_total = 0
    for start in range(0, count, a.sequence_batch):
        stop = min(start + a.sequence_batch, count)
        x_np, solar_np = normalize_r_inputs(np.asarray(pred[start:stop]), np.asarray(geom[start:stop]), mean, std)
        day_off += int((~solar_np).sum())
        pixel_total += int(solar_np.size)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            log_cot = model(x).float()
            if not torch.isfinite(log_cot).all():
                raise RuntimeError(f"Non-finite log1p(COT) at rows {start}:{stop}")
            cot = torch.expm1(log_cot)
            cot.masked_fill_(~torch.from_numpy(solar_np).to(device, non_blocking=True), 0.0)
        if not torch.isfinite(cot).all() or (cot < 0).any():
            raise RuntimeError(f"Invalid predicted COT at rows {start}:{stop}")
        out[start:stop] = cot.cpu().numpy().reshape(stop - start, 16, 1, 16, 16)
        print(f"fold{a.fold} OOF forecast-COT: {stop}/{count}", flush=True)
    out.flush()
    del out

    output_array = a.output / "forecast_cot.npy"
    report = {
        "state": f"HUBEI_SIMVP_OOF_FOLD{a.fold}_FORECAST_COT_BANK_TEST_CLOSED" if a.limit == 0 else f"HUBEI_SIMVP_OOF_FOLD{a.fold}_FORECAST_COT_SMOKE_TEST_CLOSED",
        "fold": a.fold,
        "scope": "training-date held-out OOF forecast COT; not independent test or GHI skill",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "ghi_labels_read": False,
        "r_reference_used_for_inference": False,
        "forecast_agri_bank_report_sha256": sha256(a.forecast_bank / "BANK_REPORT.json"),
        "forecast_agri_index_sha256": bank_report.get("canonical_oof_index_sha256") or bank_report.get("index_sha256"),
        "canonical_oof_index_file_sha256": sha256(a.oof_index),
        "forecast_agri_source_files_sha256": {name: sha256(a.forecast_bank / "val" / name) for name in ("keys.csv", "pred_agri_patch.npy", "future_geometry_patch.npy")},
        "forecast_agri_arrays_sha256": bank_report.get("files_sha256"),
        "simvp_checkpoint_sha256": bank_report["checkpoint_sha256"],
        "simvp_train_only_normalization_sha256": bank_report["normalization_sha256"],
        "fold_assignment_sha256": sha256(a.fold_assignment),
        "fold_R_run_audit_sha256": sha256(a.r_run / "audit_report.json"),
        "fold_R_checkpoint_sha256": sha256(a.r_run / "best.pt"),
        "fold_R_model_source_sha256": sha256(a.model_source),
        "fold_R_pack_norm_sha256": sha256(a.r_pack / "norm.json"),
        "fold_R_pack_rows_sha256": sha256(a.r_pack / "rows.csv"),
        "train_forecast_predictions_are_oof": True,
        "oof_sequences_processed": count,
        "oof_sequences_available": len(keys),
        "lead_minutes": list(range(15, 241, 15)),
        "initialization": "historical sequence end; target times are init+15..240 minutes in BJT",
        "COT_semantics": "fold-specific R applied to SimVP forecast AGRI at each future frame; zero sentinel where day_mask<=0.5",
        "daymask_off_pixels": day_off,
        "pixel_count": pixel_total,
        "prediction_shape": list(np.load(output_array, mmap_mode="r", allow_pickle=False).shape),
        "prediction_finite_nonnegative": True,
        "prediction_sha256": sha256(output_array),
        "keys_sha256": sha256(a.output / "keys.csv"),
        "full_bank_required_before_GHI_training": True,
        "inference_script_sha256": sha256(Path(__file__)),
    }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
