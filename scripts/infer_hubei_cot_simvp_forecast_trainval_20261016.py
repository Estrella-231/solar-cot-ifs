#!/usr/bin/env python3
"""Run the frozen Hubei COT-R on SimVP forecast AGRI, train/val only.

Input is a merged 8-to-16 SimVP forecast bank. This is intentionally separate
from the existing same-time real-AGRI COT bank, and never opens test/buffer
payloads or GHI labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


# Hash of the frozen sequence-index CSV used by this S run (not the day-assignment hash).
EXPECTED_SIMVP_INDEX_SHA = "4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62"
EXPECTED_SEQUENCES = {"train": 4086, "val": 407}
EXPECTED_INPUT_CHANNELS = [
    "C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10", "C11",
    "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stage_lineage(stage_report_path: Path, forecast_bank_report: dict) -> dict:
    """Bind the canonical forecast bank and its local path-remapped stage."""
    stage = json.loads(stage_report_path.read_text(encoding="utf-8"))
    if stage.get("status") != "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED":
        raise ValueError("Local stage report is not train/validation-only")
    if stage.get("source_index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Stage report is not bound to the frozen Hubei sequence index")
    # The bank retains canonical keys/indexing; stage_index is a separate
    # path-rewritten CSV used only to resolve local payloads. They must not be
    # compared to each other as if they were the same hash.
    if forecast_bank_report.get("index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Forecast bank is not bound to the frozen canonical sequence index")
    if forecast_bank_report.get("canonical_index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Merged bank does not record the canonical index lineage")
    if forecast_bank_report.get("source_stage_report_sha256") != sha256(stage_report_path):
        raise ValueError("Merged bank was not produced from the supplied stage report")
    if forecast_bank_report.get("source_stage_index_sha256") != stage.get("stage_index_sha256"):
        raise ValueError("Merged bank path-remapped stage index lineage mismatch")
    if forecast_bank_report.get("source_stage_manifest_sha256") != stage.get("stage_manifest_sha256"):
        raise ValueError("Merged bank stage-manifest lineage mismatch")
    if forecast_bank_report.get("stage_root") != stage.get("stage_root"):
        raise ValueError("Merged bank stage root differs from the supplied report")
    if stage.get("sequences") != EXPECTED_SEQUENCES or stage.get("test_sequences") != 0:
        raise ValueError("Unexpected staged train/validation sequence counts")
    if stage.get("test_or_buffer_payloads_staged") != 0:
        raise ValueError("Test/buffer payloads were staged")
    if sha256(Path(stage["source_index"])) != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Frozen canonical sequence-index checksum mismatch")
    if sha256(Path(stage["stage_index"])) != stage.get("stage_index_sha256"):
        raise ValueError("Path-rewritten stage index checksum mismatch")
    return stage


def select_shard_rows(rows: list[dict[str, str]], shard_index: int,
                      shard_count: int) -> list[tuple[int, dict[str, str]]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must satisfy 0 <= index < shard count")
    return [(i, row) for i, row in enumerate(rows) if i % shard_count == shard_index]


def prepare_r_inputs(pred_agri: np.ndarray, future_geometry: np.ndarray,
                     mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one batch of forecast sequences to normalized R frame inputs.

    Args:
        pred_agri: [B,16,13,16,16], de-normalized physical AGRI values.
        future_geometry: [B,16,3,16,16], physical [cosSOZ,cosRAA,day_mask].
        mean/std: [16], fixed-v2 R normalization fitted on train only.
    Returns:
        normalized R input [B*16,16,16,16] and solar-valid mask [B*16,1,16,16].
    """
    pred_agri = np.asarray(pred_agri, dtype=np.float32)
    future_geometry = np.asarray(future_geometry, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if pred_agri.ndim != 5 or pred_agri.shape[1:] != (16, 13, 16, 16):
        raise ValueError(f"bad forecast AGRI shape: {pred_agri.shape}")
    if future_geometry.shape != (len(pred_agri), 16, 3, 16, 16):
        raise ValueError(f"bad future geometry shape: {future_geometry.shape}")
    if mean.shape != (16,) or std.shape != (16,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("bad frozen R mean/std")
    if not np.isfinite(pred_agri).all() or not np.isfinite(future_geometry).all():
        raise ValueError("non-finite forecast inputs")

    raw = np.concatenate((pred_agri, future_geometry), axis=2)
    raw = raw.reshape(-1, 16, 16, 16)
    solar_valid = raw[:, 15:16] > 0.5
    normalized = (raw - mean[None, :, None, None]) / std[None, :, None, None]
    # The R training set has day_mask==1 and a tiny fitted std. Use the same
    # neutral normalized value outside daytime support as the audited full-bank
    # real-AGRI inference; the R output there is a zero sentinel, not a retrieval.
    normalized[:, 15:16] = np.where(solar_valid, normalized[:, 15:16], 0.0)
    if not np.isfinite(normalized).all():
        raise ValueError("non-finite normalized R inputs")
    return normalized.astype(np.float32), solar_valid


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", required=True, type=Path,
                   help="Merged SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK")
    p.add_argument("--stage-report", required=True, type=Path,
                   help="Immutable report binding the local path-rewritten index to the frozen source index")
    p.add_argument("--r-norm", required=True, type=Path,
                   help="norm.json from the frozen fixed-v2 R train/val pack")
    p.add_argument("--r-run", required=True, type=Path,
                   help="Completed validation-only frozen R run containing best.pt and metadata")
    p.add_argument("--model-source", required=True, type=Path,
                   help="Exact model source used to train frozen R")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sequence-batch", type=int, default=8,
                   help="Sequences per batch; expands to 16*sequence_batch R frames")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available() or not a.device.startswith("cuda"):
        raise RuntimeError("Forecast COT inference requires the allocated CUDA GPU; CPU fallback disabled")
    if a.sequence_batch < 1:
        raise ValueError("sequence-batch must be positive")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.output}")
    bank_report_path = a.forecast_bank / "BANK_REPORT.json"
    bank = json.loads(bank_report_path.read_text(encoding="utf-8"))
    if (bank.get("state") != "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK"
            or bank.get("test_payloads_read") != 0
            or bank.get("buffer_payloads_read") != 0
            or bank.get("test_or_buffer_rows") != 0):
        raise ValueError("Merged forecast AGRI bank is not train/validation-only")
    stage = validate_stage_lineage(a.stage_report, bank)
    for split, expected_n in EXPECTED_SEQUENCES.items():
        if int(bank.get("splits", {}).get(split, {}).get("sequences", -1)) != expected_n:
            raise ValueError(f"Unexpected forecast bank sequence count for {split}")

    status = json.loads((a.r_run / "status.json").read_text(encoding="utf-8"))
    run_meta = json.loads((a.r_run / "run_metadata.json").read_text(encoding="utf-8"))
    norm = json.loads(a.r_norm.read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or status.get("test_payloads_read") != 0:
        raise ValueError("Frozen R run is not complete/test-closed")
    if run_meta.get("test_used") is not False or run_meta.get("model_source_sha256") != sha256(a.model_source):
        raise ValueError("Frozen R provenance mismatch")
    if norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("R normalization is not train-only")
    if norm.get("input_channels") != EXPECTED_INPUT_CHANNELS:
        raise ValueError("Unexpected frozen R input channel order")
    if run_meta.get("normalization") != norm:
        raise ValueError("R run metadata and supplied normalization disagree")
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    if mean.shape != (16,) or std.shape != (16,) or np.any(std <= 0):
        raise ValueError("Invalid frozen R statistics")

    spec = importlib.util.spec_from_file_location("hubei_frozen_R_forecast_inference", a.model_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import R model source: {a.model_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device(a.device)
    model = module.COTUNet().to(device).eval()
    checkpoint = torch.load(a.r_run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    a.output.mkdir(parents=True, exist_ok=False)
    split_summary = {}
    input_reports = []
    cot_hashes = {}
    for split in ("train", "val"):
        source = a.forecast_bank / split
        keys = read_rows(source / "keys.csv")
        pred = np.load(source / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
        geom = np.load(source / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
        expected_n = int(bank.get("splits", {}).get(split, {}).get("sequences", -1))
        if len(keys) != expected_n or pred.shape != (expected_n, 16, 13, 16, 16) or geom.shape != (expected_n, 16, 3, 16, 16):
            raise ValueError(f"Forecast bank keys/array shapes disagree for {split}")
        if any(r.get("split") != split for r in keys):
            raise ValueError(f"Wrong split key in {split} forecast bank")
        for row in keys:
            if len(row.get("target_times_bjt", "").split("|")) != 16:
                raise ValueError(f"Expected 16 target timestamps for {split} row {row.get('sample_index')}")
        selected = select_shard_rows(keys, a.shard_index, a.shard_count)
        outdir = a.output / split
        outdir.mkdir()
        cot_out = np.lib.format.open_memmap(
            outdir / "forecast_cot.npy", mode="w+", dtype=np.float32,
            shape=(len(selected), 16, 1, 16, 16),
        )
        with (outdir / "keys.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths"])
            writer.writeheader()
            writer.writerows(row for _, row in selected)

        daymask_off = 0
        total_pixels = 0
        for start in range(0, len(selected), a.sequence_batch):
            batch_rows = selected[start:start + a.sequence_batch]
            ids = [idx for idx, _ in batch_rows]
            normalized, solar_valid = prepare_r_inputs(np.asarray(pred[ids]), np.asarray(geom[ids]), mean, std)
            daymask_off += int((~solar_valid).sum())
            total_pixels += int(solar_valid.size)
            x = torch.from_numpy(normalized).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_log = model(x).float()
                if not torch.isfinite(pred_log).all():
                    raise RuntimeError(f"Nonfinite log1p(COT) at {split} sequence shard rows {start}:{start+len(ids)}")
                cot = torch.expm1(pred_log)
                solar = torch.from_numpy(solar_valid).to(device, non_blocking=True)
                cot = cot.masked_fill(~solar, 0.0)
            if not torch.isfinite(cot).all() or (cot < 0).any():
                raise RuntimeError(f"Invalid COT prediction at {split} shard rows {start}:{start+len(ids)}")
            cot_np = cot.cpu().numpy().reshape(len(ids), 16, 1, 16, 16).astype(np.float32)
            cot_out[start:start + len(ids)] = cot_np
            if start + len(ids) == len(selected) or (start + len(ids)) % 128 == 0:
                print(f"{split}: {start + len(ids)}/{len(selected)} sequences", flush=True)
        cot_out.flush()
        del cot_out
        outpath = outdir / "forecast_cot.npy"
        cot_hashes[f"{split}/forecast_cot.npy"] = sha256(outpath)
        split_summary[split] = {
            "source_sequences": expected_n,
            "sequences_written": len(selected),
            "shard_index": a.shard_index,
            "shard_count": a.shard_count,
        }
        input_reports.append({"split": split, "day_mask_off_pixels": daymask_off,
                              "pixels_total": total_pixels})

    report = {
        "state": "SIMVP_FORECAST_AGRI_TO_FROZEN_R_COT_TRAINVAL_SHARD",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "ghi_labels_read": False,
        "source_bank_state": bank["state"],
        "source_index_sha256": bank["index_sha256"],
        "canonical_source_index_sha256": stage["source_index_sha256"],
        "source_stage_report_sha256": sha256(a.stage_report),
        "source_stage_index_sha256": stage["stage_index_sha256"],
        "source_stage_manifest_sha256": stage["stage_manifest_sha256"],
        "source_simvp_checkpoint_sha256": bank["checkpoint_sha256"],
        "source_simvp_normalization_sha256": bank["normalization_sha256"],
        "source_forecast_bank_report_sha256": sha256(bank_report_path),
        "frozen_R_checkpoint_sha256": sha256(a.r_run / "best.pt"),
        "R_model_source_sha256": sha256(a.model_source),
        "R_norm_sha256": sha256(a.r_norm),
        "R_train_only_norm": True,
        "forecast_leads_min": list(range(15, 241, 15)),
        "initialization": "last historical frame timestamp; target times copied from canonical SimVP bank keys",
        "COT_support": "R output at future forecast-AGRI frames; zero sentinel where future day_mask <= 0.5",
        "train_predictions_are_oof": False,
        "shard_index": a.shard_index,
        "shard_count": a.shard_count,
        "splits": split_summary,
        "day_mask_support": input_reports,
        "prediction_shape_per_split": ["N_shard", 16, 1, 16, 16],
        "prediction_finite_nonnegative": True,
        "prediction_sha256": cot_hashes,
        "cpp_reference_is_not_independent_truth": True,
        "note": "This is forecast AGRI -> frozen R inference, distinct from the real-AGRI same-time COT bank. Train forecasts are not OOF; assess train/val distribution before GHI head training.",
    }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
