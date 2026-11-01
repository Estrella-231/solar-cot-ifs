"""Build a validation-only information-ceiling pack with observed-future AGRI→R COT.

Only future COT context slots are replaced. Forecast AGRI, geometry, labels,
sample keys, historical COT, and masks are shared byte-for-byte from the OOF
GHI pack. Test/buffer directories and payloads are never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


OFFSETS_MINUTES = (-30, -15, 0)
SPLITS = ("train", "val")
STATIONS = ("sili", "zhujia")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def target_key(station: str, timestamp: str) -> tuple[str, str]:
    # Validate and canonicalize to the exact UTC second string used by the packs.
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return station.lower(), dt.strftime("%Y-%m-%d %H:%M:%S")


def materialize_split(forecast_pack: Path, output: Path,
                      split: str, actual_index: dict[tuple[str, str], int],
                      actual_cot: np.ndarray, chunk_rows: int = 8192) -> dict:
    src = forecast_pack / split
    dst = output / split
    dst.mkdir(parents=True, exist_ok=False)
    rows = read_csv(src / "keys.csv")
    cot_src = np.load(src / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    expected = (len(rows), 3, 1, 3, 3)
    if cot_src.shape != expected or cot_src.dtype != np.float32:
        raise ValueError(f"{split} forecast COT expected float32 {expected}, got {cot_src.dtype} {cot_src.shape}")
    cot_out = np.lib.format.open_memmap(dst / "cot_log1p.npy", mode="w+", dtype=np.float32,
                                        shape=expected)
    replaced = 0
    for begin in range(0, len(rows), chunk_rows):
        end = min(begin + chunk_rows, len(rows))
        block = np.asarray(cot_src[begin:end], dtype=np.float32).copy()
        for local_i, row in enumerate(rows[begin:end]):
            if row.get("split") != split or row.get("station", "").lower() not in STATIONS:
                raise ValueError(f"unexpected key metadata in {split} row {begin + local_i}")
            lead = int(row["lead_minutes"])
            target = datetime.strptime(row["target_time_utc"], "%Y-%m-%d %H:%M:%S")
            for pos, offset in enumerate(OFFSETS_MINUTES):
                context_lead = lead + offset
                if context_lead <= 0:
                    continue
                timestamp = (target + timedelta(minutes=offset)).strftime("%Y-%m-%d %H:%M:%S")
                key = target_key(row["station"], timestamp)
                if key not in actual_index:
                    raise ValueError(f"missing same-split observed-AGRI→R COT context {split}/{key}")
                bank_index, station_index = actual_index[key]
                ref = np.asarray(actual_cot[bank_index, station_index, :, 7:10, 7:10], dtype=np.float32)
                if ref.shape != (1, 3, 3) or not np.isfinite(ref).all():
                    raise ValueError(f"invalid actual-R COT at {split}/{key}: {ref.shape}")
                block[local_i, pos] = ref
                replaced += 1
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite COT in {split} oracle pack")
        cot_out[begin:end] = block
    cot_out.flush()
    del cot_out
    # All non-COT payloads and keys are linked from the exact forecast pack.
    for name in ("agri_raw.npy", "geometry.npy", "ghi.npy", "clear.npy", "keys.csv"):
        try:
            os.symlink((src / name).resolve(), dst / name)
        except OSError:
            # Windows test environments may not grant symlink privileges;
            # production executes on Linux and retains the no-copy symlink path.
            shutil.copy2(src / name, dst / name)
    return {"rows": len(rows), "future_context_slots_replaced": replaced,
            "cot_log1p_sha256": sha(dst / "cot_log1p.npy"),
            "shared_source_sha256": {name: sha(src / name) for name in
                ("agri_raw.npy", "geometry.npy", "ghi.npy", "clear.npy", "keys.csv")}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-pack", type=Path, required=True)
    p.add_argument("--actual-r-bank", type=Path, required=True)
    p.add_argument("--actual-r-ghi-pack", type=Path, required=True,
                   help="compact labelled R pack used only as an exact crop/index witness")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-rows", type=int, default=8192)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.chunk_rows < 1:
        raise ValueError("chunk-rows must be positive")
    fm = json.loads((a.forecast_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    rm = json.loads((a.actual_r_bank / "bank_manifest.json").read_text(encoding="utf-8"))
    compact_manifest = json.loads((a.actual_r_ghi_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if fm.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("forecast GHI pack is incomplete")
    if fm.get("train_forecast_oof") is not True or fm.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("forecast AGRI must have five-fold date-block OOF provenance")
    if fm.get("test_payload_opened") is not False or fm.get("buffer_payload_opened") is not False:
        raise ValueError("forecast pack does not certify test/buffer closure")
    if fm.get("future_observed_agri_used_as_input") is not False or fm.get("cpp_used_as_input") is not False:
        raise ValueError("source forecast pack already includes a forbidden inference input")
    if rm.get("state") != "COMPLETE" or rm.get("split_id") != fm.get("split_id"):
        raise ValueError("actual-R bank incomplete or split mismatch")
    if rm.get("test_payload_opened") is not False:
        raise ValueError("actual-R bank does not certify test closure")
    if compact_manifest.get("state") != "COMPLETE" or compact_manifest.get("split_id") != fm.get("split_id"):
        raise ValueError("compact actual-R witness is incomplete or split mismatch")
    if compact_manifest.get("test_payload_opened") is not False:
        raise ValueError("compact actual-R witness does not certify test closure")
    frame_rows = read_csv(a.actual_r_bank / "frame_keys.csv")
    actual_cot = np.load(a.actual_r_bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if actual_cot.shape != (len(frame_rows), 2, 1, 16, 16) or actual_cot.dtype != np.float32:
        raise ValueError(f"actual-R bank expected float32 [frames,2,1,16,16], got {actual_cot.dtype} {actual_cot.shape}")
    actual_index_by_split: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    for split in SPLITS:
        index: dict[tuple[str, str], tuple[int, int]] = {}
        for row in frame_rows:
            if row.get("split") != split:
                continue
            frame_id = int(row["index"])
            for station_id, station in enumerate(STATIONS):
                key = target_key(station, row["target_time_utc"])
                if key in index:
                    raise ValueError(f"duplicate actual-R frame key {split}/{key}")
                index[key] = (frame_id, station_id)
        actual_index_by_split[split] = index

    # Verify the 3x3 extraction and bank indices against the already-produced
    # labelled pack for every common row; this prevents a silent spatial or
    # frame-index offset in the oracle reference.
    compact_rows = read_csv(a.actual_r_ghi_pack / "rows.csv")
    compact_cot = np.load(a.actual_r_ghi_pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if compact_cot.shape != (len(compact_rows), 1, 3, 3) or compact_cot.dtype != np.float32:
        raise ValueError("compact R witness has unexpected shape/dtype")
    for i, row in enumerate(compact_rows):
        split = row["split"]
        key = target_key(row["station"], row["target_time_utc"])
        mapped = actual_index_by_split.get(split, {}).get(key)
        expected = (int(row["bank_index"]), int(row["station_index"]))
        if mapped != expected:
            raise ValueError(f"compact-R bank index mismatch at row {i}: {mapped} != {expected}")
        bank_crop = np.asarray(actual_cot[expected[0], expected[1], :, 7:10, 7:10], dtype=np.float32)
        if not np.array_equal(bank_crop, compact_cot[i]):
            raise ValueError(f"compact-R 3x3 crop differs from frozen-R bank at row {i}")

    a.output.mkdir(parents=True)
    try:
        split_reports = {}
        for split in SPLITS:
            split_reports[split] = materialize_split(
                a.forecast_pack, a.output, split,
                actual_index_by_split[split], actual_cot, a.chunk_rows)
        train_cot = np.load(a.output / "train/cot_log1p.npy", mmap_mode="r", allow_pickle=False)
        # Match the standard GHI-pack convention: fit COT normalization on train
        # instances only, using all three 3x3 frames in the constructed pack.
        sum_ = sum_sq = count = 0.0
        for begin in range(0, len(train_cot), a.chunk_rows):
            x = np.asarray(train_cot[begin:begin+a.chunk_rows], dtype=np.float64)
            sum_ += float(x.sum()); sum_sq += float(np.square(x).sum()); count += x.size
        mean = sum_ / count
        std = max((sum_sq / count - mean * mean) ** 0.5, 1e-6)
        manifest = {
            "state": "COMPLETE_ORACLE_FUTURE_COT_GHI_PACK",
            "oracle_only": True, "deployable": False,
            "scope": "validation information-ceiling diagnostic; observed future AGRI is used only to create frozen-R COT oracle features",
            "split_id": fm["split_id"], "sample_key": fm["sample_key"],
            "rows": {s: split_reports[s]["rows"] for s in SPLITS},
            "stations": ["sili", "zhujia"], "lead_minutes": fm["lead_minutes"],
            "context_rule": fm["context_rule"],
            "input_sources": {**fm["input_sources"],
                "future_cot": "frozen R applied to observed AGRI at each future context time; oracle-only information ceiling",
                "observed_history_cot": "unchanged from causal forecast pack"},
            "future_agri_forecast_source": "five-fold date-block OOF SimVP predictions for train; fixed full-train SimVP predictions for validation",
            "train_forecast_oof": True,
            "forecast_bank_oof_fold_count": 5,
            "future_observed_agri_used_as_input": True,
            "future_cpp_used_as_input": False,
            "test_payload_opened": False, "buffer_payload_opened": False,
            "independent_test_read": False,
            "history_context_source_unchanged": True,
            "GHI_labels_and_sample_keys_unchanged": True,
            "COT_log1p_mean": mean, "COT_log1p_std": std,
            "AGRI_mean": fm["AGRI_mean"], "AGRI_std": fm["AGRI_std"],
            "forecast_pack_manifest_sha256": sha(a.forecast_pack / "pack_manifest.json"),
            "actual_R_bank_manifest_sha256": sha(a.actual_r_bank / "bank_manifest.json"),
            "actual_R_compact_pack_manifest_sha256": sha(a.actual_r_ghi_pack / "pack_manifest.json"),
            "actual_R_checkpoint_sha256": rm["R_checkpoint_sha256"],
            "actual_R_cot_array_sha256": sha(a.actual_r_bank / "cot_log1p.npy"),
            "actual_R_frame_keys_sha256": sha(a.actual_r_bank / "frame_keys.csv"),
            "actual_R_compact_cot_array_sha256": sha(a.actual_r_ghi_pack / "cot_log1p.npy"),
            "actual_R_compact_rows_sha256": sha(a.actual_r_ghi_pack / "rows.csv"),
            "split_reports": split_reports,
            "test_payloads_read": 0, "cpp_inputs_read": 0,
            "builder_sha256": sha(Path(__file__)),
        }
        (a.output / "pack_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # Keep partial output as an explicitly incomplete forensic artifact.
        (a.output / "pack_manifest.json").write_text(json.dumps({
            "state": "INCOMPLETE_ORACLE_FUTURE_COT_GHI_PACK",
            "test_payload_opened": False, "buffer_payload_opened": False,
        }, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"state": manifest["state"], "rows": manifest["rows"],
                      "future_context_slots_replaced": {s: split_reports[s]["future_context_slots_replaced"] for s in SPLITS},
                      "cot_log1p_norm_train_only": [mean, std],
                      "future_observed_agri_used_as_input": True,
                      "test_payload_opened": False, "buffer_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
