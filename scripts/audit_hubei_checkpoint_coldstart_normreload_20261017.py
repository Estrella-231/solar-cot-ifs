#!/usr/bin/env python3
"""Cold-start validation inference from checkpoint-bound normalization.

The evaluator deliberately rebuilds normalized validation inputs from raw pack
arrays and each loaded checkpoint's station_normalization. It never opens test
or buffer payloads and writes a parity receipt for every seed/station/arm.
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

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = {"same_time": ("A", "AC", "AP"), "forecast": ("FA", "FAC")}
PACK_FLOAT_ATOL = 5e-5  # W/m^2: pack arrays are float32; bundles preserve float64 values.


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj: object) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_model_class(source: Path):
    """Load the model class from the exact trainer source used for the run."""
    spec = importlib.util.spec_from_file_location("hubei_normreload_trainer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HubeiLocalGHIHead


def parse_runs(items: list[str]) -> dict[int, Path]:
    out = {}
    for item in items:
        seed_s, sep, path_s = item.partition("=")
        if not sep:
            raise ValueError("--run must be SEED=RUN_DIRECTORY")
        seed = int(seed_s)
        if seed in out:
            raise ValueError(f"duplicate seed {seed}")
        out[seed] = Path(path_s)
    if not out:
        raise ValueError("at least one --run is required")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--input-mode", choices=tuple(ARMS), required=True)
    p.add_argument("--trainer-source", type=Path, required=True,
                   help="Exact trainer source whose SHA is recorded in run_metadata.json")
    p.add_argument("--run", action="append", default=[], help="SEED=RUN_DIRECTORY; repeat for each seed")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run through an allocated GPU job")
    device = torch.device(a.device)
    runs = parse_runs(a.run)
    arms = ARMS[a.input_mode]
    trainer_sha = sha(a.trainer_source)
    HubeiLocalGHIHead = load_model_class(a.trainer_source)

    pack_report_path = a.pack / "pack_report.json"
    pack_report_sha = sha(pack_report_path)
    report_meta = json.loads(pack_report_path.read_text(encoding="utf-8"))
    if report_meta.get("input_mode", a.input_mode) != a.input_mode:
        raise AssertionError("input mode does not match pack report")
    if report_meta.get("test_and_buffer_values_read", 0) != 0:
        raise AssertionError("pack report indicates test/buffer reads")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(encoding="utf-8-sig", newline="")))
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise AssertionError("pack contains a non-train/validation split; refusing to load")
    split = np.asarray([r["split"] for r in rows])
    stations = np.asarray([r["station"] for r in rows])
    agri = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(a.pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(a.pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    n = len(rows)
    if agri.shape != (n, 13, 3, 3) or geometry.shape != (n, 2, 3, 3) or cot.shape != (n, 1, 3, 3):
        raise AssertionError("raw feature shape mismatch")
    if not all(np.isfinite(x).all() for x in (agri, geometry, cot, clear, truth)):
        raise AssertionError("non-finite train/validation pack data")

    array_hashes = {name: sha(a.pack / f"{name}.npy") for name in
                    ("agri_raw", "geometry", "cot", "clear_sky_wm2", "ghi_wm2")}
    result_rows = []
    provenance = {}
    for seed, run_dir in sorted(runs.items()):
        run_meta_path = run_dir / "run_metadata.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta.get("seed") != seed or run_meta.get("input_mode") != a.input_mode:
            raise AssertionError(f"run metadata mismatch for seed {seed}")
        if run_meta.get("pack_sha256") != pack_report_sha:
            raise AssertionError(f"pack report SHA mismatch for seed {seed}")
        if run_meta.get("trainer_sha256") != trainer_sha:
            raise AssertionError(f"trainer source SHA mismatch for seed {seed}: {trainer_sha}")
        if run_meta.get("test_payloads_read") != 0:
            raise AssertionError(f"test payload read count is not zero for seed {seed}")
        batch_size = int(run_meta.get("batch_size", 1536))
        if batch_size <= 0:
            raise AssertionError(f"invalid inference batch_size for seed {seed}: {batch_size}")
        norm_path = run_dir / "normalization_contract.json"
        norm_doc = json.loads(norm_path.read_text(encoding="utf-8"))
        norm_payload = norm_doc["stations"]
        norm_sha = canonical_sha(norm_payload)
        if norm_doc.get("sha256") != norm_sha or run_meta.get("normalization_contract_sha256") != norm_sha:
            raise AssertionError(f"normalization contract hash mismatch for seed {seed}")
        if run_meta.get("arms") != list(arms):
            raise AssertionError(f"arm contract mismatch for seed {seed}")
        provenance[str(seed)] = {
            "run_metadata_sha256": sha(run_meta_path),
            "normalization_contract_sha256": norm_sha,
            "normalization_contract_file_sha256": sha(norm_path),
            "validation_inference_batch_size": batch_size,
        }

        for st in STATIONS:
            all_ids = np.flatnonzero(stations == st)
            val_ids = all_ids[split[all_ids] == "val"]
            if not len(val_ids):
                raise AssertionError(f"no validation rows for {seed}/{st}")
            station_norm = norm_payload[st]
            x = np.asarray(agri[val_ids], dtype=np.float32)
            g = np.asarray(geometry[val_ids], dtype=np.float32)
            x_mean = np.asarray(station_norm["agri_mean"], dtype=np.float32)
            x_std = np.asarray(station_norm["agri_std"], dtype=np.float32)
            x = (x - x_mean[None, :, None, None]) / x_std[None, :, None, None]
            x = np.concatenate((x, g), axis=1).astype(np.float32, copy=False)
            clog = np.log1p(np.asarray(cot[val_ids], dtype=np.float32))
            clog = ((clog - np.float32(station_norm["cot_log1p_mean"])) /
                    np.float32(station_norm["cot_log1p_std"])).astype(np.float32, copy=False)
            X = torch.from_numpy(x.copy()).to(device)
            C = torch.from_numpy(clog.copy()).to(device)
            CL = torch.from_numpy(np.asarray(clear[val_ids], dtype=np.float32).copy()).to(device)
            pred = {}
            for arm in arms:
                checkpoint_path = run_dir / st / f"{arm}_best.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if checkpoint.get("station") != st or checkpoint.get("arm") != arm or checkpoint.get("seed") != seed:
                    raise AssertionError(f"checkpoint identity mismatch: {seed}/{st}/{arm}")
                if checkpoint.get("normalization_contract_sha256") != norm_sha:
                    raise AssertionError(f"checkpoint normalization SHA mismatch: {seed}/{st}/{arm}")
                if checkpoint.get("station_normalization") != station_norm:
                    raise AssertionError(f"checkpoint station normalization differs: {seed}/{st}/{arm}")
                model = HubeiLocalGHIHead(arm).to(device).eval()
                model.load_state_dict(checkpoint["model"], strict=True)
                with torch.inference_mode():
                    # Match the training-time validation batch partition.
                    # Different CUDA kernel shapes can otherwise introduce drift.
                    pieces = []
                    for offset in range(0, len(val_ids), batch_size):
                        sl = slice(offset, offset + batch_size)
                        cot_batch = C[sl] if arm in ("AC", "FAC") else None
                        pieces.append((model(X[sl], cot_batch) * CL[sl]).detach().cpu().numpy())
                    pred[arm] = np.concatenate(pieces).astype(np.float64)
                bundle_path = run_dir / st / f"{arm}_validation_predictions_with_keys.npz"
                with np.load(bundle_path, allow_pickle=False) as z:
                    stored_pred = np.asarray(z["pred_ghi_wm2"], dtype=np.float64)
                    stored_truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
                    stored_clear = np.asarray(z["clear_sky_ghi_wm2"], dtype=np.float64)
                    stored_station = z["station"].astype(str)
                    stored_target = z["target_time_bjt"].astype(str)
                    expected_target = np.asarray([r.get("target_time_bjt", r.get("time_bjt")) for r in np.asarray(rows, dtype=object)[val_ids]], dtype=str)
                    expected_station = stations[val_ids]
                    if "init_time_bjt" in z:
                        stored_init = z["init_time_bjt"].astype(str)
                        stored_lead = np.asarray(z["lead_minutes"], dtype=np.int32)
                        expected_init = np.asarray([r["init_time_bjt"] for r in np.asarray(rows, dtype=object)[val_ids]], dtype=str)
                        expected_lead = np.asarray([int(r["lead_minutes"]) for r in np.asarray(rows, dtype=object)[val_ids]], dtype=np.int32)
                        if not (np.array_equal(stored_init, expected_init) and np.array_equal(stored_lead, expected_lead)):
                            raise AssertionError(f"forecast keys differ: {seed}/{st}/{arm}")
                if not (np.array_equal(stored_station, expected_station) and np.array_equal(stored_target, expected_target)):
                    raise AssertionError(f"validation keys differ: {seed}/{st}/{arm}")
                truth_delta = float(np.max(np.abs(stored_truth - np.asarray(truth[val_ids], dtype=np.float64))))
                clear_delta = float(np.max(np.abs(stored_clear - np.asarray(clear[val_ids], dtype=np.float64))))
                if not np.allclose(stored_truth, np.asarray(truth[val_ids], dtype=np.float64), rtol=0.0,
                                   atol=PACK_FLOAT_ATOL):
                    raise AssertionError(f"validation truth differs beyond float32 tolerance: {seed}/{st}/{arm}: {truth_delta}")
                if not np.allclose(stored_clear, np.asarray(clear[val_ids], dtype=np.float64), rtol=0.0,
                                   atol=PACK_FLOAT_ATOL):
                    raise AssertionError(f"validation clear-sky differs beyond float32 tolerance: {seed}/{st}/{arm}: {clear_delta}")
                delta = float(np.max(np.abs(pred[arm] - stored_pred)))
                if not np.allclose(pred[arm], stored_pred, rtol=1e-6, atol=1e-4):
                    raise AssertionError(f"cold-start prediction mismatch {seed}/{st}/{arm}: {delta}")
                result_rows.append({"seed": seed, "station": st, "arm": arm,
                                    "n": int(len(val_ids)), "max_abs_delta_wm2": delta,
                                    "max_abs_truth_pack_rounding_wm2": truth_delta,
                                    "max_abs_clear_sky_pack_rounding_wm2": clear_delta,
                                    "truth_clear_sky_pack_atol_wm2": PACK_FLOAT_ATOL,
                                    "validation_inference_batch_size": batch_size,
                                    "checkpoint_sha256": sha(checkpoint_path),
                                    "prediction_bundle_sha256": sha(bundle_path),
                                    "cold_start_normalization_from_checkpoint": True})
                del model

    report = {
        "state": "HUBEI_CHECKPOINT_COLDSTART_NORMRELOAD_AUDIT_PASS_TEST_CLOSED",
        "input_mode": a.input_mode,
        "seeds": sorted(runs),
        "arms": list(arms),
        "pack_report_sha256": pack_report_sha,
        "pack_rows_sha256": sha(a.pack / "rows.csv"),
        "pack_array_sha256": array_hashes,
        "trainer_source": str(a.trainer_source),
        "trainer_source_sha256": trainer_sha,
        "run_provenance": provenance,
        "results": result_rows,
        "n_checkpoint_bundle_pairs": len(result_rows),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "device": str(device),
        "method": "for each loaded checkpoint, verify serialized train-only normalization; rebuild X/C from raw pack arrays using that checkpoint's normalization; rerun validation; compare keys and predictions against saved bundle; compare truth and clear-sky arrays within float32 pack-rounding tolerance",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "input_mode": a.input_mode,
                      "n_checkpoint_bundle_pairs": len(result_rows), "test_payloads_read": 0,
                      "max_abs_delta_wm2": max(x["max_abs_delta_wm2"] for x in result_rows)}, indent=2))


if __name__ == "__main__":
    main()
