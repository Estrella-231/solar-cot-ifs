#!/usr/bin/env python3
"""Repeat the frozen val bank and additionally save station-centered forecast AGRI 3x3 crops."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from replay_forecast_cot_witness import (
    CHANNELS, EXPECTED, STATIONS, key_from_rel, sha,
)


def main():
    p = argparse.ArgumentParser()
    for name in ("solar_root", "data_root", "manifest", "norm", "simvp_checkpoint",
                 "simvp_source", "r_checkpoint", "r_source", "r_norm", "r_pack_rows",
                 "r_validation_predictions", "candidate", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--num-windows", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--full-bank", action="store_true")
    p.add_argument("--reference-bank", type=Path, required=True)
    p.add_argument("--expected-simvp-sha", required=True)
    p.add_argument("--geometry-contract-receipt", type=Path, required=True)
    a = p.parse_args()
    if len(a.expected_simvp_sha) != 64 or any(c not in "0123456789abcdef" for c in a.expected_simvp_sha.lower()):
        raise ValueError("--expected-simvp-sha must be a SHA-256 digest")
    EXPECTED["simvp"] = a.expected_simvp_sha.lower()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded GPU profile")
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")
    if not a.full_bank and not (1 <= a.num_windows <= 100):
        raise ValueError("this profile is capped at 100 windows")
    if a.full_bank and a.num_windows < 0:
        raise ValueError("--num-windows must be 0 (all) or a positive bank smoke limit")

    expected = ((a.manifest, EXPECTED["manifest"]), (a.norm, EXPECTED["s_norm"]),
                (a.simvp_checkpoint, EXPECTED["simvp"]),
                (a.simvp_source / "triad_models.py", EXPECTED["triad_models"]),
                (a.simvp_source / "triad_data.py", EXPECTED["triad_data"]),
                (a.simvp_source / "hunan_data.py", EXPECTED["hunan_data"]),
                (a.r_checkpoint, EXPECTED["r_checkpoint"]), (a.r_source, EXPECTED["r_source"]))
    for path, digest in expected:
        got = sha(path)
        if got != digest:
            raise RuntimeError(f"source/checkpoint hash mismatch: {path}: {got} != {digest}")

    import sys
    sys.path.insert(0, str(a.simvp_source))
    sys.path.insert(0, str(a.r_source.parent))
    from triad_data import ForecastHunanDataset
    from triad_models import SimVPBackbone
    from train_station16_R import COTUNet

    norm = json.loads(a.norm.read_text())
    s_mean = torch.as_tensor(norm["mean"], device="cuda", dtype=torch.float32)[None, None, :, None, None]
    s_std = torch.as_tensor(norm["std"], device="cuda", dtype=torch.float32)[None, None, :, None, None]
    ck = torch.load(a.simvp_checkpoint, map_location="cpu", weights_only=False)
    meta = ck["metadata"]
    geometry_receipt = json.loads(a.geometry_contract_receipt.read_text(encoding="utf-8"))
    geometry_receipt_sha256 = sha(a.geometry_contract_receipt)
    if (geometry_receipt.get("state") != "PASS_SOURCE_VERIFIED_GEOMETRY_CONTRACT" or
            geometry_receipt.get("geometry_contract") != "history_8_then_target_16_v2" or
            geometry_receipt.get("test_payloads_read") != 0 or
            geometry_receipt.get("hashes", {}).get("manifest") != EXPECTED["manifest"] or
            geometry_receipt.get("hashes", {}).get("normalization") != EXPECTED["s_norm"] or
            geometry_receipt.get("checkpoints", {}).get(Path(a.simvp_checkpoint).parent.name, {}).get("sha256")
            != EXPECTED["simvp"]):
        raise RuntimeError("source-backed geometry receipt does not match this checkpoint/contract")
    if (meta.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1" or
            meta.get("manifest_sha256") != EXPECTED["manifest"] or
            meta.get("normalization_sha256") != EXPECTED["s_norm"] or
            meta.get("geometry_contract") not in (None, "history_8_then_target_16_v2") or
            meta.get("test_payloads_read") != 0):
        raise RuntimeError("SimVP checkpoint metadata conflicts with frozen val contract")
    simvp = SimVPBackbone(a.simvp_source).cuda().eval().requires_grad_(False)
    simvp.load_state_dict(ck["model"], strict=True)
    del ck
    rck = torch.load(a.r_checkpoint, map_location="cpu", weights_only=False)
    if rck["metadata"].get("test_payloads_read") != 0:
        raise RuntimeError("R checkpoint metadata reports test payload access")
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(rck["model"], strict=True)
    del rck
    rnorm = json.loads(a.r_norm.read_text())
    rmean = torch.as_tensor(rnorm["mean"], device="cuda", dtype=torch.float32)[None, :, None, None]
    rstd = torch.as_tensor(rnorm["std"], device="cuda", dtype=torch.float32)[None, :, None, None]

    with a.r_pack_rows.open(newline="", encoding="utf-8") as f:
        supervised = [r for r in csv.DictReader(f) if r["state"] == "SUPERVISED"]
    with np.load(a.r_validation_predictions, allow_pickle=False) as z:
        indices = z["sample_indices"].astype(np.int64)
        ref = z["reference_cot"].astype(np.float32)
        mask = z["mask"].astype(bool)
    saved = {int(i): j for j, i in enumerate(indices)}
    expected_val = {i for i, row in enumerate(supervised) if row["split"] == "val"}
    if len(saved) != len(indices) or set(saved) != expected_val:
        raise RuntimeError("saved R indices must use the full SUPERVISED row index, not validation-local positions")
    ref_keys = {}
    for idx, row in enumerate(supervised):
        if row["split"] == "val" and idx in saved:
            j = saved[idx]
            ref_keys[(row["station"], row["target_time_utc"])] = (ref[j, 0], mask[j, 0])
    with a.candidate.open(newline="", encoding="utf-8-sig") as f:
        candidate = [r for r in csv.DictReader(f) if r["split"] == "val"]
    station_coords = {}
    for r in candidate:
        station_coords.setdefault(r["station"], tuple(map(int, r["station_grid_rc"].split(","))))

    dataset = ForecastHunanDataset(a.simvp_source, a.data_root, a.manifest, norm, "val", include_index=True)
    selected = []
    for i, row in enumerate(dataset.base.rows):
        rels = row["data_relpaths"].split("|")
        if len(rels) != 24:
            continue
        keys = [key_from_rel(rel) for rel in rels[8:24]]
        # This diagnostic uses every validation sequence, irrespective of CPP
        # availability. CPP arrays remain sparse and never filter the AGRI->R comparison.
        selected.append(i)
    if len(selected) != len(dataset):
        raise RuntimeError(f"all-val selection mismatch: {len(selected)} != {len(dataset)}")
    if not selected:
        raise RuntimeError("no validation windows found")

    loader = DataLoader(Subset(dataset, selected), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
                        prefetch_factor=2 if a.workers > 0 else None)
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    io_seconds = simvp_seconds = r_seconds = 0.0
    windows = finite_windows = support_pixels = 0
    digest = hashlib.sha256()
    selected_ids = []
    forecast_chunks, target_chunks, persistence_chunks, agri_patch_chunks = [], [], [], []
    cpp_chunks, mask_chunks, key_chunks = [], [], []
    for seq_ids, history_np, geometry_np, future_np, future_valid_np in loader:
        t0 = time.perf_counter()
        # DataLoader has already fetched/collated this batch; account for wall time
        # via inter-batch timing below, while CUDA stage timings are synchronized.
        history = history_np.cuda(non_blocking=True)
        geometry = geometry_np.cuda(non_blocking=True)
        future = future_np.cuda(non_blocking=True)
        batch = history.shape[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_norm = simvp(history, geometry)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        pred = pred_norm.float() * s_std + s_mean
        target = future.float() * s_std + s_mean
        future_geo = geometry[:, 8:].float()
        hist_last = history[:, -1].float() * s_std[:, 0] + s_mean[:, 0]

        forecast_inputs, target_inputs, persistence_inputs = [], [], []
        batch_refs, batch_masks, batch_keys, batch_agri_patches = [], [], [], []
        for b in range(batch):
            row = dataset.base.rows[selected[windows + b]]
            rels = row["data_relpaths"].split("|")
            keys = [key_from_rel(rel) for rel in rels[8:24]]
            batch_keys.append(keys)
            for s, station in enumerate(STATIONS):
                for lead, key in enumerate(keys):
                    rr, cc = station_coords[station]
                    r0, c0 = rr - 8, cc - 8
                    batch_agri_patches.append(pred[b, lead, :, rr-1:rr+2, cc-1:cc+2])
                    forecast_inputs.append(torch.cat((pred[b, lead, :, r0:r0+16, c0:c0+16],
                                                      future_geo[b, lead, :, r0:r0+16, c0:c0+16])))
                    target_inputs.append(torch.cat((target[b, lead, :, r0:r0+16, c0:c0+16],
                                                    future_geo[b, lead, :, r0:r0+16, c0:c0+16])))
                    pair = ref_keys.get((station, key))
                    if pair is None:
                        cot = np.full((16, 16), np.nan, dtype=np.float32)
                        m = np.zeros((16, 16), dtype=bool)
                    else:
                        cot, m = pair
                    support_pixels += int(m.sum())
                    batch_refs.append(cot)
                    batch_masks.append(m)
                    if lead == 0:
                        # Persistence uses the final observed field and its geometry.
                        persistence_inputs.append(torch.cat((hist_last[b, :, r0:r0+16, c0:c0+16],
                                                             geometry[b, 7, :, r0:r0+16, c0:c0+16])))
        agri_patch_batch = torch.stack(batch_agri_patches).reshape(batch, 2, 16, 13, 3, 3).detach().cpu().numpy()
        forecast_x = (torch.stack(forecast_inputs) - rmean) / rstd
        target_x = (torch.stack(target_inputs) - rmean) / rstd
        persistence_x = (torch.stack(persistence_inputs) - rmean) / rstd
        forecast_x = torch.where(torch.isfinite(forecast_x), forecast_x, torch.zeros_like(forecast_x))
        target_x = torch.where(torch.isfinite(target_x), target_x, torch.zeros_like(target_x))
        persistence_x = torch.where(torch.isfinite(persistence_x), persistence_x, torch.zeros_like(persistence_x))
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            f = rmodel(forecast_x.float())
            tr = rmodel(target_x.float())
            pr = rmodel(persistence_x.float())
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        ok = (torch.isfinite(pred_norm).all() and torch.isfinite(f).all() and
              torch.isfinite(tr).all() and torch.isfinite(pr).all())
        finite_windows += int(bool(ok)) * batch
        windows += batch
        simvp_seconds += t2 - t1
        r_seconds += t3 - t2
        for sid in seq_ids.tolist():
            selected_ids.append(int(sid))
        digest.update(f.detach().cpu().numpy().tobytes())
        io_seconds += max(0.0, t1 - t0)
        if a.full_bank:
            forecast_chunks.append(f.detach().float().reshape(batch, 2, 16, 16, 16).cpu().numpy())
            agri_patch_chunks.append(agri_patch_batch)
            target_chunks.append(tr.detach().float().reshape(batch, 2, 16, 16, 16).cpu().numpy())
            persistence_chunks.append(pr.detach().float().reshape(batch, 2, 1, 16, 16)
                                      .expand(-1, -1, 16, -1, -1).cpu().numpy())
            cpp_chunks.append(np.asarray(batch_refs, dtype=np.float32).reshape(batch, 2, 16, 16, 16))
            mask_chunks.append(np.asarray(batch_masks, dtype=bool).reshape(batch, 2, 16, 16, 16))
            key_chunks.append(np.asarray(batch_keys, dtype="U19"))
        print(json.dumps({"completed_windows": windows, "batch_size": batch,
                          "simvp_seconds": round(simvp_seconds, 2),
                          "r_seconds": round(r_seconds, 2)}, separators=(",", ":")), flush=True)

    elapsed = time.perf_counter() - t_start
    if a.full_bank:
        forecast_np = np.concatenate(forecast_chunks, axis=0)
        agri_patch_np = np.concatenate(agri_patch_chunks, axis=0)
        keys_np = np.concatenate(key_chunks, axis=0)
        expected_keys = np.asarray([[key_from_rel(rel) for rel in dataset.base.rows[i]["data_relpaths"].split("|")[8:24]] for i in selected], dtype="U19")
        expected_ids = np.asarray([dataset.base.rows[i].get("seq_id", "") for i in selected], dtype="U64")
        if not (np.array_equal(keys_np, expected_keys) and
                np.array_equal(np.asarray(selected, dtype=np.int64), np.arange(len(dataset), dtype=np.int64)) and
                np.array_equal(expected_ids, np.asarray([dataset.base.rows[i].get("seq_id", "") for i in selected], dtype="U64"))):
            raise RuntimeError("all-validation forecast arrays do not match their own dataset rows")
        target_np = np.concatenate(target_chunks, axis=0)
        persistence_np = np.concatenate(persistence_chunks, axis=0)
        cpp_np = np.concatenate(cpp_chunks, axis=0)
        mask_np = np.concatenate(mask_chunks, axis=0)
        if not (len(forecast_np) == len(selected) == len(keys_np)):
            raise RuntimeError("forecast bank arrays and selected val keys are misaligned")
        if not (np.isfinite(forecast_np).all() and np.isfinite(target_np).all() and
                np.isfinite(persistence_np).all() and np.isfinite(cpp_np[mask_np]).all()):
            raise RuntimeError("full validation bank contains non-finite outputs/reference")
        if not np.isfinite(agri_patch_np).all():
            raise RuntimeError("forecast station AGRI crop contains non-finite values")
        a.output.mkdir(parents=True)
        np.savez_compressed(a.output / "forecast_cot_val_bank.npz",
                            forecast_log1p_cot=forecast_np,
                            target_agri_r_log1p_cot=target_np,
                            persistence_log1p_cot=persistence_np,
                            cpp_reference_cot=cpp_np, cpp_valid_mask=mask_np,
                            forecast_agri_station_3x3=agri_patch_np,
                            target_time_utc=keys_np,
                            dataset_indices=np.asarray(selected, dtype=np.int64),
                            sequence_ids=np.asarray([dataset.base.rows[i].get("seq_id", "")
                                                     for i in selected], dtype="U64"))
    a.output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "state": ("COMPLETE_ALL_VALIDATION_S_TO_R_COT_AND_AGRI3X3_BANK" if a.full_bank else
                  "PASS_100_CPP_SUPPORTED_VALIDATION_WINDOWS_S2R_PROFILE"),
        "split": "val_only", "test_payloads_read": 0,
        "windows_requested": ("all_validation_windows" if a.full_bank and a.num_windows == 0
                              else a.num_windows),
        "windows_completed": windows,
        "finite_windows": finite_windows, "batch_size": a.batch_size, "workers": a.workers,
        "simvp_seconds_cuda": simvp_seconds, "r_seconds_cuda": r_seconds,
        "end_to_end_wall_seconds": elapsed,
        "windows_per_second_end_to_end": windows / elapsed,
        "windows_per_second_cuda_only": windows / max(simvp_seconds + r_seconds, 1e-9),
        "cpp_common_support_pixels": support_pixels,
        "forecast_shape": list(forecast_np.shape) if a.full_bank else None,
        "forecast_agri_station_3x3_shape": list(agri_patch_np.shape) if a.full_bank else None,
        "forecast_agri_units": "physical AGRI channel values after inverse train normalization",
        "reference_bank_sha256": sha(a.reference_bank),
        "geometry_contract": "history_8_then_target_16_v2",
        "geometry_contract_evidence": "source-backed receipt; checkpoint field may be absent",
        "geometry_contract_receipt_sha256": geometry_receipt_sha256,
        "writer_script_sha256": sha(__file__),
        "cpp_mask_pixels": int(mask_np.sum()) if a.full_bank else None,
        "forecast_output_sha256": digest.hexdigest(),
        "selected_dataset_indices": selected,
        "selected_sequence_ids": [dataset.base.rows[i].get("seq_id") for i in selected],
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "manifest_sha256": sha(a.manifest), "simvp_checkpoint_sha256": sha(a.simvp_checkpoint),
        "r_checkpoint_sha256": sha(a.r_checkpoint),
        "future_agri_passed_to_simvp": False, "future_cpp_passed_to_simvp": False,
        "outputs": (["profile.json", "forecast_cot_val_bank.npz"] if a.full_bank else ["profile.json"]),
        "profile_scope": ("all validation windows; CPP is sparse ancillary reference, not a filter; metrics not computed" if a.full_bank else
                          "frozen SimVP forecast + new monthly R forecast/reference/persistence; no metrics, no tuning"),
    }
    (a.output / "profile.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()


