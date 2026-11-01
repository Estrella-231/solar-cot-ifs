#!/usr/bin/env python3
"""Train the frozen-input, four-arm paired station GHI experiment.

Only the registered train/validation arrays are read. The validation exclusion
is the user-confirmed Sili BJT 2024-09-05 invalid-observation day.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPLIT_ID = "hunan_monthly_block_2d_buffer1d_seed42_v1"
STATIONS = ("sili", "zhujia")
ARMS = ("M00", "M10", "M01", "M11")
STATION_INDEX = {"sili": 0, "zhujia": 1}
HISTORY_OFFSETS = (-30, -15, 0)
EXPECTED_COT_MEAN = 1.8354828989218208
EXPECTED_COT_STD = 1.3974485910139944
EXCLUDED_STATION_DAYS = {("sili", "2024-09-05")}


class FourArmIntervalHead(nn.Module):
    """17-slot counterpart of the registered 16-slot three-frame GHI head."""

    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv3d(17, 32, (3, 3, 3), padding=(0, 1, 1))
        self.spatial = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(32 * 3 * 3, 64), nn.SiLU(), nn.Linear(64, 1))
        with torch.no_grad():
            # Exact function match at initialization; both additional slots
            # remain trainable and receive gradients when present.
            self.temporal.weight[:, 15:17].zero_()

    def forward(self, x):
        # x: [B,T=3,C=17,H=3,W=3]
        h = x.permute(0, 2, 1, 3, 4)
        h = F.silu(self.temporal(h)).squeeze(2)
        h = F.silu(self.spatial(h))
        return F.softplus(self.fc(h.flatten(1)).squeeze(1))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bjt_date(utc_text: str) -> str:
    return (datetime.fromisoformat(utc_text) + timedelta(hours=8)).date().isoformat()


def make_inputs(pack: Path, rbank: Path, pbank: Path, exclusion_policy: Path, output: Path):
    gm = json.loads((pack / "pack_manifest.json").read_text(encoding="utf-8"))
    rm = json.loads((rbank / "bank_manifest.json").read_text(encoding="utf-8"))
    pm = json.loads((pbank / "bank_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads(exclusion_policy.read_text(encoding="utf-8"))
    policy_days = {(x["station"], x["bjt_date"]) for x in policy.get("exclusions", [])}
    if policy.get("date_timezone") != "Asia/Shanghai" or policy_days != EXCLUDED_STATION_DAYS:
        raise RuntimeError(f"station-day exclusion policy differs from the frozen list: {policy_days}")
    if not (gm.get("state") == "COMPLETE" and gm.get("split_id") == SPLIT_ID and gm.get("test_payload_opened") is False):
        raise RuntimeError("GHI pack is not the registered train/validation pack")
    if not (rm.get("state") == "COMPLETE_PAIRED_R_COT_HISTORY_BANK" and rm.get("split_id") == SPLIT_ID and rm.get("test_payloads_read") == 0 and rm.get("all_rows_common") is True and rm.get("all_R_frozen") is True):
        raise RuntimeError("paired R bank provenance gate failed")
    if not (pm.get("state") == "COMPLETE" and pm.get("split_id") == SPLIT_ID and pm.get("test_payload_opened") is False):
        raise RuntimeError("probability bank is not the registered train/validation bank")

    rows = load_csv(pack / "rows.csv")
    pkeys = load_csv(pbank / "frame_keys.csv")
    assignments = load_csv(pbank / "classifier_assignment.csv")
    if len(rows) != 51929 or sum(r["split"] == "train" for r in rows) != 47666 or sum(r["split"] == "val" for r in rows) != 4263:
        raise RuntimeError("unexpected source pack counts")
    if rm.get("rows") != len(rows) or rm.get("source_GHI_rows_sha256") != sha256(pack / "rows.csv"):
        raise RuntimeError("R bank was not built from this exact GHI row manifest")
    if {r["split"] for r in rows} != {"train", "val"}:
        raise RuntimeError("non train/validation split encountered")
    rowkeys = [(r["station"], r["target_time_utc"]) for r in rows]
    if len(set(rowkeys)) != len(rowkeys):
        raise RuntimeError("duplicate station-target rows")

    pindex = {r["target_time_utc"]: int(r["index"]) for r in pkeys}
    if len(pindex) != len(pkeys) or len(assignments) != len(pkeys):
        raise RuntimeError("duplicate or mismatched classifier frame keys")
    assignment_by_index = {int(r["frame_index"]): r for r in assignments}
    if len(assignment_by_index) != len(assignments):
        raise RuntimeError("duplicate classifier assignments")
    rkeys = load_csv(rbank / "frame_keys.csv")
    rindex = {(r["station"], r["target_time_utc"]): r for r in rkeys}
    if len(rindex) != len(rkeys):
        raise RuntimeError("duplicate R station-frame keys")

    n = len(rows)
    ag_raw = np.load(pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    ag_valid = np.load(pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    geo = np.load(pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(rbank / "cot_corrected_log1p.npy", mmap_mode="r", allow_pickle=False)
    pcloud = np.load(pbank / "p_cloud.npy", mmap_mode="r", allow_pickle=False)
    if ag_raw.shape != (n, 3, 13, 3, 3) or ag_valid.shape != ag_raw.shape or geo.shape != (n, 3, 2, 3, 3):
        raise RuntimeError("GHI input shape mismatch")
    if cot.shape != (n, 3, 1, 3, 3):
        raise RuntimeError(f"unexpected corrected R COT shape: {cot.shape}")
    if pcloud.shape != (len(pkeys), 2, 16, 16) or pcloud.dtype != np.float16:
        raise RuntimeError(f"unexpected p_cloud layout: {pcloud.shape}, {pcloud.dtype}")
    if not np.isfinite(ag_raw).all() or not np.isfinite(geo).all() or not np.isfinite(cot).all():
        raise RuntimeError("non-finite AGRI, geometry, or corrected COT input")

    pc = np.empty((n, 3, 1, 3, 3), dtype=np.float32)
    source_counts = {}
    for i, row in enumerate(rows):
        station = row["station"]
        if station not in STATION_INDEX:
            raise RuntimeError(f"unknown station {station}")
        st_idx = STATION_INDEX[station]
        target = datetime.fromisoformat(row["target_time_utc"])
        times = (row["earlier_time_utc"], row["previous_time_utc"], row["target_time_utc"])
        got_offsets = tuple(int((target - datetime.fromisoformat(t)).total_seconds() // 60) for t in times)
        if got_offsets != (30, 15, 0):
            raise RuntimeError(f"wrong history offsets at row {i}: {got_offsets}")
        if row["agri_frame_relpath"] != f"{target:%Y/%Y%m/%Y%m%d}/{target:%Y%m%d%H%M}.npy":
            raise RuntimeError(f"AGRI row path/time mismatch at row {i}")
        for ti, stamp in enumerate(times):
            pi = pindex.get(stamp)
            if pi is None:
                raise RuntimeError(f"missing p_cloud frame key {(station, stamp)}")
            pk = pkeys[pi]
            assignment = assignment_by_index[pi]
            if pk["split"] != row["split"] or assignment["split"] != row["split"]:
                raise RuntimeError(f"split mismatch at {(station, stamp)}")
            expected_rc = f"{row['row']},{row['col']}"
            if pk[f"{station}_grid_rc"] != expected_rc:
                raise RuntimeError(f"station pixel mismatch at {(station, stamp)}")
            rk = rindex.get((station, stamp))
            if rk is None or rk["split"] != row["split"] or (rk["row"], rk["col"]) != (row["row"], row["col"]):
                raise RuntimeError(f"R-bank key mismatch at {(station, stamp)}")
            if row["split"] == "val" and assignment["classifier_source"] != "full_train":
                raise RuntimeError(f"validation p_cloud was not produced by full_train at {(station, stamp)}")
            source = assignment["classifier_source"]
            source_counts[source] = source_counts.get(source, 0) + 1
            pc[i, ti, 0] = np.asarray(pcloud[pi, st_idx, 7:10, 7:10], dtype=np.float32)

    means = np.asarray(gm["AGRI_norm_mean"], np.float32)[None, None, :, None, None]
    stds = np.asarray(gm["AGRI_norm_std"], np.float32)[None, None, :, None, None]
    if np.any(stds <= 0):
        raise RuntimeError("invalid AGRI normalization")
    agri = np.where(ag_valid, (np.asarray(ag_raw, np.float32) - means) / stds, 0.0).astype(np.float32)
    geometry = np.asarray(geo, np.float32)
    cot = np.asarray(cot, np.float32)
    pcenter = pc - np.float32(0.5)

    # Freeze the already approved common train-only COT scaler from the paired
    # three-arm setup; do not refit on validation or per arm.
    if abs(EXPECTED_COT_MEAN - 1.8354828989218208) > 1e-12 or abs(EXPECTED_COT_STD - 1.3974485910139944) > 1e-12:
        raise AssertionError("unexpected frozen COT scaler")
    cot = (cot - np.float32(EXPECTED_COT_MEAN)) / np.float32(EXPECTED_COT_STD)
    x = np.concatenate((agri, geometry, cot, pcenter), axis=2)
    if x.shape != (n, 3, 17, 3, 3) or not np.isfinite(x).all():
        raise RuntimeError(f"four-arm tensor invalid: {x.shape}")

    y = np.asarray([float(r["ghi"]) for r in rows], np.float32)
    clear = np.asarray([float(r["clear"]) for r in rows], np.float32)
    if not np.isfinite(y).all() or not np.isfinite(clear).all() or np.any(clear < 0):
        raise RuntimeError("invalid GHI labels or interval clear-sky values")
    split = np.asarray([r["split"] for r in rows])
    stations = np.asarray([r["station"] for r in rows])
    train = split == "train"
    val = split == "val"
    exclude = np.asarray([(r["station"], bjt_date(r["target_time_utc"])) in EXCLUDED_STATION_DAYS for r in rows])
    if np.any(train & exclude):
        raise RuntimeError("station-day exclusion unexpectedly removed training rows")
    val &= ~exclude
    val_counts = {s: int(np.count_nonzero(val & (stations == s))) for s in STATIONS}
    excluded_counts = {s: int(np.count_nonzero(exclude & (stations == s))) for s in STATIONS}
    if val_counts != {"sili": 1999, "zhujia": 2227} or int(val.sum()) != 4226 or excluded_counts != {"sili": 37, "zhujia": 0}:
        raise RuntimeError(f"station-day-clean validation counts mismatch: val={val_counts}, excluded={excluded_counts}")
    if not np.any(train) or not np.any(val):
        raise RuntimeError("empty train or validation split")

    source_meta = {
        "state": "PASS_FOUR_ARM_ALIGNED_INPUTS",
        "train_val_only": True,
        "test_payloads_read": 0,
        "split_id": SPLIT_ID,
        "pack_rows": n,
        "train_rows": int(train.sum()),
        "raw_validation_rows": int(np.count_nonzero(split == "val")),
        "clean_validation_rows": int(val.sum()),
        "excluded_validation_rows": int(exclude.sum()),
        "excluded_station_days": [{"station": s, "bjt_date": d} for s, d in sorted(EXCLUDED_STATION_DAYS)],
        "exclusion_policy_sha256": sha256(exclusion_policy),
        "history_offsets_minutes": list(HISTORY_OFFSETS),
        "p_cloud_center_crop": "station-specific p_cloud[...,7:10,7:10], exact (station,UTC frame time) keys",
        "p_cloud_centering": "p_cloud - 0.5; continuous classifier score, not cloud fraction or calibrated radiative transmission",
        "classifier_geometry": "frozen legacy classifier scores produced using archived old aux cosSOZ/cosRAA/day_mask; no geometry mutation",
        "GHI_R_geometry": "corrected NOAA nominal-UTC geometry in fixed data_attempt2 pack",
        "COT_source": "frozen corrected-geometry R cot_corrected_log1p.npy; already log1p(COT), normalized only",
        "COT_scaler": {"mean": EXPECTED_COT_MEAN, "std": EXPECTED_COT_STD, "source": "paired_three_arm_v1 shared train-only old+corrected-R scaler"},
        "classifier_assignment_counts_over_three_frames": source_counts,
        "GHI_rows_sha256": sha256(pack / "rows.csv"),
        "GHI_manifest_sha256": sha256(pack / "pack_manifest.json"),
        "R_manifest_sha256": sha256(rbank / "bank_manifest.json"),
        "p_manifest_sha256": sha256(pbank / "bank_manifest.json"),
        "p_cloud_sha256": pm["array_sha256"]["p_cloud.npy"],
        "R_corrected_COT_sha256": rm["cot_corrected_sha256"],
        "p_keys_sha256": sha256(pbank / "frame_keys.csv"),
        "classifier_assignment_sha256": sha256(pbank / "classifier_assignment.csv"),
        "AGRI_normalization": "fixed train-fitted normalization from GHI pack manifest",
        "geometry_channels": 2,
        "input_layout": "[N,T=3,C=17,H=3,W=3], slots=AGRI13+geometry2+COT1+p_cloud1",
        "exact_key_join": "PASS; history time, station grid pixel and split checked for every frame",
    }
    write_json(output / "input_source_manifest.json", source_meta)
    with (output / "validation_rows.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for i, row in enumerate(rows):
            if val[i]:
                wr.writerow(row)
    indices = np.flatnonzero(val).astype(np.int64)
    np.savez_compressed(output / "validation_inputs.npz", pack_index=indices,
                        agri=np.asarray(agri[indices]), geometry=np.asarray(geometry[indices]),
                        cot_corrected_log1p_normalized=np.asarray(cot[indices]), p_cloud_centered=np.asarray(pcenter[indices]),
                        ghi=y[indices].astype(np.float64), clear=clear[indices].astype(np.float64))
    return x, y, clear, train, val, stations, rows, indices, source_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--r-bank", type=Path, required=True)
    ap.add_argument("--p-bank", type=Path, required=True)
    ap.add_argument("--exclusion-policy", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1792)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--stations", nargs="+", choices=STATIONS, default=list(STATIONS))
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch != 1792 or args.epochs != 30 or tuple(args.seeds) != (42, 43, 44) or tuple(args.stations) != STATIONS:
        raise RuntimeError("training contract differs from the frozen four-arm plan")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved training run")

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    x, y, clear, train, val, stations, rows, valid_ids, source_meta = make_inputs(
        args.pack, args.r_bank, args.p_bank, args.exclusion_policy, args.output)
    dev = torch.device("cuda:0")
    X = torch.from_numpy(x).to(dev)
    Y = torch.from_numpy(y).to(dev)
    CL = torch.from_numpy(clear).to(dev)
    del x, y, clear
    torch.cuda.synchronize()
    gpu_loaded = {"device": torch.cuda.get_device_name(0), "allocated_bytes": int(torch.cuda.memory_allocated(0)),
                  "reserved_bytes": int(torch.cuda.memory_reserved(0))}
    val_global = np.flatnonzero(val)

    # Same first-layer initialization for every arm, with additional channels
    # zeroed so all four functions initially agree exactly.
    param_model = FourArmIntervalHead().to(dev)
    param_count = sum(p.numel() for p in param_model.parameters())
    if param_count != 42529:
        raise RuntimeError(f"unexpected parameter count {param_count}")
    del param_model

    # Check exact initialization equality and both added channel gradients on
    # the first real training batch before launching all 24 fits.
    probe_ids = torch.as_tensor(np.flatnonzero(train)[:min(64, int(train.sum()))], device=dev)
    xprobe = X[probe_ids]
    target_probe = Y[probe_ids]
    clear_probe = CL[probe_ids]
    torch.manual_seed(20261020)
    probe = FourArmIntervalHead().to(dev)
    init = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    pred0 = probe(xprobe)
    differences = {}
    gradient_norms = {}
    cot_channel = X[:, :, 15].abs().sum().item()
    p_channel = X[:, :, 16].abs().sum().item()
    if cot_channel <= 0 or p_channel <= 0:
        raise RuntimeError(f"an added channel is constant zero: COT={cot_channel}, p={p_channel}")
    for arm in ARMS:
        tx = xprobe.clone()
        if arm not in ("M10", "M11"):
            tx[:, :, 15] = 0.0
        if arm not in ("M01", "M11"):
            tx[:, :, 16] = 0.0
        differences[arm] = float((probe(tx) - pred0).abs().max())
        if differences[arm] != 0.0:
            raise RuntimeError(f"initial function mismatch for {arm}: {differences[arm]}")
        probe.zero_grad(set_to_none=True)
        loss = (((probe(tx) * clear_probe - target_probe) / 1000.0) ** 2).mean()
        loss.backward()
        gradient_norms[arm] = {
            "cot_slot": float(probe.temporal.weight.grad[:, 15].norm()),
            "p_cloud_slot": float(probe.temporal.weight.grad[:, 16].norm()),
        }
        if arm in ("M10", "M11") and gradient_norms[arm]["cot_slot"] <= 0:
            raise RuntimeError(f"COT channel is disconnected for {arm}")
        if arm in ("M01", "M11") and gradient_norms[arm]["p_cloud_slot"] <= 0:
            raise RuntimeError(f"p_cloud channel is disconnected for {arm}")
    del probe, xprobe, target_probe, clear_probe
    torch.cuda.reset_peak_memory_stats(0)

    setup = {
        "state": "PASS_FOUR_ARM_TRAIN_SETUP",
        "arms": list(ARMS), "stations": list(STATIONS), "seeds": list(args.seeds),
        "pack_rows": source_meta["pack_rows"], "train": int(train.sum()),
        "raw_validation": source_meta["raw_validation_rows"],
        "clean_validation": int(val.sum()), "excluded_validation": source_meta["excluded_validation_rows"],
        "validation_pack_indices": valid_ids.tolist(),
        "input_layout": "[B,T=3,C=17,H=3,W=3]",
        "channel_order": ["AGRI_C01", "AGRI_C02", "AGRI_C03", "AGRI_C04", "AGRI_C05", "AGRI_C06", "AGRI_C09", "AGRI_C10", "AGRI_C11", "AGRI_C12", "AGRI_C13", "AGRI_C14", "AGRI_C15", "cosSOZ", "cosRAA", "COT_corrected_log1p_normalized", "p_cloud_minus_0.5"],
        "parameters_each_arm": param_count,
        "batch": args.batch, "epochs": args.epochs, "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "schedule": "CosineAnnealingLR(T_max=30, eta_min=1e-5)", "gradient_clip_norm": 5.0,
        "loss": "mean(((pred_k*interval_clear_GHI-GHI)/1000)^2)",
        "selection": "minimum station-day-clean validation GHI RMSE; first epoch wins exact ties",
        "zero_channel_initialization": "17-slot common head; first-layer COT and p_cloud input weights zeroed",
        "initial_prediction_max_abs_diff": differences,
        "added_channel_gradient_norms": gradient_norms,
        "gpu_loaded": gpu_loaded,
        "geometry_provenance_note": source_meta["classifier_geometry"],
        "test_payloads_read": 0,
        "input_source_manifest_sha256": sha256(args.output / "input_source_manifest.json"),
    }
    write_json(args.output / "setup.json", setup)
    print("FOUR_ARM_SETUP_PASS " + json.dumps({k: v for k, v in setup.items() if k != "validation_pack_indices"}, ensure_ascii=False), flush=True)

    # One real GPU optimizer step is an in-job smoke test. It validates tensor
    # shape, loss, backward pass and optimizer before the 24 planned fits, while
    # keeping the full data resident on the single allocated device.
    smoke_ids = torch.as_tensor(np.flatnonzero(train)[:min(args.batch, int(train.sum()))], device=dev)
    smoke_model = FourArmIntervalHead().to(dev)
    smoke_model.load_state_dict(init, strict=True)
    smoke_optimizer = torch.optim.AdamW(smoke_model.parameters(), lr=1e-3, weight_decay=1e-4)
    smoke_x = X[smoke_ids].clone()
    smoke_loss = (((smoke_model(smoke_x) * CL[smoke_ids] - Y[smoke_ids]) / 1000.0) ** 2).mean()
    if not torch.isfinite(smoke_loss):
        raise RuntimeError("single-batch GPU smoke loss is nonfinite")
    smoke_optimizer.zero_grad(set_to_none=True)
    smoke_loss.backward()
    torch.nn.utils.clip_grad_norm_(smoke_model.parameters(), 5.0)
    smoke_optimizer.step()
    torch.cuda.synchronize()
    if not all(torch.isfinite(p).all().item() for p in smoke_model.parameters()):
        raise RuntimeError("single-batch GPU smoke produced nonfinite parameters")
    write_json(args.output / "gpu_smoke_receipt.json", {
        "state": "PASS_SINGLE_BATCH_GPU_SMOKE", "device": torch.cuda.get_device_name(0),
        "batch": int(len(smoke_ids)), "loss_scaled": float(smoke_loss.detach()),
        "updates": 1, "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(0)),
        "test_payloads_read": 0,
    })
    print("GPU_SMOKE_PASS " + json.dumps({"device": torch.cuda.get_device_name(0), "batch": int(len(smoke_ids)), "loss_scaled": float(smoke_loss.detach())}), flush=True)
    del smoke_model, smoke_optimizer, smoke_x, smoke_ids
    torch.cuda.empty_cache()

    total_started = time.time()
    run_records = []
    arm_n = 0
    for seed in args.seeds:
        # One fixed shuffle schedule per seed/station, reused by all four arms.
        for station in STATIONS:
            train_ids = np.flatnonzero(train & (stations == station))
            valid_ids_station = np.flatnonzero(val & (stations == station))
            if len(train_ids) == 0 or len(valid_ids_station) == 0:
                raise RuntimeError(f"empty station split for {station}")
            shuffle_rng = np.random.default_rng(seed * 100 + STATION_INDEX[station])
            orders = [shuffle_rng.permutation(train_ids) for _ in range(args.epochs)]
            torch.manual_seed(seed * 100 + STATION_INDEX[station])
            base = FourArmIntervalHead().to(dev)
            init_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
            del base
            vi = torch.as_tensor(valid_ids_station, device=dev)

            for arm in ARMS:
                arm_n += 1
                started = time.time()
                folder = args.output / f"seed{seed}" / station / arm
                folder.mkdir(parents=True, exist_ok=False)
                model = FourArmIntervalHead().to(dev)
                model.load_state_dict(init_state, strict=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
                best_rmse = math.inf
                best_epoch = None
                best_state = None
                last_state = None
                total_updates = 0
                epoch_records = []
                torch.cuda.reset_peak_memory_stats(0)

                for epoch, order in enumerate(orders):
                    model.train()
                    loss_total = 0.0
                    for offset in range(0, len(order), args.batch):
                        batch_ids = torch.as_tensor(order[offset:offset + args.batch], device=dev)
                        xb = X[batch_ids].clone()
                        if arm not in ("M10", "M11"):
                            xb[:, :, 15] = 0.0
                        if arm not in ("M01", "M11"):
                            xb[:, :, 16] = 0.0
                        pred_k = model(xb)
                        loss = (((pred_k * CL[batch_ids] - Y[batch_ids]) / 1000.0) ** 2).mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        total_updates += 1
                        loss_total += float(loss.detach()) * len(batch_ids)
                    scheduler.step()
                    model.eval()
                    vpred = []
                    with torch.inference_mode():
                        for off in range(0, len(valid_ids_station), args.batch):
                            ids = vi[off:off + args.batch]
                            xb = X[ids].clone()
                            if arm not in ("M10", "M11"):
                                xb[:, :, 15] = 0.0
                            if arm not in ("M01", "M11"):
                                xb[:, :, 16] = 0.0
                            vpred.append((model(xb) * CL[ids]).cpu().numpy())
                    vpred = np.concatenate(vpred).astype(np.float64)
                    rmse = float(np.sqrt(np.mean((vpred - Y[vi].cpu().numpy().astype(np.float64)) ** 2)))
                    rec = {"epoch": epoch + 1, "val_rmse_Wm2": rmse,
                           "train_mse_scaled": loss_total / len(train_ids),
                           "learning_rate": float(optimizer.param_groups[0]["lr"]),
                           "cumulative_updates": total_updates}
                    epoch_records.append(rec)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_epoch = epoch + 1
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                        print("EPOCH " + json.dumps({"run": arm_n, "seed": seed, "station": station, "arm": arm, **rec}), flush=True)

                model.load_state_dict(best_state, strict=True)
                model.eval()
                # Apply the fixed zero slots explicitly for arms that omit a feature.
                xval = X[vi].clone()
                if arm not in ("M10", "M11"):
                    xval[:, :, 15] = 0.0
                if arm not in ("M01", "M11"):
                    xval[:, :, 16] = 0.0
                with torch.inference_mode():
                    pred_gpu = (model(xval) * CL[vi]).cpu().numpy().astype(np.float64)
                atomic_ckpt = {
                    "model": best_state, "seed": seed, "station": station, "arm": arm,
                    "best_epoch": best_epoch, "parameters": param_count,
                    "train_n": len(train_ids), "validation_n": len(valid_ids_station),
                    "input_source_manifest_sha256": setup["input_source_manifest_sha256"],
                    "test_payloads_read": 0,
                }
                torch.save(atomic_ckpt, folder / "best.pt")
                torch.save({**atomic_ckpt, "model": last_state, "saved_epoch": args.epochs}, folder / "last.pt")
                del model

                cpu_model = FourArmIntervalHead()
                cpu_model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["model"])
                cpu_model.eval()
                xval_cpu = X[vi].cpu().clone()
                if arm not in ("M10", "M11"):
                    xval_cpu[:, :, 15] = 0.0
                if arm not in ("M01", "M11"):
                    xval_cpu[:, :, 16] = 0.0
                with torch.inference_mode():
                    pred_cpu = (cpu_model(xval_cpu) * CL[vi].cpu()).numpy().astype(np.float64)
                reload_delta = float(np.max(np.abs(pred_cpu - pred_gpu)))
                if reload_delta > 0.1:
                    raise RuntimeError(f"CPU checkpoint reload differs by {reload_delta} W/m2")
                truth = Y[vi].cpu().numpy().astype(np.float64)
                pack_indices = valid_ids_station.astype(np.int64)
                np.savez_compressed(folder / "validation_predictions.npz", prediction=pred_gpu,
                                    cpu_prediction=pred_cpu, truth=truth, pack_index=pack_indices)
                pred_diff = pred_gpu - truth
                metrics = {
                    "state": "COMPLETE_ARM", "seed": seed, "station": station, "arm": arm,
                    "n_validation": len(valid_ids_station), "n_train": len(train_ids),
                    "rmse_Wm2": float(np.sqrt(np.mean(pred_diff ** 2))),
                    "mae_Wm2": float(np.mean(np.abs(pred_diff))),
                    "mbe_Wm2": float(np.mean(pred_diff)), "best_epoch": best_epoch,
                    "updates": total_updates, "expected_updates": int(math.ceil(len(train_ids) / args.batch) * args.epochs),
                    "seconds": time.time() - started,
                    "cpu_reload_max_abs_GHI_Wm2": reload_delta,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "best_sha256": sha256(folder / "best.pt"), "last_sha256": sha256(folder / "last.pt"),
                    "prediction_sha256": sha256(folder / "validation_predictions.npz"),
                    "test_payloads_read": 0,
                }
                write_json(folder / "metrics.json", metrics)
                (folder / "training.json").write_text(json.dumps(epoch_records, indent=2) + "\n", encoding="utf-8")
                run_records.append(metrics)
                write_json(args.output / "progress.json", {
                    "state": "RUNNING", "completed_arms": len(run_records), "planned_arms": 24,
                    "latest": metrics, "runs": run_records, "test_payloads_read": 0,
                })
                print("ARM_COMPLETE " + json.dumps(metrics), flush=True)
                del cpu_model, xval_cpu, pred_cpu, pred_gpu, xval
                torch.cuda.empty_cache()

    if len(run_records) != 24:
        raise RuntimeError(f"expected 24 completed arms, got {len(run_records)}")
    completion = {
        "state": "COMPLETE_24_PAIRED_RUNS", "planned_arms": 24, "completed_arms": len(run_records),
        "elapsed_seconds": time.time() - total_started, "runs": run_records,
        "test_payloads_read": 0,
    }
    write_json(args.output / "completion.json", completion)
    write_json(args.output / "progress.json", completion)
    print("FOUR_ARM_PAIRED_GHI_COMPLETE " + json.dumps({"elapsed_seconds": completion["elapsed_seconds"], "completed_arms": len(run_records)}), flush=True)


if __name__ == "__main__":
    main()
