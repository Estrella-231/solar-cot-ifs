#!/usr/bin/env python3
"""Paired GHI forecast head with identical history AGRI and optional history COT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
OFFSETS = (105, 90, 75, 60, 45, 30, 15, 0)
ARMS = ("FA_HIST", "FAC_HIST")


class HistoryCOTGHIHead(nn.Module):
    """Both arms share identical future, history-AGRI, COT branches and readout."""
    def __init__(self, width: int = 32, aux_width: int = 4):
        super().__init__()
        self.future_encoder = nn.Sequential(nn.Conv2d(15, width, 1), nn.SiLU(),
                                            nn.Conv2d(width, width, 1), nn.SiLU())
        self.history_encoder = nn.Sequential(nn.Conv2d(8 * 15, width, 1), nn.SiLU(),
                                             nn.Conv2d(width, width, 1), nn.SiLU())
        self.cot_encoder = nn.Sequential(nn.Conv2d(8, aux_width, 1, bias=False), nn.SiLU())
        fused = 2 * width + aux_width
        self.readout = nn.Sequential(nn.Linear(fused * 3 * 3, 64), nn.SiLU(),
                                     nn.Linear(64, 32), nn.SiLU(),
                                     nn.Linear(32, 1), nn.Softplus())

    def forward(self, future: torch.Tensor, history: torch.Tensor, cot_history: torch.Tensor) -> torch.Tensor:
        if tuple(future.shape[1:]) != (15, 3, 3):
            raise ValueError("future expects 13 AGRI + 2 solar geometry on 3x3")
        if tuple(history.shape[1:]) != (8, 15, 3, 3):
            raise ValueError("history expects [8,13 AGRI + 2 solar geometry,3,3]")
        if tuple(cot_history.shape[1:]) != (8, 1, 3, 3):
            raise ValueError("historical COT expects [8,1,3,3]")
        b = future.shape[0]
        f = self.future_encoder(future)
        h = self.history_encoder(history.reshape(b, 8 * 15, 3, 3))
        c = self.cot_encoder(cot_history.reshape(b, 8, 3, 3))
        return self.readout(torch.cat((f, h, c), dim=1).flatten(1)).squeeze(-1)


def sha256(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(arrays: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(arrays[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metric(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {"n": int(e.size), "rmse_wm2": float(np.sqrt(np.mean(e * e, dtype=np.float64))),
            "mae_wm2": float(np.mean(np.abs(e), dtype=np.float64)),
            "bias_wm2": float(np.mean(e, dtype=np.float64))}


def prep_data(pack: Path, bank: Path, station: str, fit_rows=None):
    if station not in STATIONS:
        raise ValueError(station)
    report = json.loads((pack / "pack_report.json").read_text(encoding="utf-8"))
    if report.get("state") != "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("Input is not the audited corrected forecast GHI pack")
    if report.get("test_and_buffer_values_read") != 0 or report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError("The GHI pack is not test/buffer closed")
    if report.get("train_forecast_inputs_are_oof") is not True or report.get("oof_fold_count") != 5:
        raise ValueError("Train predictions must be the audited 5-fold OOF bank")
    if report.get("validation_forecast_source") != "frozen_full_train_S_R" or report.get("correctutc_cot_rebuilt") is not True:
        raise ValueError("Validation forecast source/time binding is not the corrected frozen full S->R chain")
    bank_report = json.loads((bank / "bank_report.json").read_text(encoding="utf-8"))
    if bank_report.get("state") != "HISTORICAL_COT_BANK_TRAIN_VAL_COMPLETE" or bank_report.get("test_payloads_read") != 0:
        raise ValueError("Historical COT bank is not complete/train-val-only")
    bank_audit = json.loads((bank / "bank_audit.json").read_text(encoding="utf-8"))
    if bank_audit.get("state") != "PASS_HISTORY_COT_BANK_INDEPENDENT_AUDIT" or bank_audit.get("test_payloads_read") != 0:
        raise ValueError("Independent history bank audit has not passed")
    rows = list(csv.DictReader((pack / "rows.csv").open("r", newline="", encoding="utf-8-sig")))
    nall = len(rows)
    all_indices = np.flatnonzero(np.asarray([r["station"] == station for r in rows]))
    station_rows = [rows[i] for i in all_indices]
    split = np.asarray([r["split"] for r in station_rows])
    if set(split.tolist()) != {"train", "val"}:
        raise ValueError(f"Missing train/val data for {station}")

    agri_mm = np.load(pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geom_mm = np.load(pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    ghi_mm = np.load(pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear_mm = np.load(pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    if agri_mm.shape != (nall, 13, 3, 3) or geom_mm.shape != (nall, 2, 3, 3):
        raise ValueError("Forecast AGRI/geometry pack shapes mismatch")
    future = np.concatenate((np.asarray(agri_mm[all_indices], np.float32),
                             np.asarray(geom_mm[all_indices], np.float32)), axis=1)
    ghi = np.asarray(ghi_mm[all_indices], np.float32)
    clear = np.asarray(clear_mm[all_indices], np.float32)
    if not all(np.isfinite(x).all() for x in (future, ghi, clear)) or (clear <= 0).any() or (ghi < 0).any():
        raise ValueError("Invalid future/label pack values")

    key_rows = list(csv.DictReader((bank / "frame_keys.csv").open("r", newline="", encoding="utf-8-sig")))
    frame_idx = {r["bjt"]: int(r["frame_index"]) for r in key_rows}
    frame_split = {r["bjt"]: r["split"] for r in key_rows}
    feature_mm = np.load(bank / "history_agri_geometry_16x16.npy", mmap_mode="r", allow_pickle=False)
    cot_map_mm = np.load(bank / "history_cot_16x16.npy", mmap_mode="r", allow_pickle=False)
    cot_valid_mm = np.load(bank / "history_cot_valid_16x16.npy", mmap_mode="r", allow_pickle=False)
    cy, cx = CENTERS[station]
    ysl, xsl = slice(cy - 1, cy + 2), slice(cx - 1, cx + 2)
    # Read just the station neighborhood from each frame, not the repeated 16x16 bank.
    hist_frame = np.asarray(feature_mm[:, :15, ysl, xsl], dtype=np.float32)
    cot_frame = np.asarray(cot_map_mm[:, :, ysl, xsl], dtype=np.float32)
    cot_valid_frame = np.asarray(cot_valid_mm[:, :, ysl, xsl], dtype=bool)
    if (hist_frame.shape[1:] != (15, 3, 3) or cot_frame.shape[1:] != (1, 3, 3)
            or cot_valid_frame.shape != cot_frame.shape):
        raise ValueError("Station center is outside the saved history patch")

    hist_indices = np.empty((len(station_rows), 8), dtype=np.int32)
    for i, row in enumerate(station_rows):
        init = datetime.strptime(row["init_time_bjt"], "%Y%m%d%H%M")
        for j, off in enumerate(OFFSETS):
            k = (init - timedelta(minutes=off)).strftime("%Y%m%d%H%M")
            if k not in frame_idx or frame_split[k] != row["split"]:
                raise ValueError(f"Bad history mapping for {station}/{row['init_time_bjt']}/{k}")
            hist_indices[i, j] = frame_idx[k]

    train_ids = np.flatnonzero(split == "train")
    if fit_rows is not None:
        train_ids = np.flatnonzero(np.asarray(fit_rows, dtype=bool) & (split == "train"))
    val_ids = np.flatnonzero(split == "val")
    unique_train_frames = np.unique(hist_indices[train_ids])
    if not len(train_ids) or not len(val_ids) or not len(unique_train_frames):
        raise ValueError("Empty station support or train-only history fit set")

    # Future AGRI and historical AGRI/COT each receive train-only statistics,
    # fitted on their own physically distinct input distributions.
    fmu = future[train_ids, :13].astype(np.float64).mean(axis=(0, 2, 3))
    fsd = np.maximum(future[train_ids, :13].astype(np.float64).std(axis=(0, 2, 3)), 1e-6)
    htrain = hist_frame[unique_train_frames, :13].astype(np.float64)
    hmu = htrain.mean(axis=(0, 2, 3))
    hsd = np.maximum(htrain.std(axis=(0, 2, 3)), 1e-6)
    cot_log_train = np.log1p(cot_frame[unique_train_frames].astype(np.float64))
    cot_valid_train = cot_valid_frame[unique_train_frames]
    valid_cot_values = cot_log_train[cot_valid_train]
    if valid_cot_values.size == 0 or not np.isfinite(valid_cot_values).all():
        raise ValueError("No finite valid historical COT pixels in train frames")
    cmu = float(valid_cot_values.mean(dtype=np.float64))
    csd = max(float(valid_cot_values.std(dtype=np.float64)), 1e-6)

    future_norm = future.copy()
    future_norm[:, :13] = ((future_norm[:, :13] - fmu.astype(np.float32)[None, :, None, None]) /
                           fsd.astype(np.float32)[None, :, None, None])
    history = np.asarray(hist_frame[hist_indices], dtype=np.float32)
    history[:, :, :13] = ((history[:, :, :13] - hmu.astype(np.float32)[None, None, :, None, None]) /
                          hsd.astype(np.float32)[None, None, :, None, None])
    cot_seq = np.log1p(np.asarray(cot_frame[hist_indices], dtype=np.float32))
    cot_valid_seq = cot_valid_frame[hist_indices]
    cot_seq = ((cot_seq - np.float32(cmu)) / np.float32(csd)).astype(np.float32)
    # Missing/nighttime COT is neutral in both paired arms after normalization.
    cot_seq = np.where(cot_valid_seq, cot_seq, np.float32(0.0)).astype(np.float32, copy=False)

    data = {
        "future": torch.from_numpy(np.ascontiguousarray(future_norm)),
        "history": torch.from_numpy(np.ascontiguousarray(history)),
        "cot": torch.from_numpy(np.ascontiguousarray(cot_seq)),
        "ghi": torch.from_numpy(ghi), "clear": torch.from_numpy(clear),
        "split": split, "train_ids": train_ids, "val_ids": val_ids,
        "rows": station_rows,
        "norm": {"fit_split": "train only", "future_agri_mean": fmu.tolist(), "future_agri_std": fsd.tolist(),
                 "history_agri_mean": hmu.tolist(), "history_agri_std": hsd.tolist(),
                 "history_cot_log1p_mean": cmu, "history_cot_log1p_std": csd,
                 "history_cot_fit_support": "valid day_mask>0.5 and cosSOZ>cos80 pixels only; invalid inputs set to standardized zero in both arms",
                 "history_norm_frame_count": int(len(unique_train_frames)),
                 "station_center_rc_in_patch": list(CENTERS[station])},
        "input_keys": {"pack_report_sha256": sha256(pack / "pack_report.json"),
                       "bank_report_sha256": sha256(bank / "bank_report.json"),
                       "station": station, "rows": int(len(station_rows)),
                       "train_rows": int(len(train_ids)), "val_rows": int(len(val_ids))},
    }
    return data


