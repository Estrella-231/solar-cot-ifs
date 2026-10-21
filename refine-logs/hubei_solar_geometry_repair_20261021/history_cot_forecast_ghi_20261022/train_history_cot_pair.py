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


def prep_data(pack: Path, bank: Path, station: str):
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


def train_steps(model, optimizer, future, history, cot, ghi, clear, order, batch_size, arm):
    model.train()
    total = 0.0
    seen = 0
    updates = 0
    for start in range(0, len(order), batch_size):
        ids = torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=future.device)
        aux = cot[ids] if arm == "FAC_HIST" else torch.zeros_like(cot[ids])
        pred_kt = model(future[ids], history[ids], aux)
        err = (pred_kt * clear[ids] - ghi[ids]) / 1000.0
        loss = err.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm} training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(ids)
        seen += len(ids)
        updates += 1
    return total / max(seen, 1), updates


@torch.no_grad()
def predict(model, future, history, cot, clear, ids, batch_size, arm):
    model.eval()
    out = []
    for start in range(0, len(ids), batch_size):
        ix0 = ids[start:start + batch_size]
        ix = torch.as_tensor(ix0, dtype=torch.long, device=future.device)
        aux = cot[ix] if arm == "FAC_HIST" else torch.zeros_like(cot[ix])
        out.append((model(future[ix], history[ix], aux) * clear[ix]).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--history-bank", type=Path, required=True)
    p.add_argument("--station", choices=STATIONS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1536)
    p.add_argument("--output", type=Path)
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--profile-updates", type=int, default=64)
    p.add_argument("--profile-output", type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Paired historical COT GHI worker needs one allocated CUDA device")
    if a.batch_size < 1 or a.epochs < 1:
        raise ValueError("Invalid training budget")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    data = prep_data(a.pack, a.history_bank, a.station)
    future = data["future"].to(device, non_blocking=True)
    history = data["history"].to(device, non_blocking=True)
    cot = data["cot"].to(device, non_blocking=True)
    ghi = data["ghi"].to(device, non_blocking=True)
    clear = data["clear"].to(device, non_blocking=True)
    train_ids = data["train_ids"]
    val_ids = data["val_ids"]
    val_ids_t = torch.as_tensor(val_ids, dtype=torch.long, device=device)

    if a.profile_only:
        if a.profile_output is None or a.profile_updates < 1:
            raise ValueError("profile-only needs --profile-output and positive --profile-updates")
        torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        template = HistoryCOTGHIHead().to(device)
        rng = np.random.default_rng(a.seed)
        order = rng.permutation(train_ids)
        batch = min(a.batch_size, len(order))
        order = np.resize(order, (a.profile_updates + 12) * batch)
        output = []
        for arm in ARMS:
            model = HistoryCOTGHIHead().to(device); model.load_state_dict(template.state_dict(), strict=True)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for j in range(12):
                sl = order[j * batch:(j + 1) * batch]
                ids = torch.as_tensor(sl, dtype=torch.long, device=device)
                acot = cot[ids] if arm == "FAC_HIST" else torch.zeros_like(cot[ids])
                pred = model(future[ids], history[ids], acot)
                loss = (((pred * clear[ids] - ghi[ids]) / 1000.0) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); start = time.perf_counter()
            for j in range(12, 12 + a.profile_updates):
                sl = order[j * batch:(j + 1) * batch]
                ids = torch.as_tensor(sl, dtype=torch.long, device=device)
                acot = cot[ids] if arm == "FAC_HIST" else torch.zeros_like(cot[ids])
                pred = model(future[ids], history[ids], acot)
                loss = (((pred * clear[ids] - ghi[ids]) / 1000.0) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            output.append({"arm": arm, "updates": a.profile_updates,
                           "examples_per_second": float(batch * a.profile_updates / elapsed),
                           "seconds_per_update": float(elapsed / a.profile_updates)})
            del model, opt
        receipt = {"state": "HISTORY_COT_GHI_HEAD_PROFILE_COMPLETE", "station": a.station,
                   "seed": a.seed, "batch_size": batch, "profile_updates_per_arm": a.profile_updates,
                   "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                   "rows": data["input_keys"]["rows"], "results": output,
                   "trainer_sha256": sha256(Path(__file__)), "test_payloads_read": 0, "buffer_payloads_read": 0}
        a.profile_output.parent.mkdir(parents=True, exist_ok=True)
        a.profile_output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return

    if a.output is None:
        raise ValueError("Training requires --output")
    a.output.mkdir(parents=True, exist_ok=False)
    seed = int(a.seed)
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(train_ids) for _ in range(a.epochs)]
    order_sha = hashlib.sha256(np.concatenate(orders).astype(np.int64).tobytes()).hexdigest()
    template = HistoryCOTGHIHead().to(device)
    init = {k: v.detach().clone() for k, v in template.state_dict().items()}
    init_sha = array_sha256(init)
    metadata = {
        "state": "HISTORY_COT_PAIRED_TRAINING_STARTED", "station": a.station, "seed": seed,
        "arms": list(ARMS), "epochs": a.epochs, "batch_size": a.batch_size,
        "train_rows": int(len(train_ids)), "val_rows": int(len(val_ids)),
        "train_updates_per_arm": int(sum((len(o) + a.batch_size - 1) // a.batch_size for o in orders)),
        "model": "same future-AGRI branch + same 8-frame historical AGRI/geometry branch + same 8-frame COT branch + same readout",
        "paired_difference": "only FAC_HIST receives train-standardized frozen-R historical COT; FA_HIST receives zeros in that same branch",
        "initialization_sha256": init_sha, "row_order_sha256": order_sha,
        "normalization": data["norm"], "lineage": data["input_keys"],
        "loss": "MSE((pred_kt*15min_interval_clear_sky_GHI - observed_GHI)/1000)",
        "selection": "lowest saved validation RMSE among the fixed 30-epoch schedule, same rule for both arms",
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "trainer_sha256": sha256(Path(__file__)), "test_payloads_read": 0, "buffer_payloads_read": 0,
    }
    (a.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_metrics = []
    for arm in ARMS:
        model = HistoryCOTGHIHead().to(device)
        model.load_state_dict(init, strict=True)
        if array_sha256({k: v.detach().clone() for k, v in model.state_dict().items()}) != init_sha:
            raise RuntimeError(f"Paired {arm} initialization mismatch")
        init_arm_sha = init_sha
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-5)
        best_rmse, best_epoch, best_state = float("inf"), -1, None
        arm_logs = []
        actual_updates = 0
        t0 = time.time()
        for epoch, order in enumerate(orders, start=1):
            train_mse, nupdate = train_steps(model, optimizer, future, history, cot, ghi, clear,
                                             order, a.batch_size, arm)
            actual_updates += nupdate
            scheduler.step()
            val_pred = predict(model, future, history, cot, clear, val_ids, a.batch_size, arm)
            truth = ghi[val_ids_t].detach().cpu().numpy().astype(np.float64)
            met = metric(val_pred, truth)
            row = {"station": a.station, "seed": seed, "arm": arm, "epoch": epoch,
                   "train_scaled_mse": train_mse, **{f"val_{k}": v for k, v in met.items()},
                   "learning_rate_next": optimizer.param_groups[0]["lr"], "updates_so_far": actual_updates}
            arm_logs.append(row); all_metrics.append(row)
            if met["rmse_wm2"] < best_rmse:
                best_rmse, best_epoch = met["rmse_wm2"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch == a.epochs or epoch % 5 == 0:
                print(json.dumps({"event": "EPOCH", **row}), flush=True)
        if actual_updates != int(metadata["train_updates_per_arm"]):
            raise RuntimeError(f"Update count mismatch {actual_updates} != {metadata['train_updates_per_arm']}")
        last_path = a.output / f"{arm}_last.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": a.epochs,
                    "actual_updates": actual_updates, "initialization_sha256": init_arm_sha,
                    "row_order_sha256": order_sha}, last_path)
        if best_state is None:
            raise RuntimeError("No validation-selected checkpoint")
        model.load_state_dict(best_state, strict=True)
        pred_before = predict(model, future, history, cot, clear, val_ids, a.batch_size, arm)
        best_path = a.output / f"{arm}_best.pt"
        torch.save({"model": best_state, "station": a.station, "arm": arm, "seed": seed,
                    "best_epoch": best_epoch, "best_val_rmse_wm2": best_rmse,
                    "initialization_sha256": init_sha, "row_order_sha256": order_sha,
                    "normalization": data["norm"]}, best_path)
        chk = torch.load(best_path, map_location=device, weights_only=False)
        reload_model = HistoryCOTGHIHead().to(device).eval()
        reload_model.load_state_dict(chk["model"], strict=True)
        pred_after = predict(reload_model, future, history, cot, clear, val_ids, a.batch_size, arm)
        reload_delta = float(np.max(np.abs(pred_before - pred_after)))
        if not np.allclose(pred_before, pred_after, rtol=1e-6, atol=1e-4):
            raise RuntimeError(f"Checkpoint reload mismatch for {arm}: {reload_delta}")
        val_rows = [data["rows"][i] for i in val_ids]
        pred_path = a.output / f"{arm}_validation_predictions.npz"
        np.savez_compressed(pred_path,
                            station=np.asarray([r["station"] for r in val_rows]),
                            split=np.asarray([r["split"] for r in val_rows]),
                            init_time_bjt=np.asarray([r["init_time_bjt"] for r in val_rows]),
                            target_time_bjt=np.asarray([r["target_time_bjt"] for r in val_rows]),
                            lead_minutes=np.asarray([int(r["lead_minutes"]) for r in val_rows], dtype=np.int16),
                            source_role=np.asarray([r["source_role"] for r in val_rows]),
                            pred_ghi_wm2=pred_after, truth_ghi_wm2=truth,
                            clear_sky_wm2=clear[val_ids_t].detach().cpu().numpy().astype(np.float64))
        result = {"state": "HISTORY_COT_PAIRED_ARM_COMPLETE", "station": a.station, "seed": seed,
                  "arm": arm, "best_epoch": best_epoch, "actual_updates": actual_updates,
                  "initialization_sha256": init_sha, "row_order_sha256": order_sha,
                  "best_validation_fp64": metric(pred_after, truth),
                  "checkpoint_sha256": sha256(best_path), "last_checkpoint_sha256": sha256(last_path),
                  "prediction_sha256": sha256(pred_path), "checkpoint_reload_max_abs_delta_wm2": reload_delta,
                  "checkpoint_reload_parity": "PASS", "test_payloads_read": 0, "buffer_payloads_read": 0,
                  "elapsed_seconds": float(time.time() - t0)}
        (a.output / f"{arm}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (a.output / f"{arm}_epochs.jsonl").open("w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in arm_logs) + "\n")
        print(json.dumps({"event": "ARM_COMPLETE", **result}, ensure_ascii=False), flush=True)
        del model, reload_model, optimizer, scheduler, best_state
        torch.cuda.empty_cache()

    if all_metrics:
        with (a.output / "all_epochs.jsonl").open("w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in all_metrics) + "\n")
    results = [json.loads((a.output / f"{arm}_result.json").read_text(encoding="utf-8")) for arm in ARMS]
    if results[0]["initialization_sha256"] != results[1]["initialization_sha256"] or results[0]["row_order_sha256"] != results[1]["row_order_sha256"] or results[0]["actual_updates"] != results[1]["actual_updates"]:
        raise RuntimeError("Paired arms violate initialization/order/update equality")
    metadata.update({"state": "HISTORY_COT_PAIRED_TRAINING_COMPLETE",
                     "pair_initialization_equal": True, "pair_row_order_equal": True,
                     "pair_updates_equal": True, "actual_updates_per_arm": results[0]["actual_updates"],
                     "arm_results": results,
                     "elapsed_seconds": float(sum(r["elapsed_seconds"] for r in results))})
    (a.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "PAIR_COMPLETE", "station": a.station, "seed": seed,
                      "fa_rmse": results[0]["best_validation_fp64"]["rmse_wm2"],
                      "fac_rmse": results[1]["best_validation_fp64"]["rmse_wm2"],
                      "output": str(a.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
