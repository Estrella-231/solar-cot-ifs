#!/usr/bin/env python3
"""GPU training/profile for independent Hubei same-time or forecast GHI heads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from hubei_forecast_pack_gate import validate_input_pack_contract

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS_BY_MODE = {
    "same_time": ("A", "AC", "AP"),
    "forecast": ("FA", "FAC", "FAP"),
}


class HubeiLocalGHIHead(nn.Module):
    """Pointwise 3x3 AGRI/geometry encoder with a fixed-position readout."""
    def __init__(self, variant: str, trunk_width: int = 32, auxiliary_width: int = 4):
        super().__init__()
        if variant not in ("A", "AC", "AP", "FA", "FAC", "FAP"):
            raise ValueError(variant)
        self.variant = variant
        base_variant = {"FA": "A", "FAC": "AC", "FAP": "AP"}.get(variant, variant)
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(15, trunk_width, 1), nn.SiLU(),
            nn.Conv2d(trunk_width, trunk_width, 1), nn.SiLU(),
        )
        if base_variant == "AC":
            self.aux_encoder = nn.Sequential(nn.Conv2d(1, auxiliary_width, 1), nn.SiLU())
        elif base_variant == "AP":
            self.aux_encoder = nn.Sequential(nn.Conv2d(trunk_width, auxiliary_width, 1), nn.SiLU())
        else:
            self.aux_encoder = None
        fused = trunk_width + (auxiliary_width if base_variant != "A" else 0)
        self.readout = nn.Sequential(
            nn.Linear(fused * 3 * 3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, agri: torch.Tensor, cot: torch.Tensor | None = None) -> torch.Tensor:
        if agri.ndim != 4 or tuple(agri.shape[1:]) != (15, 3, 3):
            raise ValueError("agri must be [B,15,3,3] = 13 AGRI + cosSOZ + cosRAA")
        features = self.agri_encoder(agri)
        if self.variant in ("AC", "FAC"):
            if cot is None or cot.shape != (agri.shape[0], 1, 3, 3):
                raise ValueError("AC requires registered log1p(COT) [B,1,3,3]")
            features = torch.cat((features, self.aux_encoder(cot)), dim=1)
        elif self.variant in ("AP", "FAP"):
            features = torch.cat((features, self.aux_encoder(features)), dim=1)
        return self.readout(features.flatten(1)).squeeze(-1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_sha(obj: object) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_counts(arms: tuple[str, ...]) -> dict[str, int]:
    return {arm: sum(p.numel() for p in HubeiLocalGHIHead(arm).parameters()) for arm in arms}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--input-mode", choices=("same_time", "forecast"), default="same_time")
    p.add_argument("--arms", nargs="+", choices=("A", "AC", "AP", "FA", "FAC", "FAP"))
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--profile-output", type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA; there is no CPU training fallback")
    meta = json.loads((a.pack / "pack_report.json").read_text(encoding="utf-8"))
    validate_input_pack_contract(meta, a.input_mode)
    default_arms = ARMS_BY_MODE[a.input_mode]
    arms = tuple(a.arms or default_arms)
    permitted_arms = set(default_arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(permitted_arms):
        raise ValueError(f"Arms {arms} do not match input mode {a.input_mode}")
    if meta.get("input_mode", a.input_mode) != a.input_mode:
        raise ValueError("Pack input mode does not match requested training mode")
    if a.output.exists() and not a.profile_only:
        raise FileExistsError(a.output)
    rows = list(csv.DictReader((a.pack / "rows.csv").open(encoding="utf-8-sig", newline="")))
    split = np.asarray([r["split"] for r in rows])
    station = np.asarray([r["station"] for r in rows])
    agri = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    ghi = np.load(a.pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(a.pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    n = len(rows)
    if agri.shape != (n, 13, 3, 3) or geometry.shape != (n, 2, 3, 3) or cot.shape != (n, 1, 3, 3):
        raise ValueError("Packed model input shapes do not match contract")
    if any(not np.isfinite(x).all() for x in (agri, geometry, cot, ghi, clear)):
        raise ValueError("Nonfinite train/validation payload")
    if (clear <= 0).any() or (ghi < 0).any():
        raise ValueError("Invalid daylight/clear-sky support")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)

    # Train-only, per-station normalization. Geometry stays in physical units;
    # COT is log1p transformed before its own train-only standardization.
    norms = {}
    for st in STATIONS:
        train_ids = np.flatnonzero((split == "train") & (station == st))
        if not len(train_ids):
            raise ValueError(f"No train rows for {st}")
        x = np.asarray(agri[train_ids], dtype=np.float64)
        c = np.log1p(np.asarray(cot[train_ids], dtype=np.float64))
        means = x.mean(axis=(0, 2, 3))
        stds = np.maximum(x.std(axis=(0, 2, 3)), 1e-6)
        cmean = float(c.mean())
        cstd = float(max(c.std(), 1e-6))
        norms[st] = {"agri_mean": means.astype(np.float32), "agri_std": stds.astype(np.float32),
                     "cot_log1p_mean": cmean, "cot_log1p_std": cstd,
                     "train_rows_for_normalization": int(len(train_ids))}

    # Persist the exact train-only transform, not only its descriptive name.
    # Arrays are serialized as float32 lists so an inference/reload path can
    # reconstruct the same normalization without refitting on validation data.
    norms_payload = {
        st: {"agri_mean": z["agri_mean"].astype(np.float32).tolist(),
             "agri_std": z["agri_std"].astype(np.float32).tolist(),
             "cot_log1p_mean": float(z["cot_log1p_mean"]),
             "cot_log1p_std": float(z["cot_log1p_std"]),
             "train_rows_for_normalization": int(z["train_rows_for_normalization"])}
        for st, z in norms.items()
    }
    norms_sha256 = canonical_json_sha(norms_payload)

    def normalized_arrays(st: str):
        ids = np.flatnonzero(station == st)
        x = np.asarray(agri[ids], dtype=np.float32)
        g = np.asarray(geometry[ids], dtype=np.float32)
        clog = np.log1p(np.asarray(cot[ids], dtype=np.float32))
        z = norms[st]
        x = (x - z["agri_mean"][None, :, None, None]) / z["agri_std"][None, :, None, None]
        clog = (clog - z["cot_log1p_mean"]) / z["cot_log1p_std"]
        x = np.concatenate((x, g), axis=1).astype(np.float32)
        return ids, x, clog.astype(np.float32)

    if a.profile_only:
        if a.profile_output is None:
            raise ValueError("--profile-output is required")
        # Use the smallest station support, and enforce at least 12 updates/epoch.
        st = min(STATIONS, key=lambda s: int(np.sum((split == "train") & (station == s))))
        ids, x, c = normalized_arrays(st)
        local_train = np.flatnonzero(split[ids] == "train")
        usable = len(local_train)
        candidates = [b for b in (128, 256, 512, 1024, 1536, 1792, 2000)
                      if b <= usable and usable // b >= 12]
        if not candidates:
            raise RuntimeError("No batch candidate provides at least 12 updates/epoch")
        X = torch.from_numpy(x).to(device)
        C = torch.from_numpy(c).to(device)
        GH = torch.from_numpy(np.asarray(ghi[ids], dtype=np.float32)).to(device)
        CL = torch.from_numpy(np.asarray(clear[ids], dtype=np.float32)).to(device)
        result = []
        for arm in arms:
            for batch in candidates:
                ids_b = torch.as_tensor(local_train[:batch], device=device)
                model = HubeiLocalGHIHead(arm).to(device).train()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                def step():
                    pred = model(X[ids_b], C[ids_b] if arm in ("AC", "FAC") else None)
                    loss = (((pred * CL[ids_b] - GH[ids_b]) / 1000.0) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                for _ in range(3): step()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                for _ in range(12): step()
                torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
                result.append({"station": st, "arm": arm, "batch": batch,
                               "updates_per_epoch_min_support": usable // batch,
                               "examples_per_second": batch * 12 / elapsed,
                               "peak_GiB": torch.cuda.max_memory_allocated() / 1024**3})
                del model, opt
                torch.cuda.empty_cache()
        a.profile_output.parent.mkdir(parents=True, exist_ok=True)
        profile = {"state": "HUBEI_GHI_HEAD_SINGLE_GPU_PROFILE_COMPLETE",
                   "profile_station": st, "profile_rows": usable,
                   "input_mode": a.input_mode, "arms": list(arms),
                   "parameter_counts": parameter_counts(arms),
                   "results": result, "test_payloads_read": 0,
                   "selection": "largest batch with >=12 updates/epoch and within 3% of best measured examples/s across all arms"}
        a.profile_output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)
        return

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    run_start = time.time()
    a.output.mkdir(parents=True)
    run = {"state": "HUBEI_GHI_HEADS_TRAINING_STARTED", "seed": a.seed,
           "epochs": a.epochs, "batch_size": a.batch_size, "pack_sha256": sha(a.pack / "pack_report.json"),
           "trainer_sha256": sha(Path(__file__)), "parameter_counts": parameter_counts(arms),
           "input_mode": a.input_mode, "stations": list(STATIONS), "arms": list(arms), "split_sha256": meta["split_sha256"],
           "normalization": "station-specific train-only AGRI and log1p(COT); geometry physical",
           "normalization_contract_sha256": norms_sha256,
           "loss": "MSE((pred_kt*interval_clear_sky_GHI - observed_GHI)/1000)",
           "target": "15-minute label in the original engineering W/m2 operating convention; sensor calibration/export metadata not independently verified",
           "test_payloads_read": 0, "gpu": torch.cuda.get_device_name(0)}
    (a.output / "normalization_contract.json").write_text(
        json.dumps({"schema": "hubei_ghi_head_norm_v1", "sha256": norms_sha256,
                    "fit_split": "train", "stations": norms_payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (a.output / "run_metadata.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(a.seed)
    all_logs = []
    for st in STATIONS:
        ids, x, c = normalized_arrays(st)
        local_split = split[ids]
        train_ids = np.flatnonzero(local_split == "train")
        val_ids = np.flatnonzero(local_split == "val")
        if not len(train_ids) or not len(val_ids):
            raise ValueError(f"Empty train/validation for {st}")
        if len(train_ids) // a.batch_size < 12:
            raise ValueError(f"batch_size={a.batch_size} yields <12 updates for {st}")
        X = torch.from_numpy(x).to(device)
        C = torch.from_numpy(c).to(device)
        GH = torch.from_numpy(np.asarray(ghi[ids], dtype=np.float32)).to(device)
        CL = torch.from_numpy(np.asarray(clear[ids], dtype=np.float32)).to(device)
        base = HubeiLocalGHIHead("A").to(device)
        common_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        orders = [rng.permutation(train_ids) for _ in range(a.epochs)]
        station_dir = a.output / st
        station_dir.mkdir()
        station_logs = []
        for arm in arms:
            torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
            model = HubeiLocalGHIHead(arm).to(device)
            if arm == "A":
                model.load_state_dict(common_state, strict=True)
            else:
                with torch.no_grad():
                    model.agri_encoder.load_state_dict(base.agri_encoder.state_dict())
                    model.readout[0].weight.zero_()
                    model.readout[0].weight[:, :base.readout[0].in_features].copy_(base.readout[0].weight)
                    model.readout[0].bias.copy_(base.readout[0].bias)
                    model.readout[2:].load_state_dict(base.readout[2:].state_dict(), strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-5)
            best_rmse = float("inf"); best_state = None; best_epoch = -1
            start = time.time()
            for epoch, order in enumerate(orders):
                model.train(); total = 0.0; nseen = 0
                for offset in range(0, len(order), a.batch_size):
                    ix = torch.as_tensor(order[offset:offset+a.batch_size], device=device)
                    pred_kt = model(X[ix], C[ix] if arm in ("AC", "FAC") else None)
                    error = (pred_kt * CL[ix] - GH[ix]) / 1000.0
                    loss = error.square().mean()
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step(); total += float(loss.detach()) * len(ix); nseen += len(ix)
                scheduler.step(); model.eval()
                with torch.inference_mode():
                    pred = []
                    for offset in range(0, len(val_ids), a.batch_size):
                        ix = torch.as_tensor(val_ids[offset:offset+a.batch_size], device=device)
                        pred.append((model(X[ix], C[ix] if arm in ("AC", "FAC") else None) * CL[ix]).cpu().numpy())
                    pred = np.concatenate(pred).astype(np.float64)
                truth = np.asarray(ghi[ids[val_ids]], dtype=np.float64)
                rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                mae = float(np.mean(np.abs(pred - truth)))
                rec = {"station": st, "arm": arm, "epoch": epoch, "train_mse_scaled": total/nseen,
                       "val_rmse_wm2": rmse, "val_mae_wm2": mae,
                       "lr": optimizer.param_groups[0]["lr"]}
                station_logs.append(rec); all_logs.append(rec)
                if rmse < best_rmse:
                    best_rmse = rmse; best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch == 0 or epoch + 1 == a.epochs or (epoch + 1) % 5 == 0:
                    print(json.dumps(rec), flush=True)
            model.load_state_dict(best_state, strict=True); model.eval()
            with torch.inference_mode():
                pred = []
                for offset in range(0, len(val_ids), a.batch_size):
                    ix = torch.as_tensor(val_ids[offset:offset+a.batch_size], device=device)
                    pred.append((model(X[ix], C[ix] if arm in ("AC", "FAC") else None) * CL[ix]).cpu().numpy())
                pred_before_reload = np.concatenate(pred).astype(np.float64)
            checkpoint_path = station_dir / f"{arm}_best.pt"
            torch.save({"model": best_state, "station": st, "arm": arm, "seed": a.seed,
                        "best_epoch": best_epoch, "best_val_rmse_wm2": best_rmse,
                        "normalization_contract_sha256": norms_sha256,
                        "station_normalization": norms_payload[st]}, checkpoint_path)
            loaded = torch.load(checkpoint_path, map_location=device)
            if loaded.get("normalization_contract_sha256") != norms_sha256:
                raise ValueError(f"Normalization contract hash mismatch after checkpoint reload: {st}/{arm}")
            reloaded_model = HubeiLocalGHIHead(arm).to(device).eval()
            reloaded_model.load_state_dict(loaded["model"], strict=True)
            with torch.inference_mode():
                pred = []
                for offset in range(0, len(val_ids), a.batch_size):
                    ix = torch.as_tensor(val_ids[offset:offset+a.batch_size], device=device)
                    pred.append((reloaded_model(X[ix], C[ix] if arm in ("AC", "FAC") else None) * CL[ix]).cpu().numpy())
                pred = np.concatenate(pred).astype(np.float64)
            reload_delta = float(np.max(np.abs(pred - pred_before_reload)))
            if not np.allclose(pred, pred_before_reload, rtol=1e-6, atol=1e-4):
                raise RuntimeError(f"Checkpoint reload parity failed for {st}/{arm}: max_abs={reload_delta}")
            truth = np.asarray(ghi[ids[val_ids]], dtype=np.float64)
            np.save(station_dir / f"{arm}_val_prediction_ghi_wm2.npy", pred)
            val_rows = [rows[ids[i]] for i in val_ids]
            pred_bundle = {"station": np.asarray([r["station"] for r in val_rows]),
                           "truth_ghi_wm2": truth.astype(np.float64),
                           "pred_ghi_wm2": pred.astype(np.float64),
                           "clear_sky_ghi_wm2": np.asarray([float(r["clear_sky_wm2"]) for r in val_rows], dtype=np.float64)}
            if a.input_mode == "forecast":
                pred_bundle.update({"init_time_bjt": np.asarray([r["init_time_bjt"] for r in val_rows]),
                                    "target_time_bjt": np.asarray([r["target_time_bjt"] for r in val_rows]),
                                    "lead_minutes": np.asarray([int(r["lead_minutes"]) for r in val_rows], dtype=np.int16)})
            else:
                pred_bundle["target_time_bjt"] = np.asarray([r["time_bjt"] for r in val_rows])
            np.savez_compressed(station_dir / f"{arm}_validation_predictions_with_keys.npz", **pred_bundle)
            result = {"station": st, "arm": arm, "seed": a.seed, "best_epoch": best_epoch,
                      "val_n": int(len(val_ids)), "rmse_wm2": float(np.sqrt(np.mean((pred-truth)**2))),
                      "mae_wm2": float(np.mean(np.abs(pred-truth))),
                      "prediction_sha256": sha(station_dir / f"{arm}_val_prediction_ghi_wm2.npy"),
                      "prediction_bundle_sha256": sha(station_dir / f"{arm}_validation_predictions_with_keys.npz"),
                      "input_mode": a.input_mode,
                      "checkpoint_sha256": sha(checkpoint_path),
                      "normalization_contract_sha256": norms_sha256,
                      "checkpoint_reload_max_abs_delta_wm2": reload_delta,
                      "checkpoint_reload_parity": "PASS",
                      "test_payloads_read": 0}
            (station_dir / f"{arm}_validation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"ARM_COMPLETE station={st} arm={arm} elapsed_s={time.time()-start:.1f}", flush=True)
            del model, reloaded_model, optimizer, scheduler
            torch.cuda.empty_cache()
        del X, C, GH, CL
        with (station_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
            for rec in station_logs:
                f.write(json.dumps(rec) + "\n")
    with (a.output / "all_metrics.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_logs:
            f.write(json.dumps(rec) + "\n")
    run["state"] = "HUBEI_GHI_HEADS_SEED_COMPLETE"
    run["elapsed_s"] = float(time.time() - run_start)
    (a.output / "run_metadata.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print("HUBEI_GHI_HEADS_SEED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
