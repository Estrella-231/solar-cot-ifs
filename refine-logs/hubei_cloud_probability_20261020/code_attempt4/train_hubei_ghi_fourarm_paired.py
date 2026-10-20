#!/usr/bin/env python3
"""Paired A/AC/AM/ACM Hubei same-time GHI heads, train/validation only.

All arms use identical parameter shapes, initialization, row order, optimizer,
updates, and validation selection. Inactive auxiliary inputs are exactly zero.
"""
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


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AM", "ACM")
ARM_BITS = {"A": (0, 0), "AC": (1, 0), "AM": (0, 1), "ACM": (1, 1)}


class PairedGHIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(15, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 1), nn.SiLU(),
        )
        # Two reserved auxiliary channels give every arm exactly the same
        # parameter count. Missing modalities are zeroed before this encoder.
        self.aux_encoder = nn.Sequential(nn.Conv2d(2, 4, 1, bias=False), nn.SiLU())
        self.readout = nn.Sequential(
            nn.Linear(36 * 3 * 3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, agri, aux):
        if tuple(agri.shape[1:]) != (15, 3, 3) or tuple(aux.shape[1:]) != (2, 3, 3):
            raise ValueError("expected AGRI [B,15,3,3] and aux [B,2,3,3]")
        return self.readout(torch.cat((self.agri_encoder(agri), self.aux_encoder(aux)), 1).flatten(1)).squeeze(-1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {"n": int(err.size), "rmse_wm2": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
            "mae_wm2": float(np.mean(np.abs(err), dtype=np.float64)),
            "bias_wm2": float(np.mean(err, dtype=np.float64))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Four-arm paired training requires allocated CUDA")
    if a.output.exists():
        raise FileExistsError(f"refusing existing output {a.output}")
    report_path = a.pack / "pack_report.json"
    pack_report = json.loads(report_path.read_text(encoding="utf-8"))
    if pack_report.get("state") != "HUBEI_GHI_FOURARM_COMMON_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("unexpected four-arm GHI pack state")
    if pack_report.get("test_payloads_read") != 0 or pack_report.get("buffer_payloads_read") != 0:
        raise ValueError("four-arm pack violates test/buffer boundary")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8-sig")))
    n = len(rows)
    split = np.asarray([r["split"] for r in rows])
    station = np.asarray([r["station"] for r in rows])
    agri = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.pack / "cot_pred_3x3.npy", mmap_mode="r", allow_pickle=False)
    pcloud = np.load(a.pack / "p_cloud_3x3.npy", mmap_mode="r", allow_pickle=False)
    ghi = np.load(a.pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(a.pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    if agri.shape != (n, 13, 3, 3) or geometry.shape != (n, 2, 3, 3):
        raise ValueError("AGRI/geometry dimensions mismatch")
    if cot.shape != (n, 1, 3, 3) or pcloud.shape != (n, 1, 3, 3):
        raise ValueError("COT/p_cloud dimensions mismatch")
    if any(not np.isfinite(x).all() for x in (agri, geometry, cot, pcloud, ghi, clear)):
        raise ValueError("nonfinite train/validation values")
    if np.any(cot < 0) or np.any((pcloud < 0) | (pcloud > 1)) or np.any(clear <= 0) or np.any(ghi < 0):
        raise ValueError("physical input/label range check failed")

    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    run_start = time.time()
    run = {"state": "HUBEI_GHI_FOURARM_TRAINING_STARTED", "seed": a.seed,
           "arms": list(ARMS), "stations": list(STATIONS), "epochs": a.epochs,
           "batch_size": a.batch_size, "pack_report_sha256": sha(report_path),
           "trainer_sha256": sha(Path(__file__)), "test_payloads_read": 0,
           "gpu": torch.cuda.get_device_name(0),
           "pairing": "same per-station initialization, row permutations, updates, optimizer and epoch-selection rule across all four arms",
           "normalization": "per-station train-only AGRI; physical geometry; per-station train-only log1p predicted-COT and p_cloud standardization",
           "input_switch": "same two-channel auxiliary encoder in every arm; inactive COT/p_cloud channels are exactly zero",
           "loss": "MSE((predicted kt * interval clear-sky GHI - observed GHI)/1000); labels in W m-2"}
    (a.output / "run_metadata.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rng = np.random.default_rng(a.seed)
    all_metrics = []
    norm_contract = {}

    for st in STATIONS:
        ids = np.flatnonzero(station == st)
        local_train = np.flatnonzero(split[ids] == "train")
        local_val = np.flatnonzero(split[ids] == "val")
        if len(local_train) // a.batch_size < 12 or not len(local_val):
            raise ValueError(f"invalid sample support for {st}")
        Xraw = np.concatenate((np.asarray(agri[ids]), np.asarray(geometry[ids])), axis=1).astype(np.float32)
        COT = np.asarray(cot[ids], dtype=np.float32)
        PC = np.asarray(pcloud[ids], dtype=np.float32)
        GH = np.asarray(ghi[ids], dtype=np.float32)
        CL = np.asarray(clear[ids], dtype=np.float32)
        xm = Xraw[local_train, :13].astype(np.float64).mean(axis=(0, 2, 3))
        xs = Xraw[local_train, :13].astype(np.float64).std(axis=(0, 2, 3))
        xs = np.maximum(xs, 1e-6)
        X = Xraw.copy()
        X[:, :13] = (Xraw[:, :13] - xm.astype(np.float32)[None, :, None, None]) / xs.astype(np.float32)[None, :, None, None]
        c_log = np.log1p(COT.astype(np.float64))
        cm = c_log[local_train].mean(dtype=np.float64)
        cs = max(float(c_log[local_train].std(dtype=np.float64)), 1e-6)
        pm = float(PC[local_train].mean(dtype=np.float64))
        ps = max(float(PC[local_train].std(dtype=np.float64)), 1e-6)
        aux0 = ((c_log - cm) / cs).astype(np.float32)
        aux1 = ((PC - pm) / ps).astype(np.float32)
        aux_full = np.concatenate((aux0, aux1), axis=1)
        X[~np.isfinite(X)] = 0.0
        aux_full[~np.isfinite(aux_full)] = 0.0
        norm_contract[st] = {"agri_mean": xm.tolist(), "agri_std": xs.tolist(),
                             "cot_log1p_mean": cm, "cot_log1p_std": cs,
                             "p_cloud_mean": pm, "p_cloud_std": ps,
                             "fit_split": "train only"}
        XT = torch.as_tensor(X, dtype=torch.float32, device=device)
        AT = torch.as_tensor(aux_full, dtype=torch.float32, device=device)
        GHT = torch.as_tensor(GH, dtype=torch.float32, device=device)
        CLT = torch.as_tensor(CL, dtype=torch.float32, device=device)
        orders = [rng.permutation(local_train) for _ in range(a.epochs)]

        torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        template = PairedGHIHead().to(device)
        init_state = {k: v.detach().clone() for k, v in template.state_dict().items()}
        init_sha = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for _, v in sorted(init_state.items()))).hexdigest()
        station_dir = a.output / st
        station_dir.mkdir()
        for arm in ARMS:
            model = PairedGHIHead().to(device)
            model.load_state_dict(init_state, strict=True)
            cot_on, cloud_on = ARM_BITS[arm]
            aux = AT.clone()
            if not cot_on:
                aux[:, 0] = 0.0
            if not cloud_on:
                aux[:, 1] = 0.0
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-5)
            best_rmse, best_state, best_epoch = float("inf"), None, -1
            arm_records = []
            t0 = time.time()
            for epoch, order in enumerate(orders, start=1):
                model.train()
                total, count = 0.0, 0
                for start in range(0, len(order), a.batch_size):
                    batch_ids = torch.as_tensor(order[start:start + a.batch_size], device=device)
                    pred_kt = model(XT[batch_ids], aux[batch_ids])
                    err = (pred_kt * CLT[batch_ids] - GHT[batch_ids]) / 1000.0
                    loss = err.square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nonfinite loss {st}/{arm}/{epoch}")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step(); total += float(loss.detach()) * len(batch_ids); count += len(batch_ids)
                scheduler.step()
                model.eval()
                with torch.inference_mode():
                    vpred = torch.cat([model(XT[ix], aux[ix]) * CLT[ix]
                                       for ix in torch.as_tensor(local_val, device=device).split(a.batch_size)]).cpu().numpy().astype(np.float64)
                truth = GH[local_val].astype(np.float64)
                met = metric(vpred, truth)
                rec = {"station": st, "arm": arm, "seed": a.seed, "epoch": epoch,
                       "train_scaled_mse": total / max(count, 1), **{f"val_{k}": v for k, v in met.items()}}
                arm_records.append(rec); all_metrics.append(rec)
                if met["rmse_wm2"] < best_rmse:
                    best_rmse, best_epoch = met["rmse_wm2"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch == 1 or epoch == a.epochs or epoch % 5 == 0:
                    print(json.dumps(rec), flush=True)
            if best_state is None:
                raise RuntimeError(f"no selected checkpoint for {st}/{arm}")
            model.load_state_dict(best_state, strict=True); model.eval()
            with torch.inference_mode():
                val_ids_t = torch.as_tensor(local_val, device=device)
                pred0 = torch.cat([model(XT[ix], aux[ix]) * CLT[ix] for ix in val_ids_t.split(a.batch_size)]).cpu().numpy().astype(np.float64)
            ckpt = station_dir / f"{arm}_best.pt"
            torch.save({"model": best_state, "station": st, "arm": arm, "seed": a.seed,
                        "best_epoch": best_epoch, "initialization_sha256": init_sha,
                        "norm_contract": norm_contract[st], "parameter_count": sum(v.numel() for v in model.parameters()),
                        "test_payloads_read": 0}, ckpt)
            chk = torch.load(ckpt, map_location=device, weights_only=False)
            reload_model = PairedGHIHead().to(device).eval(); reload_model.load_state_dict(chk["model"], strict=True)
            with torch.inference_mode():
                pred1 = torch.cat([reload_model(XT[ix], aux[ix]) * CLT[ix] for ix in val_ids_t.split(a.batch_size)]).cpu().numpy().astype(np.float64)
            delta = float(np.max(np.abs(pred0 - pred1)))
            if not np.allclose(pred0, pred1, rtol=1e-6, atol=1e-4):
                raise RuntimeError(f"checkpoint reload parity failed {st}/{arm}: {delta}")
            val_rows = [rows[ids[i]] for i in local_val]
            pred_path = station_dir / f"{arm}_validation_predictions.npz"
            np.savez_compressed(pred_path, station=np.asarray([r["station"] for r in val_rows]),
                                target_time_bjt=np.asarray([r["time_bjt"] for r in val_rows]),
                                split=np.asarray([r["split"] for r in val_rows]),
                                pred_ghi_wm2=pred1, truth_ghi_wm2=GH[local_val].astype(np.float64),
                                clear_sky_wm2=CL[local_val].astype(np.float64))
            result = {"station": st, "arm": arm, "seed": a.seed, "best_epoch": best_epoch,
                      "initialization_sha256": init_sha, "val_metrics_fp64": metric(pred1, GH[local_val]),
                      "prediction_sha256": sha(pred_path), "checkpoint_sha256": sha(ckpt),
                      "checkpoint_reload_max_abs_delta_wm2": delta, "checkpoint_reload_parity": "PASS",
                      "test_payloads_read": 0}
            (station_dir / f"{arm}_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            with (station_dir / f"{arm}_metrics.jsonl").open("w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(x) for x in arm_records) + "\n")
            print(f"ARM_COMPLETE {st} {arm} seed={a.seed} best_epoch={best_epoch} seconds={time.time()-t0:.1f}", flush=True)
            del model, reload_model, optimizer, scheduler, aux
            torch.cuda.empty_cache()
        del XT, AT, GHT, CLT

    (a.output / "all_metrics.jsonl").write_text("\n".join(json.dumps(x) for x in all_metrics) + "\n", encoding="utf-8")
    (a.output / "normalization_contract.json").write_text(json.dumps({"fit_split": "train only", "stations": norm_contract}, indent=2) + "\n", encoding="utf-8")
    run["state"] = "HUBEI_GHI_FOURARM_SEED_COMPLETE"
    run["parameter_counts"] = {arm: sum(v.numel() for v in PairedGHIHead().parameters()) for arm in ARMS}
    run["elapsed_seconds"] = float(time.time() - run_start)
    (a.output / "run_metadata.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("HUBEI_GHI_FOURARM_SEED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
