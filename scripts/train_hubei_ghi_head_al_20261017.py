#!/usr/bin/env python3
"""Train the plan-triggered same-time Hubei AL control on train/val only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from hubei_ghi_head_al import HubeiLocalGHIHeadAL, parameter_count
from train_hubei_ghi_heads import HubeiLocalGHIHead, STATIONS, sha as file_sha


def sha(path: Path) -> str:
    return file_sha(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--r-feature-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--profile-output", type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AL training requires a scheduler-assigned CUDA GPU")
    if a.output.exists():
        raise FileExistsError(a.output)

    pack_meta = json.loads((a.pack / "pack_report.json").read_text(encoding="utf-8"))
    bank_meta = json.loads((a.r_feature_bank / "AL_FEATURE_REPORT.json").read_text(encoding="utf-8"))
    if pack_meta.get("state") != "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("Same-time train/validation pack gate failed")
    if pack_meta.get("test_and_buffer_values_read") != 0 or bank_meta.get("test_payloads_read") != 0:
        raise ValueError("Test/buffer payload gate failed")
    if bank_meta.get("state") != "HUBEI_FROZEN_R_DEC1_AL_FEATURE_BANK_TRAINVAL_COMPLETE_TEST_CLOSED":
        raise ValueError("Frozen-R feature bank is incomplete")
    if bank_meta.get("pack_report_sha256") != sha(a.pack / "pack_report.json"):
        raise ValueError("AL features and GHI pack are not bound to the same pack")

    with (a.pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with (a.r_feature_bank / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        feature_rows = list(csv.DictReader(f))
    feature_index = {}
    for i, r in enumerate(feature_rows):
        key = (r["time_bjt"], r["split"])
        if key in feature_index:
            raise ValueError(f"Duplicate AL feature key {key}")
        feature_index[key] = i
    r_all = np.load(a.r_feature_bank / "r_dec1_station_3x3.npy", mmap_mode="r", allow_pickle=False)
    if r_all.shape != (len(feature_rows), 3, 16, 3, 3):
        raise ValueError(f"Unexpected AL sidecar shape {r_all.shape}")
    split = np.asarray([r["split"] for r in rows])
    station = np.asarray([r["station"] for r in rows])
    times = np.asarray([r["time_bjt"].replace("-", "").replace(":", "").replace(" ", "")[:12] for r in rows])
    if len(set(zip(station.tolist(), times.tolist()))) != len(rows):
        raise ValueError("Duplicate GHI station/time key")

    agri = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geom = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    ghi = np.load(a.pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(a.pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    if agri.shape != (len(rows), 13, 3, 3) or geom.shape != (len(rows), 2, 3, 3):
        raise ValueError("Same-time GHI pack shape mismatch")
    if any(not np.isfinite(x).all() for x in (agri, geom, ghi, clear)) or (clear <= 0).any() or (ghi < 0).any():
        raise ValueError("Nonfinite or invalid train/validation values")

    # Build a keyed AL feature matrix from the exact GHI pack ordering.
    r_input = np.empty((len(rows), 16, 3, 3), dtype=np.float32)
    for i, row in enumerate(rows):
        t = row["time_bjt"].replace("-", "").replace(":", "").replace(" ", "")[:12]
        key = (t, row["split"])
        if key not in feature_index:
            raise ValueError(f"Missing AL sidecar feature for {(row['station'], t)}")
        r_input[i] = r_all[feature_index[key], STATIONS.index(row["station"])]
    if not np.isfinite(r_input).all():
        raise ValueError("Nonfinite AL feature values")

    # Exact station-specific train-only standardization of AGRI, COT-like
    # hidden features; physical geometry is left unscaled as in A/AC/AP.
    norms = {}
    for st in STATIONS:
        train_ids = np.flatnonzero((split == "train") & (station == st))
        if not len(train_ids):
            raise ValueError(f"No train rows for {st}")
        xa = np.asarray(agri[train_ids], dtype=np.float64)
        rf = np.asarray(r_input[train_ids], dtype=np.float64)
        am = xa.mean(axis=(0, 2, 3)); ast = np.maximum(xa.std(axis=(0, 2, 3)), 1e-6)
        rm = rf.mean(axis=(0, 2, 3)); rst = np.maximum(rf.std(axis=(0, 2, 3)), 1e-6)
        norms[st] = {"agri_mean": am, "agri_std": ast, "r_mean": rm, "r_std": rst,
                     "fit_rows": int(len(train_ids))}

    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    if a.profile_only:
        if a.profile_output is None:
            raise ValueError("--profile-output is required with --profile-only")
        st = min(STATIONS, key=lambda s: int(np.sum((split == "train") & (station == s))))
        ids = np.flatnonzero(station == st)
        local_train = np.flatnonzero(split[ids] == "train")
        z = norms[st]
        x = np.asarray(agri[ids], dtype=np.float32)
        x = (x-z["agri_mean"].astype(np.float32)[None,:,None,None])/z["agri_std"].astype(np.float32)[None,:,None,None]
        x = np.concatenate((x, np.asarray(geom[ids], dtype=np.float32)), axis=1)
        rr = np.asarray(r_input[ids], dtype=np.float32)
        rr = (rr-z["r_mean"].astype(np.float32)[None,:,None,None])/z["r_std"].astype(np.float32)[None,:,None,None]
        X=torch.from_numpy(x).to(device); R=torch.from_numpy(rr).to(device)
        GH=torch.as_tensor(np.asarray(ghi[ids],dtype=np.float32),device=device)
        CL=torch.as_tensor(np.asarray(clear[ids],dtype=np.float32),device=device)
        candidates=[b for b in (128,256,512,1024) if len(local_train)//b>=12]
        measurements=[]
        for batch in candidates:
            model=HubeiLocalGHIHeadAL().to(device).train()
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            ix=torch.as_tensor(local_train[:batch],device=device)
            def step():
                pred=model(X[ix],R[ix]); loss=(((pred*CL[ix]-GH[ix])/1000.0)**2).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            for _ in range(3): step()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
            for _ in range(20): step()
            torch.cuda.synchronize();elapsed=time.perf_counter()-t0
            measurements.append({"batch":batch,"updates_per_epoch":len(local_train)//batch,
                                 "examples_per_second":batch*20/elapsed,
                                 "peak_GiB":torch.cuda.max_memory_allocated()/1024**3})
            del model,opt;torch.cuda.empty_cache()
        result={"state":"HUBEI_GHI_HEAD_AL_SINGLE_GPU_PROFILE_COMPLETE","profile_station":st,
                "train_rows":len(local_train),"input_mode":"same_time","measurements":measurements,
                "AP_parameter_count":sum(x.numel() for x in HubeiLocalGHIHead("AP").parameters()),
                "AL_parameter_count":parameter_count(),"test_payloads_read":0,
                "selection":"largest batch with at least 12 train updates per epoch and within 3% of measured throughput maximum"}
        a.profile_output.parent.mkdir(parents=True,exist_ok=True)
        a.profile_output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2),flush=True)
        return

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)
    # The existing A/AC/AP trainer advances torch RNG as follows: one A
    # template, then seeded A/AC/AP constructors per station. Recreate each A
    # template so AL's AGRI trunk/readout start from the same A weights.
    base_states = {}
    for st in STATIONS:
        base = HubeiLocalGHIHead("A")
        base_states[st] = {k: v.detach().clone() for k, v in base.state_dict().items()}
        for variant in ("A", "AC", "AP"):
            torch.manual_seed(a.seed); _ = HubeiLocalGHIHead(variant)

    a.output.mkdir(parents=True)
    metadata = {
        "state": "HUBEI_GHI_HEAD_AL_TRAINING_STARTED", "arm": "AL",
        "definition": "A plus frozen Hubei R dec1 16-channel local 3x3 feature; R parameters detached/frozen",
        "seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size,
        "pack_report_sha256": sha(a.pack / "pack_report.json"),
        "feature_report_sha256": sha(a.r_feature_bank / "AL_FEATURE_REPORT.json"),
        "trainer_sha256": sha(Path(__file__)), "model_sha256": sha(Path(__file__).with_name("hubei_ghi_head_al.py")),
        "parameter_count": parameter_count(),
        "AP_parameter_count": sum(x.numel() for x in HubeiLocalGHIHead("AP").parameters()),
        "AC_parameter_count": sum(x.numel() for x in HubeiLocalGHIHead("AC").parameters()),
        "AL_parameter_delta_vs_AP": parameter_count() - sum(x.numel() for x in HubeiLocalGHIHead("AP").parameters()),
        "split_sha256": pack_meta["split_sha256"],
        "normalization": "station-specific train-only AGRI and R dec1 feature; physical geometry",
        "loss": "MSE((pred_kt*interval_clear_sky_GHI - observed_GHI)/1000)",
        "test_payloads_read": 0, "gpu": torch.cuda.get_device_name(0),
    }
    (a.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (a.output / "normalization_train_only.json").write_text(
        json.dumps({st: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in z.items()}
                    for st, z in norms.items()}, indent=2) + "\n", encoding="utf-8")
    all_records = []
    t_all = time.time()
    for st in STATIONS:
        ids = np.flatnonzero(station == st)
        local_split = split[ids]
        train_ids = np.flatnonzero(local_split == "train")
        val_ids = np.flatnonzero(local_split == "val")
        if len(train_ids) // a.batch_size < 12:
            raise ValueError(f"batch_size={a.batch_size} gives fewer than 12 updates for {st}")
        z = norms[st]
        x = np.asarray(agri[ids], dtype=np.float32)
        x = (x - z["agri_mean"].astype(np.float32)[None, :, None, None]) / z["agri_std"].astype(np.float32)[None, :, None, None]
        x = np.concatenate((x, np.asarray(geom[ids], dtype=np.float32)), axis=1)
        rr = np.asarray(r_input[ids], dtype=np.float32)
        rr = (rr - z["r_mean"].astype(np.float32)[None, :, None, None]) / z["r_std"].astype(np.float32)[None, :, None, None]
        X = torch.from_numpy(x).to(device); R = torch.from_numpy(rr).to(device)
        GH = torch.as_tensor(np.asarray(ghi[ids], dtype=np.float32), device=device)
        CL = torch.as_tensor(np.asarray(clear[ids], dtype=np.float32), device=device)
        station_dir = a.output / st; station_dir.mkdir()
        orders = [rng.permutation(train_ids) for _ in range(a.epochs)]
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        model = HubeiLocalGHIHeadAL().to(device)
        # Transfer the A common trunk/readout exactly; the new feature channel
        # enters via an independently initialized 16->4 adapter. As in AP,
        # the new readout weights start at zero to preserve the A prediction.
        base_state = base_states[st]
        model.agri_encoder.load_state_dict({k.removeprefix("agri_encoder."): v for k, v in base_state.items() if k.startswith("agri_encoder.")}, strict=True)
        old_w = base_state["readout.0.weight"]
        # Copy the frozen A head initialization without autograd tracking.
        # Parameters are leaf tensors requiring gradients; raw in-place writes
        # here raise in current PyTorch versions unless guarded by no_grad.
        with torch.no_grad():
            model.readout[0].weight.zero_()
            model.readout[0].weight[:, :old_w.shape[1]].copy_(old_w)
            model.readout[0].bias.copy_(base_state["readout.0.bias"])
        for name in ("2", "4"):
            prefix = f"readout.{name}."
            layer = model.readout[int(name)]
            layer.load_state_dict({k[len(prefix):]: v for k, v in base_state.items() if k.startswith(prefix)}, strict=True)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
        best_rmse = float("inf"); best_state = None; best_epoch = -1
        records = []
        for epoch, order in enumerate(orders):
            model.train(); total = 0.0; seen = 0
            for offset in range(0, len(order), a.batch_size):
                ix = torch.as_tensor(order[offset:offset+a.batch_size], device=device)
                pred = model(X[ix], R[ix]); err = (pred * CL[ix] - GH[ix]) / 1000.0
                loss = err.square().mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                opt.step(); total += float(loss.detach()) * len(ix); seen += len(ix)
            scheduler.step(); model.eval()
            with torch.inference_mode():
                pred = torch.cat([model(X[torch.as_tensor(val_ids[j:j+a.batch_size], device=device)],
                                        R[torch.as_tensor(val_ids[j:j+a.batch_size], device=device)])
                                  for j in range(0, len(val_ids), a.batch_size)]) * CL[torch.as_tensor(val_ids, device=device)]
                pred = pred.cpu().numpy().astype(np.float64)
            truth = np.asarray(ghi[ids[val_ids]], dtype=np.float64)
            rmse = float(np.sqrt(np.mean((pred-truth)**2))); mae = float(np.mean(np.abs(pred-truth)))
            rec = {"station": st, "arm": "AL", "seed": a.seed, "epoch": epoch,
                   "train_mse_scaled": total/seen, "val_rmse_wm2": rmse,
                   "val_mae_wm2": mae, "lr": opt.param_groups[0]["lr"]}
            records.append(rec); all_records.append(rec)
            if rmse < best_rmse:
                best_rmse, best_epoch = rmse, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 0 or epoch + 1 == a.epochs or (epoch + 1) % 5 == 0:
                print(json.dumps(rec), flush=True)
        model.load_state_dict(best_state, strict=True); model.eval()
        with torch.inference_mode():
            vix = torch.as_tensor(val_ids, device=device)
            prediction = (model(X[vix], R[vix]) * CL[vix]).cpu().numpy().astype(np.float64)
        truth = np.asarray(ghi[ids[val_ids]], dtype=np.float64)
        val_rows = [rows[ids[i]] for i in val_ids]
        np.save(station_dir / "AL_val_prediction_ghi_wm2.npy", prediction)
        np.savez_compressed(station_dir / "AL_validation_predictions_with_keys.npz",
                            station=np.asarray([r["station"] for r in val_rows]),
                            target_time_bjt=np.asarray([r["time_bjt"] for r in val_rows]),
                            truth_ghi_wm2=truth, pred_ghi_wm2=prediction,
                            clear_sky_ghi_wm2=np.asarray([float(r["clear_sky_wm2"]) for r in val_rows]))
        torch.save({"model": best_state, "station": st, "arm": "AL", "seed": a.seed,
                    "best_epoch": best_epoch, "best_val_rmse_wm2": best_rmse}, station_dir / "AL_best.pt")
        reloaded = HubeiLocalGHIHeadAL().to(device).eval()
        saved = torch.load(station_dir / "AL_best.pt", map_location=device, weights_only=False)
        reloaded.load_state_dict(saved["model"], strict=True)
        with torch.inference_mode():
            reloaded_prediction = (reloaded(X[vix], R[vix]) * CL[vix]).cpu().numpy().astype(np.float64)
        reload_diff = float(np.max(np.abs(reloaded_prediction - prediction)))
        if reload_diff > 1e-6:
            raise RuntimeError(f"AL checkpoint reload mismatch for {st}: {reload_diff}")
        summary = {"station": st, "arm": "AL", "seed": a.seed, "best_epoch": best_epoch,
                   "val_n": len(val_ids), "rmse_wm2": best_rmse,
                   "reload_max_abs_difference_wm2": reload_diff,
                   "mae_wm2": float(np.mean(np.abs(prediction-truth))),
                   "prediction_sha256": sha(station_dir / "AL_val_prediction_ghi_wm2.npy"),
                   "bundle_sha256": sha(station_dir / "AL_validation_predictions_with_keys.npz"),
                   "checkpoint_sha256": sha(station_dir / "AL_best.pt"), "test_payloads_read": 0}
        (station_dir / "AL_validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (station_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        del model, reloaded, saved, opt, scheduler, X, R, GH, CL
        torch.cuda.empty_cache()
    (a.output / "all_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_records), encoding="utf-8")
    metadata["state"] = "HUBEI_GHI_HEAD_AL_SEED_COMPLETE"
    metadata["elapsed_s"] = time.time() - t_all
    (a.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("HUBEI_GHI_HEAD_AL_SEED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
