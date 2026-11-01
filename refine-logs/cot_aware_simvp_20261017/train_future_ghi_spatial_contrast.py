#!/usr/bin/env python3
"""Train the one-factor, center-preserving spatial-COT GHI arm on one GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HIGH_KT = HERE.parents[1] / "experiments" / "high_kt_loss_20261016"
sys.path.insert(0, str(HIGH_KT))
from train_loss_pair import IntervalHead  # noqa: E402
from train_future_ghi_heads_cot_reliability import load_station, seed_all, score, sha  # noqa: E402


def spatial_input(data: dict, split: str, station: str, calibration: dict) -> torch.Tensor:
    out = data["cot_calibrated"].clone()
    history = data["history_cot_mask"]
    keys = data["keys"]
    offsets = (-30, -15, 0)
    for pos, offset in enumerate(offsets):
        for frame_lead in sorted({int(k["lead_minutes"]) + offset for k in keys if int(k["lead_minutes"]) + offset > 0}):
            mask_np = np.asarray([int(k["lead_minutes"]) + offset == frame_lead for k in keys], dtype=bool)
            if split == "train":
                fold_map = calibration["fold_by_sequence_index"]
                fold_ids = np.asarray([int(fold_map[str(int(k["sequence_index_within_split"]))]) for k in keys], dtype=np.int8)
                for fold in range(5):
                    ids = np.flatnonzero(mask_np & (fold_ids == fold))
                    if not len(ids): continue
                    gamma = float(calibration["parameters_excluding_each_fold"][str(fold)][station][str(frame_lead)]["gamma"])
                    patch = out[ids, pos, 0]
                    center = patch[:, 1:2, 1:2]
                    out[ids, pos, 0] = center + gamma * (patch - center)
            else:
                ids = np.flatnonzero(mask_np)
                if len(ids):
                    gamma = float(calibration["parameters_full_train"][station][str(frame_lead)]["gamma"])
                    patch = out[ids, pos, 0]
                    center = patch[:, 1:2, 1:2]
                    out[ids, pos, 0] = center + gamma * (patch - center)
    if not torch.isfinite(out).all(): raise ValueError("nonfinite spatially calibrated COT")
    if not torch.equal(out[:, :, :, 1, 1], data["cot_calibrated"][:, :, :, 1, 1]):
        raise ValueError("spatial contrast transform changed center COT")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--center-calibration", type=Path, required=True)
    p.add_argument("--spatial-calibration", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1792)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    manifest = json.loads((args.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    center = json.loads(args.center_calibration.read_text(encoding="utf-8"))
    spatial = json.loads(args.spatial_calibration.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK" or manifest.get("train_forecast_oof") is not True:
        raise ValueError("five-fold OOF GHI pack required")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer not certified closed")
    if center.get("forecast_pack_manifest_sha256") != sha(args.pack / "pack_manifest.json"):
        raise ValueError("center calibration does not match GHI pack")
    if spatial.get("forecast_pack_manifest_sha256") != sha(args.pack / "pack_manifest.json") or spatial.get("state") != "COMPLETE_CROSSFIT_TRAIN_ONLY_SPATIAL_CONTRAST_AUDIT":
        raise ValueError("spatial calibration does not match GHI pack or is incomplete")
    if spatial.get("validation_labels_used_for_fit") is not False or spatial.get("test_payload_opened") is not False or spatial.get("buffer_payload_opened") is not False:
        raise ValueError("spatial calibration provenance is invalid")
    if args.output_root.exists() and any((args.output_root / s / f"seed{args.seed}" / "ACRELSP").exists() for s in ("sili", "zhujia")):
        raise FileExistsError("candidate seed output already exists")
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    summaries = []
    for station in ("sili", "zhujia"):
        tr = load_station(args.pack, "train", station, device, manifest, center)
        va = load_station(args.pack, "val", station, device, manifest, center)
        tr_cot = spatial_input(tr, "train", station, spatial)
        va_cot = spatial_input(va, "val", station, spatial)
        n = len(tr["ghi"])
        if n // args.batch_size < 12: raise ValueError("batch size leaves fewer than 12 updates per epoch")
        seed_all(args.seed)
        base = IntervalHead().to(device)
        initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
        rng = np.random.default_rng(args.seed)
        orders = [rng.permutation(n) for _ in range(args.epochs)]
        run = args.output_root / station / f"seed{args.seed}" / "ACRELSP"
        run.mkdir(parents=True, exist_ok=False)
        model = IntervalHead().to(device); model.load_state_dict(initial, strict=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
        best_rmse = float("inf"); best_state = None; history=[]; updates=0; start=time.monotonic()
        for epoch, order in enumerate(orders):
            model.train(); weighted=0.0
            for lo in range(0,n,args.batch_size):
                ix=torch.as_tensor(order[lo:lo+args.batch_size],device=device)
                pred_kt=model(tr["agri"][ix],tr["geometry"][ix],tr_cot[ix])
                loss=(((pred_kt*tr["clear"][ix]-tr["ghi"][ix])/1000.0)**2).mean()
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite GHI loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
                opt.step();updates+=1;weighted+=float(loss.detach())*len(ix)
            sched.step()
            pred=score(model,{**va,"cot_calibrated":va_cot},"ACREL",args.batch_size)
            truth=va["ghi"].detach().cpu().numpy().astype(np.float64)
            rmse=float(np.sqrt(np.mean((pred-truth)**2)))
            history.append({"epoch":epoch+1,"train_scaled_mse":weighted/n,"val_rmse_wm2":rmse,"updates":updates})
            if rmse<best_rmse:
                best_rmse=rmse;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==0 or (epoch+1)%5==0:
                print(json.dumps({"station":station,"seed":args.seed,**history[-1]},separators=(",",":")),flush=True)
        model.load_state_dict(best_state,strict=True);model.eval()
        # Save predictions with the selected model and exact paired validation keys.
        with torch.inference_mode():
            pred_parts=[]
            for lo in range(0,len(va["ghi"]),args.batch_size):
                sl=slice(lo,min(lo+args.batch_size,len(va["ghi"])))
                kt=model(va["agri"][sl],va["geometry"][sl],va_cot[sl])
                pred_parts.append((kt*va["clear"][sl]).cpu().numpy())
        pred=np.concatenate(pred_parts).astype(np.float64);truth=va["ghi"].cpu().numpy().astype(np.float64)
        pp=run/"validation_predictions.npz"
        np.savez_compressed(pp,prediction_ghi_wm2=pred,truth_ghi_wm2=truth,
            keys=np.asarray([json.dumps(x,sort_keys=True) for x in va["keys"]]))
        ck=run/"best.pt"
        torch.save({"model":best_state,"arm":"ACRELSP","station":station,"seed":args.seed,
            "best_val_rmse_wm2":best_rmse,"pack_manifest_sha256":sha(args.pack/"pack_manifest.json"),
            "spatial_calibration_sha256":sha(args.spatial_calibration)},ck)
        metrics={"station":station,"seed":args.seed,"arm":"ACRELSP","n_train":n,"n_val":len(truth),
            "best_val_rmse_wm2":float(np.sqrt(np.mean((pred-truth)**2))),"mae_wm2":float(np.mean(np.abs(pred-truth))),
            "bias_wm2":float(np.mean(pred-truth)),"updates":updates,"elapsed_seconds":time.monotonic()-start,
            "checkpoint_sha256":sha(ck),"predictions_sha256":sha(pp),"test_payload_opened":False}
        (run/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
        (run/"training_curve.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
        summaries.append(metrics);print("ARM_COMPLETE "+json.dumps(metrics),flush=True)
        del model,opt,sched,base,tr,va,tr_cot,va_cot;torch.cuda.empty_cache()
    report={"state":"COMPLETE_FUTURE_GHI_SPATIAL_CONTRAST_SEED_RUN","seed":args.seed,"stations":["sili","zhujia"],
        "arm":"ACRELSP","epochs":args.epochs,"batch_size":args.batch_size,
        "train_forecast_oof":True,"forecast_bank_oof_fold_count":5,"split_id":manifest["split_id"],
        "baseline_comparator":"ACREL; all non-COT inputs, samples, head, optimizer, seed and epoch budget unchanged",
        "transform":"future log1p-COT only; ACREL center retained exactly; eight neighbor deviations multiplied by cross-fitted train-only gamma",
        "center_calibration_sha256":sha(args.center_calibration),"spatial_calibration_sha256":sha(args.spatial_calibration),
        "pack_manifest_sha256":sha(args.pack/"pack_manifest.json"),"runs":summaries,
        "test_payload_opened":False,"buffer_payload_opened":False,"future_observed_agri_used_as_inference_input":False,"cpp_used_as_input":False}
    out=args.output_root/f"seed{args.seed}_manifest.json"
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("SPATIAL_CONTRAST_SEED_COMPLETE "+json.dumps({"seed":args.seed,"runs":len(summaries),"test_payload_opened":False}),flush=True)


if __name__=="__main__":main()
