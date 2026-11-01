"""Paired GHI heads testing train-only OOF reliability calibration of future COT."""
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
from build_future_ghi_head_pack import history_context_mask  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_station(pack: Path, split: str, station: str, device: torch.device, norm: dict,
                 calibration: dict):
    d = pack / split
    rows = list(csv.DictReader((d / "keys.csv").open(newline="", encoding="utf-8")))
    keep = np.asarray([r["station"] == station for r in rows])
    ids = np.flatnonzero(keep)
    if not len(ids):
        raise ValueError(f"empty {split}/{station} cohort")
    ag = np.load(d / "agri_raw.npy", mmap_mode="r", allow_pickle=False)[ids]
    geo = np.load(d / "geometry.npy", mmap_mode="r", allow_pickle=False)[ids]
    cot_raw = np.asarray(np.load(d / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)[ids],
                         dtype=np.float32)
    ghi = np.load(d / "ghi.npy", mmap_mode="r", allow_pickle=False)[ids]
    clear = np.load(d / "clear.npy", mmap_mode="r", allow_pickle=False)[ids]
    if not (np.isfinite(ag).all() and np.isfinite(geo).all() and np.isfinite(cot_raw).all()):
        raise ValueError(f"non-finite {split}/{station} inputs")
    if not (np.isfinite(ghi).all() and np.isfinite(clear).all() and (clear > 0).all()):
        raise ValueError(f"invalid {split}/{station} labels")
    keys = [rows[i] for i in ids]
    cot_calibrated_raw = apply_reliability_calibration(cot_raw, keys, split, calibration)
    ag = (np.asarray(ag, dtype=np.float32) - np.asarray(norm["AGRI_mean"], np.float32)[None, None, :, None, None]) \
         / np.asarray(norm["AGRI_std"], np.float32)[None, None, :, None, None]
    cot = (cot_raw - np.float32(norm["COT_log1p_mean"])) / np.float32(norm["COT_log1p_std"])
    cot_calibrated = (cot_calibrated_raw - np.float32(norm["COT_log1p_mean"])) / np.float32(norm["COT_log1p_std"])
    history_mask = history_context_mask(
        np.asarray([int(k["lead_minutes"]) for k in keys], dtype=np.int16)
    )[:, :, None, None, None]
    return {
        "agri": torch.from_numpy(ag.copy()).to(device),
        "geometry": torch.from_numpy(np.asarray(geo, dtype=np.float32)).to(device),
        "cot": torch.from_numpy(cot.copy()).to(device),
        "cot_calibrated": torch.from_numpy(cot_calibrated.copy()).to(device),
        "history_cot_mask": torch.from_numpy(history_mask).to(device),
        "ghi": torch.from_numpy(np.asarray(ghi, dtype=np.float32)).to(device),
        "clear": torch.from_numpy(np.asarray(clear, dtype=np.float32)).to(device),
        "keys": keys,
    }


def apply_reliability_calibration(cot: np.ndarray, keys: list[dict], split: str,
                                  calibration: dict) -> np.ndarray:
    """Transform only forecast slots; cross-fit train features by heldout fold."""
    out = np.array(cot, dtype=np.float32, copy=True)
    offsets = (-30, -15, 0)
    for i, key in enumerate(keys):
        station = key["station"]
        target_lead = int(key["lead_minutes"])
        if split == "train":
            seq_index = str(int(key["sequence_index_within_split"]))
            fold = calibration["fold_by_sequence_index"].get(seq_index)
            if fold is None:
                raise ValueError("training sequence lacks its frozen OOF fold assignment")
            params = calibration["parameters_excluding_each_fold"][str(fold)]
        else:
            params = calibration["parameters_full_train"]
        for pos, offset in enumerate(offsets):
            frame_lead = target_lead + offset
            if frame_lead <= 0:
                continue
            try:
                coef = params[station][str(frame_lead)]
            except KeyError as exc:
                raise ValueError("missing station/lead COT reliability coefficient") from exc
            out[i, pos, 0] = np.float32(coef["intercept"]) + np.float32(coef["slope"]) * out[i, pos, 0]
    return out


def score(model, data, arm: str, batch: int):
    model.eval(); out = []
    n = len(data["ghi"])
    c = cot_for_arm(data, arm)
    with torch.inference_mode():
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            kt = model(data["agri"][sl], data["geometry"][sl], c[sl])
            out.append((kt * data["clear"][sl]).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def cot_for_arm(data, arm: str):
    if arm == "A":
        return torch.zeros_like(data["cot"])
    if arm == "AH":
        return data["cot"] * data["history_cot_mask"]
    if arm == "AC":
        return data["cot"]
    if arm == "AC50":
        history = data["history_cot_mask"]
        future = 1.0 - history
        return data["cot"] * history + 0.5 * data["cot"] * future
    if arm == "ACREL":
        return data["cot_calibrated"]
    raise ValueError(f"unknown COT arm: {arm}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1792)
    p.add_argument("--station", choices=("sili", "zhujia", "both"), default="both")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for future-GHI head training")
    manifest = json.loads((a.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("future-GHI pack is incomplete")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer exclusion not certified")
    train_oof = manifest.get("train_forecast_oof") is True
    if not isinstance(manifest.get("train_forecast_oof"), bool):
        raise ValueError("pack must explicitly certify whether train forecasts are OOF")
    calibration = json.loads(a.calibration.read_text(encoding="utf-8"))
    if calibration.get("state") != "COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION":
        raise ValueError("train-only cross-fitted COT calibration is incomplete")
    if calibration.get("forecast_pack_manifest_sha256") != sha(a.pack / "pack_manifest.json"):
        raise ValueError("COT calibration was fitted on another forecast input pack")
    if calibration.get("train_forecast_oof") is not True or calibration.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("COT calibration lacks five-fold OOF provenance")
    if calibration.get("test_payload_opened") is not False or calibration.get("buffer_payload_opened") is not False:
        raise ValueError("COT calibration does not certify test/buffer closure")
    if len(calibration.get("parameters_excluding_each_fold", {})) != 5 or \
            len(calibration.get("parameters_full_train", {})) != 2:
        raise ValueError("COT calibration parameter coverage is incomplete")
    a.output.mkdir(parents=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    stations = ("sili", "zhujia") if a.station == "both" else (a.station,)
    report = {"state": "RUNNING", "screening_pilot": not train_oof,
              "validation_screening": True,
              "train_forecast_oof": train_oof,
              "forecast_bank_oof_fold_count": manifest.get("forecast_bank_oof_fold_count", 0),
              "scope": ("date-block OOF training forecasts, frozen full-train validation forecasts; validation screening only, independent test not accessed"
                       if train_oof else
                       "in-sample train forecasts with frozen full-train validation forecasts; exploratory screening only, independent test not accessed"),
              "split_id": manifest["split_id"], "epochs": a.epochs, "batch_size": a.batch_size,
              "seeds": a.seeds, "stations": stations,
              "architecture": "established IntervalHead unchanged; 3x3, three frames T-30/T-15/T",
              "COT_source_arms": {"A": "no COT", "AH": "observed-history R COT only; future-predicted COT zeroed",
                                  "AC": "observed-history plus full frozen-R COT from predicted AGRI",
                                  "AC50": "observed-history COT unchanged; future normalized COT anomaly multiplied by 0.5",
                                  "ACREL": "observed-history COT unchanged; future log1p-COT affine-calibrated by train-only station/lead OOF reliability"},
              "future_cot_shrinkage": "AC50 is retained unchanged as a registered comparator; ACREL applies train-only station/lead reliability calibration",
              "primary_future_cot_contrasts": ["ACREL minus AH tests train-calibrated forecast-COT increment",
                                               "ACREL minus A tests calibrated COT versus no COT",
                                               "AC minus AH tests full future-COT increment",
                                               "AC50 minus AH tests attenuated future-COT increment",
                                               "AC50 minus AC tests 50% shrinkage versus full forecast COT",
                                               "ACREL minus AC tests reliability calibration versus raw future COT"],
              "future_cot_reliability_transform": "log1p_COT_cal = intercept[station,lead] + slope[station,lead]*forecast_log1p_COT; fold-k train rows use coefficients fitted on the other four OOF folds; validation uses all five folds",
              "reliability_calibration_sha256": sha(a.calibration),
              "test_payload_opened": False, "pack_manifest_sha256": sha(a.pack / "pack_manifest.json"),
              "source_sha256": sha(Path(__file__))}
    (a.output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summaries = []
    for station in stations:
        tr = load_station(a.pack, "train", station, device, manifest, calibration)
        va = load_station(a.pack, "val", station, device, manifest, calibration)
        n = len(tr["ghi"])
        if n // a.batch_size < 12:
            raise ValueError("batch size leaves fewer than 12 updates per epoch")
        for seed in a.seeds:
            # One common initialization and epoch order for the paired arms.
            seed_all(seed)
            base = IntervalHead().to(device)
            initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
            rng = np.random.default_rng(seed)
            orders = [rng.permutation(n) for _ in range(a.epochs)]
            for arm in ("A", "AH", "AC", "AC50", "ACREL"):
                run = a.output / station / f"seed{seed}" / arm
                run.mkdir(parents=True)
                model = IntervalHead().to(device); model.load_state_dict(initial, strict=True)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
                cot = cot_for_arm(tr, arm)
                best_rmse = float("inf"); best_state = None; history = []; updates = 0
                started = time.monotonic()
                for epoch, order in enumerate(orders):
                    model.train(); weighted = 0.0
                    for lo in range(0, n, a.batch_size):
                        ix = torch.as_tensor(order[lo:lo+a.batch_size], device=device)
                        pred_kt = model(tr["agri"][ix], tr["geometry"][ix], cot[ix])
                        loss = (((pred_kt * tr["clear"][ix] - tr["ghi"][ix]) / 1000.0) ** 2).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite future-GHI loss")
                        opt.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                        opt.step(); updates += 1; weighted += float(loss.detach()) * len(ix)
                    sched.step()
                    pred = score(model, va, arm, a.batch_size)
                    truth = va["ghi"].detach().cpu().numpy().astype(np.float64)
                    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                    history.append({"epoch": epoch + 1, "train_scaled_mse": weighted / n,
                                    "val_rmse_wm2": rmse, "updates": updates})
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if epoch == 0 or (epoch + 1) % 5 == 0:
                        print(json.dumps({"station": station, "seed": seed, "arm": arm,
                                          **history[-1]}, separators=(",", ":")), flush=True)
                model.load_state_dict(best_state, strict=True); model.eval()
                pred = score(model, va, arm, a.batch_size)
                truth = va["ghi"].detach().cpu().numpy().astype(np.float64)
                np.savez_compressed(run / "validation_predictions.npz", prediction_ghi_wm2=pred,
                                    truth_ghi_wm2=truth,
                                    keys=np.asarray([json.dumps(x, sort_keys=True) for x in va["keys"]]))
                torch.save({"model": best_state, "arm": arm, "station": station, "seed": seed,
                            "best_val_rmse_wm2": best_rmse, "pack_manifest_sha256": report["pack_manifest_sha256"]},
                           run / "best.pt")
                metrics = {"station": station, "seed": seed, "arm": arm, "n_train": n,
                           "n_val": len(truth), "best_val_rmse_wm2": float(np.sqrt(np.mean((pred-truth)**2))),
                           "mae_wm2": float(np.mean(np.abs(pred-truth))),
                           "bias_wm2": float(np.mean(pred-truth)), "updates": updates,
                           "elapsed_seconds": time.monotonic() - started,
                           "checkpoint_sha256": sha(run / "best.pt"),
                           "predictions_sha256": sha(run / "validation_predictions.npz"),
                           "test_payload_opened": False}
                (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                (run / "training_curve.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
                summaries.append(metrics)
                print("ARM_COMPLETE " + json.dumps(metrics), flush=True)
                del model, opt, sched
                torch.cuda.empty_cache()
            del base
    report["state"] = "COMPLETE_FUTURE_GHI_COT_RELIABILITY_SCREENING_PILOT"
    report["runs"] = summaries
    (a.output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FUTURE_GHI_COT_RELIABILITY_COMPLETE", json.dumps({"runs": len(summaries), "test_payload_opened": False}), flush=True)


if __name__ == "__main__":
    main()
