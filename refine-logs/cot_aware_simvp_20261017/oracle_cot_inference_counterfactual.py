#!/usr/bin/env python3
"""Swap only validation future-COT features under frozen forecast-trained GHI heads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from train_future_ghi_heads import IntervalHead, load_station, score


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def date_bjt(k: dict[str, str]) -> str:
    dt = datetime.strptime(k["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
    return dt.date().isoformat()


def rmse(pred: np.ndarray, truth: np.ndarray, weight: np.ndarray | None = None) -> float:
    err2 = (pred - truth) ** 2
    return float(np.sqrt(np.mean(err2) if weight is None else np.sum(weight * err2) / np.sum(weight)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-pack", type=Path, required=True)
    p.add_argument("--oracle-pack", type=Path, required=True)
    p.add_argument("--forecast-head-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261020)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference audit")
    fm = json.loads((a.forecast_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    om = json.loads((a.oracle_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if fm.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK" or fm.get("train_forecast_oof") is not True or fm.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("forecast pack lacks five-fold OOF provenance")
    if om.get("state") != "COMPLETE_ORACLE_FUTURE_COT_GHI_PACK" or om.get("oracle_only") is not True or om.get("deployable") is not False:
        raise ValueError("oracle pack provenance is invalid")
    if any(m.get(k) is not False for m in (fm, om) for k in ("test_payload_opened", "buffer_payload_opened")):
        raise ValueError("test/buffer closure gate failed")
    for split in ("val",):
        for name in ("agri_raw.npy", "geometry.npy", "ghi.npy", "clear.npy", "keys.csv"):
            if sha(a.forecast_pack / split / name) != sha(a.oracle_pack / split / name):
                raise ValueError(f"shared validation input differs: {split}/{name}")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    per_run: list[dict[str, object]] = []
    saved_predictions: list[dict[str, object]] = []
    all_dates: set[str] = set()
    for station in ("sili", "zhujia"):
        base_data = load_station(a.forecast_pack, "val", station, device, fm)
        oracle_data = load_station(a.oracle_pack, "val", station, device, fm)
        if base_data["keys"] != oracle_data["keys"]:
            raise ValueError(f"validation row order differs for {station}")
        if not torch.equal(base_data["agri"], oracle_data["agri"]) or not torch.equal(base_data["geometry"], oracle_data["geometry"]):
            raise ValueError(f"non-COT features changed for {station}")
        if not torch.equal(base_data["ghi"], oracle_data["ghi"]) or not torch.equal(base_data["clear"], oracle_data["clear"]):
            raise ValueError(f"labels changed for {station}")
        hmask = base_data["history_cot_mask"]
        if not torch.equal(base_data["cot"] * hmask, oracle_data["cot"] * hmask):
            raise ValueError(f"observed-history COT changed for {station}")
        # Keep every forecast-pack tensor and replace only normalized future COT.
        # Oracle COT is normalized with the forecast pack's train-only mean/std.
        cf_data = dict(base_data)
        cf_data["cot"] = base_data["cot"] * hmask + oracle_data["cot"] * (1.0 - hmask)
        keys = base_data["keys"]
        dates = np.asarray([date_bjt(k) for k in keys])
        leads = np.asarray([int(k["lead_minutes"]) for k in keys], dtype=np.int16)
        all_dates.update(dates.tolist())
        truth = base_data["ghi"].detach().cpu().numpy().astype(np.float64)
        val_pred_path = a.forecast_head_root / station
        for seed in a.seeds:
            run = val_pred_path / f"seed{seed}" / "AC"
            ckpt_path = run / "best.pt"
            saved_path = run / "validation_predictions.npz"
            if not ckpt_path.is_file() or not saved_path.is_file():
                raise FileNotFoundError(f"missing forecast-trained AC artifacts: {run}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("arm") != "AC" or ckpt.get("station") != station or int(ckpt.get("seed")) != seed:
                raise ValueError(f"checkpoint identity mismatch: {ckpt_path}")
            if ckpt.get("pack_manifest_sha256") != sha(a.forecast_pack / "pack_manifest.json"):
                raise ValueError(f"checkpoint was trained from another forecast pack: {ckpt_path}")
            model = IntervalHead().to(device)
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()
            base_pred = score(model, base_data, "AC", a.batch_size)
            counterfactual = score(model, cf_data, "AC", a.batch_size)
            with np.load(saved_path, allow_pickle=False) as z:
                saved_pred = np.asarray(z["prediction_ghi_wm2"], dtype=np.float64)
                saved_truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
                saved_keys = z["keys"].astype(str)
            expected_keys = np.asarray([json.dumps(k, sort_keys=True) for k in keys])
            if not np.array_equal(saved_keys, expected_keys) or not np.array_equal(saved_truth, truth):
                raise ValueError(f"saved validation keys/truth mismatch: {saved_path}")
            replay_abs = np.abs(base_pred - saved_pred)
            maxdiff = float(np.max(replay_abs))
            rmsdiff = float(np.sqrt(np.mean(replay_abs ** 2)))
            replay_rmse_delta = rmse(base_pred, truth) - rmse(saved_pred, truth)
            # Different A800 instances can choose numerically different kernels.
            # Keys/truth/checkpoint/pack hashes remain exact; require negligible
            # aggregate-metric drift and bound any per-row numerical deviation.
            if maxdiff > 0.25 or abs(replay_rmse_delta) > 0.01:
                raise ValueError(f"forecast-COT inference replay outside numeric tolerance: {saved_path}; max_abs_diff={maxdiff:.8g}; rms_diff={rmsdiff:.8g}; rmse_delta={replay_rmse_delta:.8g}; batch={a.batch_size}; cudnn_benchmark={torch.backends.cudnn.benchmark}")
            per_run.append({"station": station, "seed": seed, "n": len(truth),
                            "forecast_cot_rmse_wm2": rmse(base_pred, truth),
                            "oracle_cot_substitution_rmse_wm2": rmse(counterfactual, truth),
                            "delta_oracle_substitution_minus_forecast_cot_wm2": rmse(counterfactual, truth) - rmse(base_pred, truth),
                            "first_hour_forecast_rmse_wm2": rmse(base_pred[leads <= 60], truth[leads <= 60]),
                            "first_hour_oracle_substitution_rmse_wm2": rmse(counterfactual[leads <= 60], truth[leads <= 60]),
                            "late_forecast_rmse_wm2": rmse(base_pred[leads >= 120], truth[leads >= 120]),
                            "late_oracle_substitution_rmse_wm2": rmse(counterfactual[leads >= 120], truth[leads >= 120]),
                            "saved_prediction_replay_max_abs_diff_wm2": maxdiff,
                            "saved_prediction_replay_rms_diff_wm2": rmsdiff,
                            "saved_prediction_replay_rmse_delta_wm2": replay_rmse_delta,
                            "checkpoint_sha256": sha(ckpt_path), "saved_prediction_sha256": sha(saved_path)})
            saved_predictions.append({"station": station, "seed": seed, "keys": keys,
                                     "dates": dates, "leads": leads, "truth": truth,
                                     "forecast": base_pred, "counterfactual": counterfactual})
            del model, ckpt
            torch.cuda.empty_cache()
        # reload station inputs for the next station; clear GPU arrays
        del base_data, oracle_data, cf_data
        torch.cuda.empty_cache()

    date_list = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    date_index_by_run = [np.asarray([date_to_idx[d] for d in r["dates"]], dtype=np.int32) for r in saved_predictions]
    rng = np.random.default_rng(a.bootstrap_seed)
    boot = np.empty(a.bootstrap, dtype=np.float64)
    for b in range(a.bootstrap):
        counts = rng.multinomial(len(date_list), np.full(len(date_list), 1.0 / len(date_list)))
        vals = []
        for r, di in zip(saved_predictions, date_index_by_run):
            w = counts[di].astype(np.float64)
            if w.sum() > 0:
                vals.append(rmse(r["counterfactual"], r["truth"], w) - rmse(r["forecast"], r["truth"], w))
        boot[b] = float(np.mean(vals))
    point = float(np.mean([r["delta_oracle_substitution_minus_forecast_cot_wm2"] for r in per_run]))
    ci = np.quantile(boot, [0.025, 0.975]).tolist()

    a.output.mkdir(parents=True)
    pred_dir = a.output / "predictions"
    pred_dir.mkdir()
    for r in saved_predictions:
        name = f"{r['station']}_seed{r['seed']}.npz"
        np.savez_compressed(pred_dir / name,
                            keys=np.asarray([json.dumps(k, sort_keys=True) for k in r["keys"]]),
                            forecast=r["forecast"], oracle_cot_substitution=r["counterfactual"], truth=r["truth"])
    result = {"state": "COMPLETE_FROZEN_FORECAST_HEAD_ORACLE_COT_INFERENCE_COUNTERFACTUAL",
              "scope": "validation-only inference substitution; the GHI head is trained on forecast COT and frozen; replace only future COT values with observed-future-AGRI→frozen-R oracle values",
              "primary_metric": "equal mean over station×seed of paired GHI RMSE difference; negative favors oracle-COT substitution",
              "delta_oracle_substitution_minus_forecast_cot_rmse_wm2": point,
              "target_bjt_date_block_bootstrap_95pct_ci_wm2": ci,
              "target_bjt_dates": date_list, "stations": ["sili", "zhujia"], "seeds": list(a.seeds),
              "rows_per_station": {s: int(len(next(r["keys"] for r in saved_predictions if r["station"] == s))) for s in ("sili", "zhujia")},
              "runs": per_run,
              "forecast_pack_manifest_sha256": sha(a.forecast_pack / "pack_manifest.json"),
              "oracle_pack_manifest_sha256": sha(a.oracle_pack / "pack_manifest.json"),
              "source_script_sha256": sha(Path(__file__)),
              "forecast_head_root": str(a.forecast_head_root),
              "forecast_AC_head_source_sha256": {f"{r['station']}/seed{r['seed']}": r["checkpoint_sha256"] for r in per_run},
              "oracle_only": True, "deployable": False, "test_payload_opened": False, "buffer_payload_opened": False}
    (a.output / "counterfactual_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state", "delta_oracle_substitution_minus_forecast_cot_rmse_wm2", "target_bjt_date_block_bootstrap_95pct_ci_wm2", "runs", "test_payload_opened")}, indent=2))


if __name__ == "__main__":
    main()
