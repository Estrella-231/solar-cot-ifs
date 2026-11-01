#!/usr/bin/env python3
"""Descriptive validation audit by the forecast-COT regime at the station pixel.

This diagnostic uses only validation predictions and validation forecast-COT
inputs. It is not a model-selection score and does not use CPP/true future AGRI.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ARMS = ("A", "AH", "AC")
STATIONS = ("sili", "zhujia")


def key_from_row(row):
    return (row["station"], row["init_time_utc"], row["target_time_utc"],
            int(row["lead_minutes"]), int(row["ghi_row_index"]))


def load_prediction(path: Path):
    with np.load(path, allow_pickle=False) as z:
        pred = np.asarray(z["prediction_ghi_wm2"], dtype=np.float64)
        truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
        keys = [json.loads(str(x)) for x in z["keys"]]
    canon = [key_from_row(k) for k in keys]
    if pred.shape != truth.shape or pred.ndim != 1 or len(canon) != len(pred):
        raise ValueError(f"invalid saved predictions: {path}")
    if len(set(canon)) != len(canon) or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError(f"duplicate keys or nonfinite predictions: {path}")
    return {"keys": canon, "pred": pred, "truth": truth}


def csi_bin(cot: np.ndarray) -> np.ndarray:
    return np.select((cot < 1.0, cot < 10.0, cot < 30.0),
                     ("<1", "1-10", "10-30"), default=">=30")


def lead_bin(lead: np.ndarray) -> np.ndarray:
    return np.select((lead <= 60, lead <= 120),
                     ("15-60", "75-120"), default="135-240")


def summarize(err: np.ndarray):
    return {"rmse_wm2": float(np.sqrt(np.mean(np.square(err)))),
            "mae_wm2": float(np.mean(np.abs(err))),
            "bias_wm2": float(np.mean(err))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("future-GHI pack is not complete")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer exclusion is not certified")
    val_dir = a.pack / "val"
    rows = list(csv.DictReader((val_dir / "keys.csv").open(newline="", encoding="utf-8")))
    cot = np.load(val_dir / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if cot.shape != (len(rows), 3, 1, 3, 3) or cot.dtype != np.float32:
        raise ValueError("unexpected validation COT input shape/dtype")
    cot_center = np.expm1(np.asarray(cot[:, 2, 0, 1, 1], dtype=np.float64))
    if not np.isfinite(cot_center).all() or (cot_center < 0).any():
        raise ValueError("invalid predicted station-pixel COT")
    cot_by_key = {}
    for i, row in enumerate(rows):
        key = key_from_row(row)
        if key in cot_by_key:
            raise ValueError(f"duplicate validation pack key: {key}")
        cot_by_key[key] = cot_center[i]

    output_rows = []
    for station in STATIONS:
        for seed in a.seeds:
            pair = {arm: load_prediction(a.run_root / station / f"seed{seed}" / arm /
                                         "validation_predictions.npz") for arm in ARMS}
            base = pair["A"]
            for arm in ("AH", "AC"):
                if pair[arm]["keys"] != base["keys"] or not np.array_equal(pair[arm]["truth"], base["truth"]):
                    raise ValueError(f"unpaired prediction keys/truth: {station}/seed{seed}/{arm}")
            if any(k[0] != station for k in base["keys"]):
                raise ValueError(f"station mismatch in saved prediction keys: {station}/seed{seed}")
            cot_values = np.asarray([cot_by_key[k] for k in base["keys"]], dtype=np.float64)
            leads = np.asarray([k[3] for k in base["keys"]], dtype=np.int16)
            bins = csi_bin(cot_values)
            lead_groups = lead_bin(leads)
            target_days = np.asarray([
                (datetime.strptime(k[2], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d")
                for k in base["keys"]
            ])
            for cot_group in ("<1", "1-10", "10-30", ">=30"):
                for lead_group in ("15-60", "75-120", "135-240"):
                    mask = (bins == cot_group) & (lead_groups == lead_group)
                    if not mask.any():
                        continue
                    metrics = {arm: summarize(pair[arm]["pred"][mask] - base["truth"][mask]) for arm in ARMS}
                    output_rows.append({
                        "station": station, "seed": seed, "forecast_cot_bin": cot_group,
                        "lead_group_minutes": lead_group, "n_instances": int(mask.sum()),
                        "n_target_bjt_dates": int(len(set(target_days[mask]))),
                        **{f"{arm}_{metric}": value for arm in ARMS
                           for metric, value in metrics[arm].items()},
                        "delta_AC_minus_A_rmse_wm2": metrics["AC"]["rmse_wm2"] - metrics["A"]["rmse_wm2"],
                        "delta_AC_minus_AH_rmse_wm2": metrics["AC"]["rmse_wm2"] - metrics["AH"]["rmse_wm2"],
                        "delta_AH_minus_A_rmse_wm2": metrics["AH"]["rmse_wm2"] - metrics["A"]["rmse_wm2"],
                    })

    if not output_rows:
        raise ValueError("no matched validation rows found for the requested COT/lead strata")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "state": "COMPLETE_DESCRIPTIVE_FUTURE_GHI_COT_STRATA_AUDIT",
        "scope": "validation-only descriptive diagnostic; not used for model selection",
        "stratifier": "predicted frozen-R COT at station center pixel for target forecast time",
        "cot_bins": {"<1": "near-zero", "1-10": "thin", "10-30": "moderate", ">=30": "thick predicted COT"},
        "lead_groups_minutes": ["15-60", "75-120", "135-240"],
        "arms": {"A": "no COT", "AH": "observed-history COT only", "AC": "history plus forecast-COT"},
        "interpretation": "Negative deltas favor the first named COT arm. Bin metrics are descriptive; overlapping forecast instances are not independent and no pixel/sample-level significance is claimed.",
        "seed_ids": a.seeds, "test_payload_opened": False, "buffer_payload_opened": False,
        "rows": output_rows,
    }
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = a.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    print(json.dumps({"state": report["state"], "strata_rows": len(output_rows),
                      "test_payload_opened": False, "buffer_payload_opened": False}))


if __name__ == "__main__":
    main()
