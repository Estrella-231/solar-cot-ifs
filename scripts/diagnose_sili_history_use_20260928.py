"""Read-only validation: did the learned COT head use older historical maps?"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_sili_history_repeat_control_20260928 import HistoryControlNet, history_input
from train_sili_transport_thickness_pilot_20260927 import prepare


def new_acc():
    return {"n": 0, "sq": 0.0, "abs": 0.0, "bias": 0.0}


def add_error(acc, pred, truth, mask):
    err = (pred - truth)[mask]
    acc["n"] += int(err.numel())
    acc["sq"] += float(err.square().sum())
    acc["abs"] += float(err.abs().sum())
    acc["bias"] += float(err.sum())


def finish_error(acc):
    n = acc["n"]
    return {"n": n, "rmse": (acc["sq"] / n) ** .5 if n else None,
            "mae": acc["abs"] / n if n else None,
            "bias": acc["bias"] / n if n else None}


def new_change():
    return {"n": 0, "sum_abs": 0.0, "sum_signed": 0.0,
            "gt_1": 0, "gt_5": 0, "max_abs": 0.0}


def add_change(acc, original, alternate, mask):
    diff = (original - alternate)[mask]
    absolute = diff.abs()
    acc["n"] += int(diff.numel())
    acc["sum_abs"] += float(absolute.sum())
    acc["sum_signed"] += float(diff.sum())
    acc["gt_1"] += int((absolute > 1).sum())
    acc["gt_5"] += int((absolute > 5).sum())
    acc["max_abs"] = max(acc["max_abs"], float(absolute.max()) if absolute.numel() else 0.0)


def finish_change(acc):
    n = acc["n"]
    return {"n": n, "mean_abs_cot_change": acc["sum_abs"] / n if n else None,
            "mean_signed_cot_change": acc["sum_signed"] / n if n else None,
            "fraction_abs_change_gt_1": acc["gt_1"] / n if n else None,
            "fraction_abs_change_gt_5": acc["gt_5"] / n if n else None,
            "max_abs_cot_change": acc["max_abs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.limit <= 512:
        raise ValueError("limit must be within the fixed validation cohort")
    summary = json.loads(args.training_summary.read_text())
    if summary["state"] != "COMPLETE_SILI_FULL_VS_REPEAT_LAST_HISTORY_PILOT" or summary["test_used"] is not False:
        raise RuntimeError("Unexpected training summary")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("update") != summary["full_history"]["best_learned_update_by_patch_thick_rmse"]:
        raise RuntimeError("Checkpoint/update mismatch")
    model = HistoryControlNet().to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    weights = model.inp.weight.detach().cpu()
    norms = [float(torch.linalg.vector_norm(weights[:, i])) for i in range(15)]
    data = prepare(args.bank, "val", np.arange(args.limit), "cpu")
    error = {scope: {regime: {name: new_acc() for name in ("full", "replace_history", "transport", "simvp_corrected")}
                     for regime in ("all", "thick")}
             for scope in ("patch", "station")}
    change = {scope: {regime: new_change() for regime in ("all", "thick")}
              for scope in ("patch", "station")}
    history_diff = {"n": 0, "sum_abs": 0.0, "station_n": 0, "station_sum_abs": 0.0}
    with torch.inference_mode():
        for start in range(0, args.limit, 8):
            sl = slice(start, min(start + 8, args.limit))
            h, f, y, mask, motion, adv, support = (data[k][sl].to(args.device) for k in
                                                  ("h", "f", "y", "mask", "motion", "adv", "support"))
            full = torch.expm1(model(history_input(h, f, adv, motion, "full_history"), adv)[:, 0])
            replaced = torch.expm1(model(history_input(h, f, adv, motion, "repeat_last"), adv)[:, 0])
            truth = torch.expm1(y)
            fields = {"full": full, "replace_history": replaced,
                      "transport": torch.expm1(adv[:, 0]),
                      "simvp_corrected": torch.expm1(f[:, :, 0])}
            if any(z.shape != truth.shape for z in fields.values()):
                raise RuntimeError("Forecast fields have different lead/space shapes")
            valid = mask & support[:, 0]
            old = torch.expm1(h[:, :6, 0])
            latest = torch.expm1(h[:, -1:, 0])
            hd = (old - latest).abs()
            history_diff["n"] += hd.numel()
            history_diff["sum_abs"] += float(hd.sum())
            history_diff["station_n"] += hd[:, :, 32, 32].numel()
            history_diff["station_sum_abs"] += float(hd[:, :, 32, 32].sum())
            for scope in ("patch", "station"):
                if scope == "patch":
                    tar, val, sub = truth, valid, fields
                else:
                    tar, val = truth[:, :, 32, 32], valid[:, :, 32, 32]
                    sub = {k: z[:, :, 32, 32] for k, z in fields.items()}
                for regime in ("all", "thick"):
                    use = val if regime == "all" else val & (tar >= 30)
                    for name, prediction in sub.items():
                        add_error(error[scope][regime][name], prediction, tar, use)
                    add_change(change[scope][regime], sub["full"], sub["replace_history"], use)
    result = {"state": "COMPLETE_SILI_HISTORY_USE_COUNTERFACTUAL", "test_used": False,
              "validation_sequences": args.limit,
              "checkpoint_update": checkpoint["update"],
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "bank_manifest_sha256": hashlib.sha256((args.bank / "complete.json").read_bytes()).hexdigest(),
              "comparison": "same learned full-history checkpoint; replace eight history slots with last map; preserve original last-minus-previous trend, forecast, transport, motion, lead, targets and masks",
              "input_conv_l2_by_channel": norms,
              "input_conv_channels": ["transport", "forecast", "history_oldest", "history_2", "history_3", "history_4", "history_5", "history_6", "history_penultimate", "history_latest", "trend", "forecast_minus_transport", "lead", "motion_y", "motion_x"],
              "historical_input_difference": {"older_six_vs_last_patch_mean_abs_cot": history_diff["sum_abs"] / history_diff["n"],
                                              "older_six_vs_last_station_mean_abs_cot": history_diff["station_sum_abs"] / history_diff["station_n"]},
              "errors": {scope: {regime: {name: finish_error(acc) for name, acc in methods.items()}
                                 for regime, methods in regimes.items()} for scope, regimes in error.items()},
              "prediction_change": {scope: {regime: finish_change(acc) for regime, acc in regimes.items()}
                                    for scope, regimes in change.items()},
              "simvp_correction": "simvp_corrected uses all 16 forecast maps; original training summary's simvp diagnostic repeated the first forecast map; transport and learned comparisons are unaffected"}
    if args.limit == 512:
        stored = summary["full_history"]["best_learned_metrics"]
        for scope in ("patch", "station"):
            for regime, key in (("all", "rmse"), ("thick", "thick_rmse")):
                actual = result["errors"][scope][regime]["full"]["rmse"]
                expected = stored[scope]["learned"][key]
                if abs(actual - expected) > 1e-4:
                    raise RuntimeError(f"Saved metric mismatch {scope}/{regime}: {actual} != {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("SILI_HISTORY_USE_COUNTERFACTUAL_COMPLETE", json.dumps({
        "sequences": args.limit,
        "patch_thick": result["errors"]["patch"]["thick"],
        "patch_change": result["prediction_change"]["patch"],
        "station_change": result["prediction_change"]["station"]}), flush=True)


if __name__ == "__main__":
    main()
