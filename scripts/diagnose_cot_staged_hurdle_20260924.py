"""Validation-only oracle-routing and error-case extraction for staged COT."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_cot_real_agri_staged_hurdle_20260924 import (
    COTUNet, CloudClassifier, ConditionalRegressors, THRESHOLD,
)


def metrics(pred, target, valid):
    err = pred.astype(np.float64) - target.astype(np.float64)
    near = target < THRESHOLD
    cloud = ~near
    thick = target >= 30.0
    station = np.zeros((1, 1, 16, 16), dtype=bool)
    station[:, :, 6:11, 6:11] = True
    groups = {"all": valid, "near_zero_cot_lt0p1": valid & near,
              "cloud_cot_ge0p1": valid & cloud, "thick_cot_ge30": valid & thick,
              "station_5x5": valid & station}
    out = {}
    for name, mask in groups.items():
        values = err[mask]
        out[name] = {"n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(values ** 2))),
                     "mae": float(np.mean(np.abs(values))), "bias": float(np.mean(values))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cases", type=int, default=4)
    args = ap.parse_args()
    root, run, out = args.root, args.run, args.out
    out.mkdir(parents=True, exist_ok=False)
    pack = root / "data/cot_repaired_pack_20260907_v1"
    norm = json.loads((pack / "norm.json").read_text())
    means = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    stds = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    raw = np.load(pack / "x_raw.npy", mmap_mode="r")
    target_all = np.load(pack / "target.npy", mmap_mode="r")
    valid_all = np.load(pack / "mask.npy", mmap_mode="r")
    with (pack / "rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    val_ids = np.asarray([i for i, row in enumerate(rows) if row["split"] == "validation"], dtype=np.int64)
    if len(val_ids) != 6536 or any(rows[i]["split"] != "validation" for i in val_ids):
        raise RuntimeError("validation cohort mismatch or test rows detected")
    target = np.asarray(target_all[val_ids], dtype=np.float32) * np.float32(100.0)
    valid = np.asarray(valid_all[val_ids], dtype=bool)
    base_file = np.load(run / "frozen_R_validation_predictions.npz")
    cand_file = np.load(run / "stage2_conditional_regressors/best_validation_predictions.npz")
    cls_file = np.load(run / "stage1_classifier/best_validation_probabilities.npz")
    for data in (base_file, cand_file, cls_file):
        if not np.array_equal(data["row_indices"], val_ids):
            raise RuntimeError("saved prediction row order differs from verified validation split")
    baseline = np.asarray(base_file["pred_cot"], dtype=np.float32)
    final_saved = np.asarray(cand_file["pred_cot"], dtype=np.float32)
    prob = np.asarray(cls_file["cloud_probability"], dtype=np.float32)
    if not np.allclose(final_saved, cand_file["pred_cot"], rtol=0, atol=0):
        raise RuntimeError("unexpected saved prediction discrepancy")

    # The only trained components replayed here are the two saved 1x1 heads.
    cfg = json.loads((root / "r_frozen_repaired_seed42.json").read_text())
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    torch.set_num_threads(2)
    backbone = COTUNet().to(device).eval()
    backbone.load_state_dict(checkpoint["model"], strict=True)
    classifier = CloudClassifier(cloud_prior=0.5).to(device).eval()
    classifier.load_state_dict(torch.load(run / "stage1_classifier/best.pt", map_location=device,
                                          weights_only=False)["state_dict"], strict=True)
    regressors = ConditionalRegressors(backbone.head).to(device).eval()
    regressors.load_state_dict(torch.load(run / "stage2_conditional_regressors/best.pt",
                                          map_location=device, weights_only=False)["state_dict"], strict=True)
    low_rows, cloud_rows, replay_rows = [], [], []
    with torch.inference_mode():
        for start in range(0, len(val_ids), args.batch):
            ids = val_ids[start:start + args.batch]
            xraw = np.asarray(raw[ids], dtype=np.float32)
            x = (xraw - means) / stds
            x[~np.isfinite(x)] = 0.0
            xt = torch.from_numpy(x).to(device)
            feat = backbone.forward_features(xt)
            pp = torch.sigmoid(classifier(feat).float())
            low, cloud, _, _ = regressors(feat)
            low_rows.append(low.cpu().numpy())
            cloud_rows.append(cloud.cpu().numpy())
            replay_rows.append(pp.cpu().numpy())
    low = np.concatenate(low_rows).astype(np.float32)
    cloud = np.concatenate(cloud_rows).astype(np.float32)
    prob_replay = np.concatenate(replay_rows).astype(np.float32)
    probability_replay_delta = np.abs(prob - prob_replay)
    # Preserve the exact GPU gate decisions saved by the training run. CPU replay
    # is used only to expose the two regression experts for the oracle diagnostic.
    own_route = np.where(prob >= 0.5, cloud, low).astype(np.float32)
    route_replay_delta = np.abs(own_route - final_saved)
    oracle_route = np.where(target >= THRESHOLD, cloud, low).astype(np.float32)
    m = {"frozen_R": metrics(baseline, target, valid),
         "model_classification_route": metrics(own_route, target, valid),
         "CPP_reference_class_route_oracle_diagnostic": metrics(oracle_route, target, valid)}
    m["class_gate"] = {
        "threshold_probability": 0.5,
        "classification_accuracy": float((((prob >= 0.5) == (target >= THRESHOLD)) & valid).sum() / valid.sum()),
        "false_cloud_rate_among_near_zero": float((((prob >= 0.5) & (target < THRESHOLD) & valid).sum()) /
                                                   ((target < THRESHOLD) & valid).sum()),
        "missed_cloud_rate_among_cloud": float((((prob < 0.5) & (target >= THRESHOLD) & valid).sum()) /
                                                ((target >= THRESHOLD) & valid).sum()),
        "valid_pixels": int(valid.sum()),
        "self_vs_oracle_mse_delta": float(np.mean((own_route[valid] - target[valid]) ** 2) -
                                          np.mean((oracle_route[valid] - target[valid]) ** 2)),
        "near_zero_self_vs_oracle_mse_delta": float(np.mean((own_route[valid & (target < THRESHOLD)] - target[valid & (target < THRESHOLD)]) ** 2) -
                                                     np.mean((oracle_route[valid & (target < THRESHOLD)] - target[valid & (target < THRESHOLD)]) ** 2)),
    }

    # Error context: low sun and CPP cloud-edge proximity, measured only on near-zero references.
    val_raw = np.asarray(raw[val_ids], dtype=np.float32)
    cossoz = val_raw[:, 13:14]
    near = (target < THRESHOLD) & valid
    fp = (prob >= 0.5) & near
    day = val_raw[:, 15:16] > 0.5
    m["near_zero_context"] = {
        "day_mask_false_cloud_fraction": float(fp[day & near].sum() / max(1, (day & near).sum())),
        "night_mask_false_cloud_fraction": float(fp[(~day) & near].sum() / max(1, ((~day) & near).sum())),
        "low_sun_cosSOZ_lt0p2_false_cloud_fraction": float(fp[(cossoz < 0.2) & near].sum() / max(1, ((cossoz < 0.2) & near).sum())),
        "higher_sun_cosSOZ_ge0p2_false_cloud_fraction": float(fp[(cossoz >= 0.2) & near].sum() / max(1, ((cossoz >= 0.2) & near).sum())),
    }
    # One-pixel CPP cloud adjacency on 16x16 images; boundary is not wrapped.
    edge_adj = np.zeros_like(near)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            src_y = slice(max(0, -dy), min(16, 16 - dy))
            src_x = slice(max(0, -dx), min(16, 16 - dx))
            dst_y = slice(max(0, dy), min(16, 16 + dy))
            dst_x = slice(max(0, dx), min(16, 16 + dx))
            edge_adj[:, :, dst_y, dst_x] |= (target[:, :, src_y, src_x] >= THRESHOLD)
    m["near_zero_context"]["within_one_pixel_of_CPP_cloud_fraction_false_cloud_rate"] = float(
        fp[near & edge_adj].sum() / max(1, (near & edge_adj).sum()))
    m["near_zero_context"]["farther_than_one_pixel_from_CPP_cloud_false_cloud_rate"] = float(
        fp[near & ~edge_adj].sum() / max(1, (near & ~edge_adj).sum()))
    patch_border = np.zeros((1, 1, 16, 16), dtype=bool)
    patch_border[:, :, 0, :] = True
    patch_border[:, :, -1, :] = True
    patch_border[:, :, :, 0] = True
    patch_border[:, :, :, -1] = True
    m["near_zero_context"]["patch_border_false_cloud_rate"] = float(
        fp[near & patch_border].sum() / max(1, (near & patch_border).sum()))
    m["near_zero_context"]["patch_interior_false_cloud_rate"] = float(
        fp[near & ~patch_border].sum() / max(1, (near & ~patch_border).sum()))
    m["near_zero_context"]["patch_border_high_cot_ge10_rate"] = float(
        ((own_route >= 10.0) & near & patch_border).sum() / max(1, (near & patch_border).sum()))
    m["near_zero_context"]["patch_interior_high_cot_ge10_rate"] = float(
        ((own_route >= 10.0) & near & ~patch_border).sum() / max(1, (near & ~patch_border).sum()))

    # Choose distinct validation rows by the most severe near-zero false-cloud output.
    case_scores = []
    for j, row_id in enumerate(val_ids):
        select = valid[j, 0] & (target[j, 0] < THRESHOLD) & (own_route[j, 0] >= 10.0)
        if select.any():
            loc = np.unravel_index(np.argmax(np.where(select, own_route[j, 0], -1)), (16, 16))
            case_scores.append((float(own_route[j, 0, loc[0], loc[1]]), j, int(loc[0]), int(loc[1])))
    case_scores.sort(reverse=True)
    chosen = case_scores[:args.cases]
    if not chosen:
        raise RuntimeError("no near-zero -> >=10 COT examples found on validation split")
    sel = np.asarray([j for _, j, _, _ in chosen], dtype=np.int64)
    selected_raw = val_raw[sel]
    # Fixed validation-wide 1st/99th percentile stretch for the pseudo RGB channels.
    rgb_order = (2, 1, 0)  # R=C03 NIR, G=C02 red, B=C01 blue.
    rgb_limits = {}
    for ch in sorted(set(rgb_order)):
        sample = val_raw[:, ch].reshape(-1)
        rgb_limits[str(ch)] = [float(np.quantile(sample, 0.01)), float(np.quantile(sample, 0.99))]
    meta = []
    for score, j, yy, xx in chosen:
        row = dict(rows[int(val_ids[j])])
        row.update({"selected_pixel_row": yy, "selected_pixel_col": xx,
                    "selected_false_cloud_prediction_cot": score,
                    "selected_CPP_reference_cot": float(target[j, 0, yy, xx]),
                    "selected_cloud_probability": float(prob[j, 0, yy, xx]),
                    "cosSOZ": float(val_raw[j, 13, yy, xx]),
                    "day_mask": float(val_raw[j, 15, yy, xx])})
        meta.append(row)
    np.savez_compressed(out / "false_cloud_cases.npz", row_indices=val_ids[sel], raw_agri=selected_raw,
                        cpp_cot=target[sel], valid_mask=valid[sel], cloud_probability=prob[sel],
                        baseline_cot=baseline[sel], low_expert_cot=low[sel], cloud_expert_cot=cloud[sel],
                        model_route_cot=own_route[sel], oracle_route_cot=oracle_route[sel],
                        rgb_channel_order=np.asarray(rgb_order, dtype=np.int8))
    summary = {"state": "COMPLETE_VALIDATION_DIAGNOSTIC", "test_used": False,
               "input_domain": "observed real AGRI validation rows only",
               "cpu_vs_saved_gpu_replay": {"classifier_probability_max_abs_delta": float(probability_replay_delta.max()),
                                            "classifier_probability_p99_abs_delta": float(np.quantile(probability_replay_delta, .99)),
                                            "expert_route_prediction_max_abs_delta_cot": float(route_replay_delta.max()),
                                            "expert_route_prediction_p99_abs_delta_cot": float(np.quantile(route_replay_delta, .99)),
                                            "routing_uses": "saved GPU probabilities; expert values are CPU replays"},
               "route_comparison": m, "rgb_channel_limits_1_99pct": rgb_limits,
               "rgb_definition": "pseudo RGB with R=C03, G=C02, B=C01; fixed validation-wide 1st-99th percentile stretch",
               "case_selection": "distinct validation rows with CPP COT<0.1 and model-routed COT>=10; rank by largest predicted COT",
               "cases": meta, "validation_rows": int(len(val_ids)), "test_payloads_read": 0}
    (out / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": summary["state"], "cases": len(meta),
                      "oracle": m["CPP_reference_class_route_oracle_diagnostic"],
                      "context": m["near_zero_context"]}, indent=2))


if __name__ == "__main__":
    main()
