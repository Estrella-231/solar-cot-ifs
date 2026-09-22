"""Paired fixed-head GHI evaluation for FP32 real-AGRI historic COT retrieval.

The trained Hunan GHI heads, image/metadata, future forecast COT, labels,
causal mask, and original train-only COT normalization are held fixed. Only
the historic COT bank changes: original R FP32 versus continued R FP32.
Legacy BF16 historic COT is replayed as an inference compatibility witness.
No training, test data, or future real AGRI/CPP is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from causal_cot_trajectory_solarresnet_v5 import CausalCOTTrajectorySolarResNetV5
from train_causal_cot_trajectory_solarresnet_v5 import evaluate


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(root, paired, output, batch):
    root, paired, output = root.resolve(), paired.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite fixed-head GHI swap result")
    if not torch.cuda.is_available() or not (1 <= batch <= 1536):
        raise RuntimeError("GPU unavailable or unsafe batch")
    pair = json.loads((paired / "complete.json").read_text())
    if pair["state"] != "COMPLETE_PAIRED_REAL_AGRI_HISTORY_COT" or pair["test_used"] is not False:
        raise RuntimeError("paired FP32 history COT incomplete")
    head_dir = root / "runs/causal_cot_trajectory_solarresnet_v5_b1_seed42_iofix_20260921"
    head = json.loads((head_dir / "complete.json").read_text())
    if head["state"] != "COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY" or head["test_used"] is not False:
        raise RuntimeError("fixed GHI head run incomplete")
    original_history_dir = root / "data/history_cot_trajectory_trainval_20260918_v1"
    forecast_dir = root / "data/forecast_cot_trajectory_trainval_20260918_v1"
    for directory, key in ((original_history_dir, "history_sidecar_audit_sha256"),
                           (forecast_dir, "forecast_sidecar_audit_sha256")):
        if sha(directory / "audit.json") != head[key]:
            raise RuntimeError("fixed sidecar audit drift: " + key)
    image_dir = root / "data/solarresnet_v5_input_trainval_20260918_v1"
    if sha(image_dir / "audit.json") != head["input_sidecar_audit_sha256"]:
        raise RuntimeError("fixed GHI image sidecar audit drift")
    base = root / "data/head_pack_trainval_20260908_v1"
    if sha(base / "audit.json") != head["base_pack_audit_sha256"]:
        raise RuntimeError("fixed GHI labels/rows audit drift")
    image = np.load(image_dir / "image.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(image_dir / "metadata.npy", mmap_mode="r", allow_pickle=False)
    if image.shape != (637902, 16, 16, 16) or metadata.shape != (637902, 5):
        raise RuntimeError("GHI image/metadata shape drift")
    data = {key: torch.from_numpy(np.load(base / f"{key}.npy", allow_pickle=False)).cuda()
            for key in ("ghi", "clear", "station", "lead", "sequence", "split")}
    if int((data["split"] == 1).sum()) != 99849 or not bool((data["clear"] > 0).all()):
        raise RuntimeError("validation cohort or clear-sky contract drift")
    legacy_path = original_history_dir / "val_history_cot_log1p.npy"
    forecast_path = forecast_dir / "val_forecast_cot_log1p.npy"
    legacy_audit = json.loads((original_history_dir / "audit.json").read_text())
    forecast_audit = json.loads((forecast_dir / "audit.json").read_text())
    if sha(legacy_path) != legacy_audit["splits"]["val"]["sha256"] or sha(forecast_path) != forecast_audit["splits"]["val"]["sha256"]:
        raise RuntimeError("fixed history/forecast bank hash drift")
    history = {"legacy_bf16": torch.from_numpy(np.load(legacy_path, allow_pickle=False)).cuda()}
    for scenario in ("original_fp32", "continued_fp32"):
        path = paired / scenario / "val_history_cot_log1p.npy"
        if sha(path) != pair["arms"][scenario]["sha256"]:
            raise RuntimeError("paired FP32 history bank hash drift")
        history[scenario] = torch.from_numpy(np.load(path, allow_pickle=False)).cuda()
    forecast = torch.from_numpy(np.load(forecast_path, allow_pickle=False)).cuda()
    for scenario, bank in history.items():
        if bank.shape != (4258, 2, 8, 1, 16, 16) or not torch.isfinite(bank).all():
            raise RuntimeError("invalid history bank: " + scenario)
    if forecast.shape != (4258, 2, 16, 1, 16, 16) or not torch.isfinite(forecast).all():
        raise RuntimeError("invalid forecast bank")
    norm = {key: head["cot_normalization"][key] for key in ("mean", "std")}
    if not np.isfinite(list(norm.values())).all() or norm["std"] <= 0:
        raise RuntimeError("invalid fixed train-only COT normalization")
    output.mkdir(parents=True)
    report = {"state": "RUNNING_FIXED_GHI_HEAD_REAL_AGRI_R_SWAP", "test_used": False,
              "cohort": "same frozen validation rows, GHI/clear labels, station head, image, metadata, forecast COT, normalization and causal masks",
              "only_mutation": "observed historic AGRI retrieved COT: original R FP32 vs continued R FP32",
              "history_pair_sha256": sha(paired / "complete.json"),
              "fixed_head_complete_sha256": sha(head_dir / "complete.json"),
              "script_sha256": sha(__file__), "batch": batch, "runs": {}}
    for station_id, station_name in ((0, "Sili"), (1, "Zhujia")):
        ids = torch.where((data["split"] == 1) & (data["station"] == station_id))[0]
        report["runs"][station_name] = {}
        for group in ("AC_H", "AC_ST"):
            checkpoint = head_dir / f"{station_name}_{group}_best.pt"
            if sha(checkpoint) != head["runs"][station_name][group]["checkpoint_sha256"]:
                raise RuntimeError("fixed station head checkpoint drift")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["group"] != group or saved["station"] != station_name:
                raise RuntimeError("wrong fixed head checkpoint identity")
            model = CausalCOTTrajectorySolarResNetV5().cuda().eval().requires_grad_(False)
            model.load_state_dict(saved["model"], strict=True)
            report["runs"][station_name][group] = {}
            for scenario, bank in history.items():
                banks = {"val": {"history": bank, "forecast": forecast}}
                metrics, kt = evaluate(model, data, banks, image, metadata, ids, group, batch, norm, keep=True)
                pred_ghi = kt.astype(np.float64) * data["clear"][ids].cpu().numpy().astype(np.float64)
                true_ghi = data["ghi"][ids].cpu().numpy().astype(np.float64)
                rmse = float(np.sqrt(np.mean((pred_ghi - true_ghi) ** 2)))
                if abs(metrics["ghi_rmse"] - rmse) > 1e-4:
                    raise RuntimeError("NumPy/Torch GHI metric mismatch")
                result_path = output / f"{station_name}_{group}_{scenario}_validation_predictions.npz"
                np.savez(result_path, pack_row=ids.cpu().numpy(), pred_kt=kt,
                         observed_ghi=true_ghi,
                         clear_sky_ghi=data["clear"][ids].cpu().numpy(),
                         station=data["station"][ids].cpu().numpy(), lead=data["lead"][ids].cpu().numpy())
                report["runs"][station_name][group][scenario] = {**metrics,
                    "prediction_sha256": sha(result_path)}
                if scenario == "legacy_bf16":
                    with np.load(head_dir / f"{station_name}_{group}_validation_predictions.npz", allow_pickle=False) as prior:
                        if not np.array_equal(prior["pack_row"], ids.cpu().numpy()) or not np.array_equal(prior["observed_ghi"], true_ghi):
                            raise RuntimeError("fixed-head original validation cohort drift")
                        delta = np.abs(prior["pred_kt"].astype(np.float64) - kt.astype(np.float64))
                        if float(delta.max()) > 0.005:
                            raise RuntimeError("fixed-head legacy replay numerical mismatch")
                        report["runs"][station_name][group][scenario]["prior_pred_kt_max_abs_difference"] = float(delta.max())
                print(station_name, group, scenario, json.dumps(metrics), flush=True)
                (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            del model
            torch.cuda.empty_cache()
    for station_name, by_group in report["runs"].items():
        for group, scenarios in by_group.items():
            scenarios["continued_minus_original_fp32_rmse"] = scenarios["continued_fp32"]["ghi_rmse"] - scenarios["original_fp32"]["ghi_rmse"]
    report["state"] = "COMPLETE_FIXED_GHI_HEAD_REAL_AGRI_R_SWAP"
    (output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paired-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    main(args.root, args.paired_history, args.output, args.batch)
