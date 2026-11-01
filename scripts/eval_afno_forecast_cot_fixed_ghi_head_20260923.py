"""Paired same-head GHI test: replace only frozen-SimVP forecast COT by AFNO COT.

For each site, the already trained AC_F and AC_ST checkpoints, COT history,
images, metadata, labels, masks and train-only normalization remain frozen.
The AFNO/SimVP forecast-COT banks are compared on the exact same val rows.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from causal_cot_trajectory_solarresnet_v5 import CausalCOTTrajectorySolarResNetV5
from train_causal_cot_trajectory_solarresnet_v5 import evaluate


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def day_bootstrap(truth, simvp, afno, days, *, repetitions=2000, seed=42):
    """Paired init-day block bootstrap of AFNO-minus-SimVP RMSE difference."""
    unique = np.unique(days)
    simvp_sse, afno_sse, counts = [], [], []
    for day in unique:
        selected = days == day
        simvp_sse.append(np.square(simvp[selected] - truth[selected]).sum(dtype=np.float64))
        afno_sse.append(np.square(afno[selected] - truth[selected]).sum(dtype=np.float64))
        counts.append(int(selected.sum()))
    simvp_sse = np.asarray(simvp_sse, dtype=np.float64)
    afno_sse = np.asarray(afno_sse, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    n = counts[draws].sum(axis=1)
    simvp_rmse = np.sqrt(simvp_sse[draws].sum(axis=1) / n)
    afno_rmse = np.sqrt(afno_sse[draws].sum(axis=1) / n)
    difference = afno_rmse - simvp_rmse
    return {
        "unit": "BJT initialization day; all same-day forecast leads resampled together",
        "days": int(len(unique)), "repetitions": repetitions, "seed": seed,
        "delta_AFNO_minus_SimVP_95pct_Wm2": np.quantile(difference, [0.025, 0.975]).tolist(),
        "bootstrap_probability_AFNO_RMSE_lower": float(np.mean(difference < 0)),
    }


def main(root, afno_bank_dir, output):
    root, afno_bank_dir, output = root.resolve(), afno_bank_dir.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite paired fixed-head output")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output.mkdir(parents=True)

    head_dir = root / "runs/causal_cot_trajectory_solarresnet_v5_b1_seed42_iofix_20260921"
    head_complete_path = head_dir / "complete.json"
    head = json.loads(head_complete_path.read_text())
    if head.get("state") != "COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY" or head.get("test_used") is not False:
        raise RuntimeError("fixed GHI head is incomplete or test contaminated")
    afno_complete_path = afno_bank_dir / "complete.json"
    afno_contract = json.loads(afno_complete_path.read_text())
    if (afno_contract.get("state") != "COMPLETE_AFNO_FORECAST_COT_VALIDATION" or
            afno_contract.get("test_used") is not False or
            afno_contract.get("future_AGRI_read") is not False or
            afno_contract.get("future_CPP_read") is not False):
        raise RuntimeError("invalid AFNO COT sidecar contract")
    afno_path = afno_bank_dir / "forecast_cot_log1p.npy"
    if sha(afno_path) != afno_contract["forecast_cot_sha256"]:
        raise RuntimeError("AFNO COT bank hash mismatch")

    original_forecast_dir = root / "data/forecast_cot_trajectory_trainval_20260918_v1"
    history_dir = root / "data/history_cot_trajectory_trainval_20260918_v1"
    image_dir = root / "data/solarresnet_v5_input_trainval_20260918_v1"
    base = root / "data/head_pack_trainval_20260908_v1"
    for directory, key in ((original_forecast_dir, "forecast_sidecar_audit_sha256"),
                           (history_dir, "history_sidecar_audit_sha256"),
                           (image_dir, "input_sidecar_audit_sha256"),
                           (base, "base_pack_audit_sha256")):
        audit = directory / ("audit.json" if directory != base else "audit.json")
        if sha(audit) != head[key]:
            raise RuntimeError("fixed head input audit drift: " + key)
    original_forecast_audit = json.loads((original_forecast_dir / "audit.json").read_text())
    history_audit = json.loads((history_dir / "audit.json").read_text())
    if original_forecast_audit["test_used"] or history_audit["test_used"]:
        raise RuntimeError("baseline/history COT sidecar accessed test")
    original_forecast_path = original_forecast_dir / "val_forecast_cot_log1p.npy"
    history_path = history_dir / "val_history_cot_log1p.npy"
    if sha(original_forecast_path) != original_forecast_audit["splits"]["val"]["sha256"]:
        raise RuntimeError("formal SimVP COT bank hash mismatch")
    if sha(history_path) != history_audit["splits"]["val"]["sha256"]:
        raise RuntimeError("fixed history COT bank hash mismatch")

    # Confirm AFNO and the previous formal SimVP cache use identical sequence
    # ordering, and confirm the SimVP trajectory bank is copied from that cache.
    with (afno_bank_dir / "index.csv").open(newline="") as stream:
        afno_index = list(csv.DictReader(stream))
    cache_index_path = root / "data/frozen_forecast_trainval_20260907_v1/val/index.csv"
    with cache_index_path.open(newline="") as stream:
        simvp_index = list(csv.DictReader(stream))
    if len(afno_index) != 4258 or len(simvp_index) != 4258:
        raise RuntimeError("validation sequence count drift")
    for a, s in zip(afno_index, simvp_index):
        if a["seq_id"] != s["seq_id"]:
            raise RuntimeError("AFNO/SimVP validation sequence order mismatch")
    simvp_forecast = np.load(original_forecast_path, mmap_mode="r", allow_pickle=False)
    afno_forecast = np.load(afno_path, mmap_mode="r", allow_pickle=False)
    history = torch.from_numpy(np.load(history_path, allow_pickle=False)).cuda()
    if simvp_forecast.shape != afno_forecast.shape or simvp_forecast.shape != (4258, 2, 16, 1, 16, 16):
        raise RuntimeError("forecast COT array shape mismatch")
    if history.shape != (4258, 2, 8, 1, 16, 16):
        raise RuntimeError("history COT array shape mismatch")

    # COT model cache must be the same SimVP->R product that trained the GHI head.
    simvp_cache = root / "data/frozen_forecast_trainval_20260907_v1/val"
    cache_differences = []
    for shard_path in sorted(simvp_cache.glob("shard_*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            indices, cached = shard["indices"].astype(int), shard["cot_log1p"]
        delta = np.abs(np.asarray(simvp_forecast[indices, :, :, 0]) - cached)
        cache_differences.append(float(delta.max(initial=0)))
    simvp_vs_frozen_cache_max_abs = max(cache_differences, default=float("inf"))
    if simvp_vs_frozen_cache_max_abs > 1e-6:
        raise RuntimeError("formal SimVP COT sidecar differs from paired comparison cache")

    image = np.load(image_dir / "image.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(image_dir / "metadata.npy", mmap_mode="r", allow_pickle=False)
    data = {key: torch.from_numpy(np.load(base / f"{key}.npy", allow_pickle=False)).cuda()
            for key in ("ghi", "clear", "station", "lead", "sequence", "split")}
    val_total = int((data["split"] == 1).sum())
    if val_total != 99849 or image.shape != (637902, 16, 16, 16) or metadata.shape != (637902, 5):
        raise RuntimeError("fixed GHI validation input contract drift")
    if not bool((data["clear"] > 0).all()):
        raise RuntimeError("invalid clear-sky denominator")
    if not np.isfinite(afno_forecast).all():
        raise RuntimeError("non-finite AFNO COT bank")

    # Join every stored GHI row back to its sequence cache index, station,
    # lead, and BJT init date. This ensures the AFNO sequence slots line up
    # with the exact head-pack observations before any metric is computed.
    split_cpu = np.load(base / "split.npy", mmap_mode="r", allow_pickle=False)
    sequence_cpu = np.load(base / "sequence.npy", mmap_mode="r", allow_pickle=False)
    station_cpu = np.load(base / "station.npy", mmap_mode="r", allow_pickle=False)
    lead_cpu = np.load(base / "lead.npy", mmap_mode="r", allow_pickle=False)
    pack_day = np.empty(len(split_cpu), dtype="U10")
    checked_rows = 0
    with (base / "rows.csv").open(newline="") as stream:
        for pack_row, row in enumerate(csv.DictReader(stream)):
            if pack_row >= len(split_cpu):
                raise RuntimeError("GHI rows.csv contains extra rows")
            if row["split"] not in ("train", "val", "validation"):
                raise RuntimeError(f"unexpected GHI head split: {row['split']}")
            split_id = 0 if row["split"] == "train" else 1
            if int(split_cpu[pack_row]) != split_id:
                raise RuntimeError("rows.csv split order differs from head tensors")
            if split_id == 0:
                continue
            slot = int(row["cache_index"])
            station_id = int(row["station_index"])
            lead_id = int(row["lead_index"])
            if (int(sequence_cpu[pack_row]) != slot or int(station_cpu[pack_row]) != station_id or
                    int(lead_cpu[pack_row]) != lead_id or
                    int(row["lead_minutes"]) != 15 * (lead_id + 1) or slot >= len(afno_index) or
                    row["simvp_seq_id"] != afno_index[slot]["seq_id"]):
                raise RuntimeError("GHI row does not join to the AFNO/SimVP validation sequence key")
            pack_day[pack_row] = row["init_time_bjt"][:10]
            checked_rows += 1
    if pack_row + 1 != len(split_cpu):
        raise RuntimeError("GHI rows.csv is shorter than the head tensors")
    if checked_rows != val_total:
        raise RuntimeError("incomplete row-key join for validation GHI rows")

    # Reuse precisely the scalar train-only normalization from the frozen head.
    cot_norm = {name: torch.tensor(head["cot_normalization"][name], device="cuda")
                for name in ("mean", "std")}
    if float(cot_norm["std"]) <= 0:
        raise RuntimeError("invalid frozen head COT normalization")
    baseline_bank = torch.from_numpy(np.asarray(simvp_forecast).copy()).cuda()
    afno_bank = torch.from_numpy(np.asarray(afno_forecast).copy()).cuda()
    results = {"state": "RUNNING", "test_used": False, "arms": {}}
    for station_id, station_name in ((0, "Sili"), (1, "Zhujia")):
        ids = torch.where((data["split"] == 1) & (data["station"] == station_id))[0]
        pack_rows = ids.detach().cpu().numpy().astype(np.int64)
        sequence = data["sequence"][ids].detach().cpu().numpy().astype(np.int64)
        lead_index = data["lead"][ids].detach().cpu().numpy().astype(np.int64)
        if sequence.min() < 0 or sequence.max() >= len(afno_index):
            raise RuntimeError("GHI head rows point outside Hunan validation sequence bank")
        days = pack_day[pack_rows]
        results["arms"][station_name] = {}
        for group in ("AC_F", "AC_ST"):
            checkpoint = head_dir / f"{station_name}_{group}_best.pt"
            expected_hash = head["runs"][station_name][group]["checkpoint_sha256"]
            if sha(checkpoint) != expected_hash:
                raise RuntimeError("fixed GHI head checkpoint hash mismatch")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["station"] != station_name or state["group"] != group:
                raise RuntimeError("wrong frozen GHI head")
            model = CausalCOTTrajectorySolarResNetV5().cuda().eval().requires_grad_(False)
            model.load_state_dict(state["model"], strict=True)
            predictions = {}
            for source_name, forecast_bank in (("SimVP", baseline_bank), ("AFNO", afno_bank)):
                banks = {"val": {"history": history, "forecast": forecast_bank}}
                metrics, pred_kt = evaluate(model, data, banks, image, metadata, ids,
                                            group, batch=512, cot_norm=cot_norm, keep=True)
                pred_ghi = pred_kt.astype(np.float64) * data["clear"][ids].cpu().numpy().astype(np.float64)
                truth_ghi = data["ghi"][ids].cpu().numpy().astype(np.float64)
                if not np.isfinite(pred_ghi).all():
                    raise RuntimeError("non-finite GHI predictions")
                predictions[source_name] = {"kt": pred_kt, "ghi": pred_ghi, "metrics": metrics}
            del model
            torch.cuda.empty_cache()

            simvp = predictions["SimVP"]
            afno = predictions["AFNO"]
            truth = data["ghi"][ids].cpu().numpy().astype(np.float64)
            simvp_ghi, afno_ghi = simvp["ghi"], afno["ghi"]
            rmse_delta = float(np.sqrt(np.mean((afno_ghi - truth) ** 2)) -
                               np.sqrt(np.mean((simvp_ghi - truth) ** 2)))
            if abs(rmse_delta - (afno["metrics"]["ghi_rmse"] - simvp["metrics"]["ghi_rmse"])) > 1e-4:
                raise RuntimeError("NumPy/Torch paired RMSE mismatch")

            # Replay the original trained-on-SimVP prediction as an integrity witness.
            reference_path = head_dir / f"{station_name}_{group}_validation_predictions.npz"
            with np.load(reference_path, allow_pickle=False) as prior:
                if (not np.array_equal(prior["pack_row"], ids.cpu().numpy()) or
                        not np.array_equal(prior["observed_ghi"], truth)):
                    raise RuntimeError("same-head baseline rows/truth differ from original validation")
                replay_delta = float(np.max(np.abs(prior["pred_kt"].astype(np.float64) - simvp["kt"])))
                if replay_delta > 0.005:
                    raise RuntimeError("SimVP fixed-head replay disagrees with original saved predictions")

            # Per-lead RMSE; lead indices 0..15 mean +15..+240 minutes.
            per_lead = {}
            for k in range(16):
                selected = lead_index == k
                if not selected.any():
                    raise RuntimeError("missing validation lead")
                per_lead[str(15 * (k + 1))] = {
                    "samples": int(selected.sum()),
                    "SimVP_RMSE_Wm2": float(np.sqrt(np.mean((simvp_ghi[selected] - truth[selected]) ** 2))),
                    "AFNO_RMSE_Wm2": float(np.sqrt(np.mean((afno_ghi[selected] - truth[selected]) ** 2))),
                }
            bootstrap = day_bootstrap(truth, simvp_ghi, afno_ghi, days)
            results["arms"][station_name][group] = {
                "rows": len(ids), "days": int(len(np.unique(days))),
                "SimVP_RMSE_Wm2": simvp["metrics"]["ghi_rmse"],
                "AFNO_RMSE_Wm2": afno["metrics"]["ghi_rmse"],
                "delta_AFNO_minus_SimVP_RMSE_Wm2": rmse_delta,
                "percent_RMSE_change": 100.0 * rmse_delta / simvp["metrics"]["ghi_rmse"],
                "paired_day_bootstrap": bootstrap,
                "per_lead": per_lead,
                "fixed_head_replay_max_abs_kt": replay_delta,
            }
            row_ids = ids.cpu().numpy()
            pred_path = output / f"{station_name}_{group}_paired_predictions.npz"
            np.savez_compressed(
                pred_path, pack_row=row_ids, sequence_index=sequence, init_day=days,
                lead_minutes=15 * (lead_index + 1), true_ghi=truth,
                simvp_ghi=simvp_ghi, afno_ghi=afno_ghi,
                simvp_pred_kt=simvp["kt"], afno_pred_kt=afno["kt"],
            )
            results["arms"][station_name][group]["predictions_sha256"] = sha(pred_path)
            print(station_name, group, json.dumps(results["arms"][station_name][group], allow_nan=False), flush=True)
            atomic_json(output / "progress.json", results)

    results["state"] = "COMPLETE_PAIRED_FROZEN_GHI_HEAD_AFNO_COT_SWAP"
    results["contract"] = {
        "test_used": False, "input_mutation": "only future forecast COT bank: formal SimVP->R replaced by AFNO->same frozen R",
        "fixed": ["same AC_F/AC_ST head weights", "same history COT", "same train-only COT normalization",
                  "same image/metadata", "same row keys/GHI/clear labels", "same causal lead mask"],
        "head_checkpoint_set_sha256": sha(head_complete_path),
        "AFNO_COT_complete_sha256": sha(afno_complete_path),
        "SimVP_forecast_COT_audit_sha256": sha(original_forecast_dir / "audit.json"),
        "history_COT_audit_sha256": sha(history_dir / "audit.json"),
        "frozen_head_input_audit_sha256": sha(image_dir / "audit.json"),
        "base_pack_audit_sha256": sha(base / "audit.json"),
        "SimVP_COT_vs_frozen_forecast_cache_max_abs": simvp_vs_frozen_cache_max_abs,
        "normalization": "frozen original train-only COT scalar mean/std; no refit on AFNO",
        "scope": "validation-only post-hoc input swap on seed42 exploratory heads trained with SimVP COT; not an AFNO-head retraining or independent test",
    }
    atomic_json(output / "complete.json", results)
    print(results["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--afno-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.afno_bank, args.output)
