"""Paired exploratory training for causal historic-and-forecast COT trajectories.

The only difference across A_ST, AC_H, AC_F and AC_ST is which COT values are
zeroed.  The head, rows, labels, geometry, optimizer, random seed and causal
availability mask are identical.  No patch mean, COT statistics or future
reference product is read.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from causal_cot_trajectory_head_v4 import CausalCOTTrajectoryHead


ROWS, TRAIN_ROWS, VAL_ROWS = 637902, 538053, 99849
GROUPS = ("A_ST", "AC_H", "AC_F", "AC_ST")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def seed(value):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def objective(pred, ghi, clear, weight):
    return (((pred.float() * clear.float() - ghi.float()) / 1000.0).square() * weight.float()).mean()


def load_bank(directory, prefix, split):
    audit = json.loads((directory / "audit.json").read_text())
    expected = ("COMPLETE_FORECAST_COT_TRAJECTORY_SIDECAR" if prefix == "forecast"
                else "COMPLETE_HISTORY_COT_TRAJECTORY_SIDECAR")
    if audit["state"] != expected or audit["test_used"]:
        raise RuntimeError("invalid trajectory sidecar state")
    filename = directory / f"{split}_{prefix}_cot_log1p.npy"
    if sha(filename) != audit["splits"][split]["sha256"]:
        raise RuntimeError("trajectory sidecar hash drift: " + str(filename))
    return torch.from_numpy(np.load(filename, allow_pickle=False)).cuda()


def batch_trajectory(data, banks, ids, group, cot_norm):
    split_value = int(data["split"][ids[0]])
    if not bool((data["split"][ids] == split_value).all()):
        raise RuntimeError("one batch cannot mix split-local sequence identifiers")
    split = "train" if split_value == 0 else "val"
    sequence, station = data["sequence"][ids], data["station"][ids]
    historic = banks[split]["history"][sequence, station]
    forecast = banks[split]["forecast"][sequence, station]
    historic = (historic - cot_norm["mean"]) / cot_norm["std"]
    forecast = (forecast - cot_norm["mean"]) / cot_norm["std"]
    if group not in ("AC_H", "AC_ST"):
        historic = torch.zeros_like(historic)
    if group not in ("AC_F", "AC_ST"):
        forecast = torch.zeros_like(forecast)
    return historic, forecast


def predict(model, data, banks, ids, group, cot_norm):
    historic, forecast = batch_trajectory(data, banks, ids, group, cot_norm)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(data["x"][ids], historic, forecast, data["g"][ids],
                     data["station"][ids], data["lead"][ids]).float()


def update(model, optimizer, data, banks, ids, group, cot_norm):
    optimizer.zero_grad(set_to_none=True)
    pred = predict(model, data, banks, ids, group, cot_norm)
    value = objective(pred, data["ghi"][ids], data["clear"][ids], data["weight"][ids])
    if not torch.isfinite(value):
        raise RuntimeError("non-finite GHI objective")
    value.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("non-finite trajectory-head gradient")
    optimizer.step()
    return float(value.detach())


@torch.no_grad()
def evaluate(model, data, banks, ids, group, batch, cot_norm, keep=False):
    model.eval()
    sse = torch.zeros(2, dtype=torch.float64, device="cuda")
    count = torch.zeros(2, dtype=torch.int64, device="cuda")
    output = []
    for part in ids.split(batch):
        pred = predict(model, data, banks, part, group, cot_norm)
        if not torch.isfinite(pred).all():
            raise RuntimeError("non-finite validation prediction")
        error = pred.double() * data["clear"][part] - data["ghi"][part]
        for site in (0, 1):
            mask = data["station"][part] == site
            sse[site] += error[mask].square().sum()
            count[site] += mask.sum()
        if keep:
            output.append(pred.cpu().numpy())
    if not bool((count > 0).all()):
        raise RuntimeError("station missing from validation")
    rmse = torch.sqrt(sse / count)
    return ({"station_equal_ghi_rmse": float(rmse.mean()), "station_ghi_rmse": rmse.cpu().tolist(),
             "station_counts": count.cpu().tolist()}, np.concatenate(output) if keep else None)


def profile(data, banks, train, out, cot_norm):
    trials = []
    for batch in (256, 512, 1024, 2048):
        model = optimizer = None
        try:
            seed(42)
            model = CausalCOTTrajectoryHead().cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            ids = train[:batch]
            for _ in range(2):
                update(model, optimizer, data, banks, ids, "AC_ST", cot_norm)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
            for _ in range(3):
                update(model, optimizer, data, banks, ids, "AC_ST", cot_norm)
            torch.cuda.synchronize()
            seconds = (time.monotonic() - started) / 3
            trial = {"batch": batch, "rows_per_second": batch / seconds,
                     "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}
            trials.append(trial)
            if trial["peak_GiB"] >= .80 * torch.cuda.get_device_properties(0).total_memory / 2**30:
                break
        except torch.cuda.OutOfMemoryError:
            trials.append({"batch": batch, "state": "OOM"})
            break
        finally:
            del model, optimizer
            gc.collect(); torch.cuda.empty_cache()
    usable = [trial for trial in trials if "rows_per_second" in trial and trial["peak_GiB"] <
              .80 * torch.cuda.get_device_properties(0).total_memory / 2**30]
    if not usable:
        raise RuntimeError("no safe profile batch")
    if not any(trial["batch"] == 2048 for trial in usable):
        raise RuntimeError("batch2048 lacks the required 20 percent GPU memory headroom")
    result = {"trials": trials, "training_batch": 2048,
              "selection": "fixed 2048 preserves B1 optimizer-update budget; profile only records headroom"}
    atomic(out / "profile.json", result)
    return result


@torch.no_grad()
def fit_train_cot_norm(train_banks, sequence):
    """One scalar train-only normalization; it is not a spatial aggregation feature."""
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    squared = torch.zeros((), device="cuda", dtype=torch.float64)
    count = 0
    for bank in (train_banks["history"], train_banks["forecast"]):
        for part in sequence.split(512):
            value = bank[part].double()
            total += value.sum(); squared += value.square().sum(); count += value.numel()
    mean = total / count
    variance = squared / count - mean.square()
    if not bool(torch.isfinite(variance)) or float(variance) <= 0:
        raise RuntimeError("invalid train-only COT normalization")
    return {"mean": mean.float(), "std": variance.sqrt().float(), "count": count,
            "mean_value": float(mean), "std_value": float(variance.sqrt())}


def main(root, forecast_dir, history_dir, out, epochs, batch, run_seed):
    root, forecast_dir, history_dir, out = (item.resolve() for item in (root, forecast_dir, history_dir, out))
    if out.exists():
        raise RuntimeError("refuse to overwrite trajectory run")
    base = root / "data/head_pack_trainval_20260908_v1"
    audit = json.loads((base / "audit.json").read_text())
    if audit["test_used"]:
        raise RuntimeError("test-contaminated base pack")
    keys = ("x", "g", "ghi", "clear", "station", "lead", "sequence", "split", "weight")
    data = {key: torch.from_numpy(np.load(base / f"{key}.npy", allow_pickle=False)).cuda() for key in keys}
    if data["x"].shape != (ROWS, 13, 16, 16) or len(data["split"]) != ROWS:
        raise RuntimeError("frozen head-pack row contract drift")
    train, val = torch.where(data["split"] == 0)[0], torch.where(data["split"] == 1)[0]
    if len(train) != TRAIN_ROWS or len(val) != VAL_ROWS or not bool((data["clear"] > 0).all()):
        raise RuntimeError("train/validation/clear contract drift")
    banks = {split: {"forecast": load_bank(forecast_dir, "forecast", split),
                     "history": load_bank(history_dir, "history", split)} for split in ("train", "val")}
    for split in ("train", "val"):
        required = torch.unique(data["sequence"][data["split"] == (0 if split == "train" else 1)])
        if int(required.max()) >= len(banks[split]["forecast"]) or int(required.max()) >= len(banks[split]["history"]):
            raise RuntimeError("sequence-bank range drift")
        if banks[split]["forecast"].shape[1:] != (2, 16, 1, 16, 16) or banks[split]["history"].shape[1:] != (2, 8, 1, 16, 16):
            raise RuntimeError("trajectory bank shape drift")
    train_sequence = torch.unique(data["sequence"][data["split"] == 0])
    cot_norm = fit_train_cot_norm(banks["train"], train_sequence)
    out.mkdir(parents=True)
    profile_result = profile(data, banks, train, out, cot_norm)
    if batch != 2048:
        raise RuntimeError("batch is frozen at 2048 for matched B1 update budget")
    contract = {"state": "EXPLORATORY_CAUSAL_COT_TRAJECTORY_CPP_PROVENANCE_UNRESOLVED", "test_used": False,
                "groups": list(GROUPS), "seed": run_seed, "epochs_max": epochs, "batch": batch,
                "frozen_S": True, "frozen_R": True, "head": "causal dilated 3D COT trajectory then station [8,8] readout",
                "forbidden": "no full-patch average; no COT statistics; no post-target forecast map; no reference CPP",
                "cot_normalization": {"fit": "one scalar over only unique training sequences and all COT pixels/times",
                                      "mean": cot_norm["mean_value"], "std": cot_norm["std_value"], "count": cot_norm["count"]},
                "loss": "mean(w*((pred_kt*clear-ghi)/1000)^2)", "base_pack_audit_sha256": sha(base / "audit.json"),
                "forecast_sidecar_audit_sha256": sha(forecast_dir / "audit.json"),
                "history_sidecar_audit_sha256": sha(history_dir / "audit.json"), "script_sha256": sha(__file__)}
    atomic(out / "execution_contract.json", contract)
    report = {"state": "RUNNING_EXPLORATORY_CAUSAL_COT_TRAJECTORY", **contract, "profile": profile_result, "runs": {}}
    for group in GROUPS:
        seed(run_seed)
        model = CausalCOTTrajectoryHead().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best, best_epoch, records = float("inf"), None, []
        for epoch in range(epochs):
            model.train(); order = train[torch.randperm(len(train), generator=torch.Generator(device="cuda").manual_seed(run_seed + epoch), device="cuda")]
            total, started = 0.0, time.monotonic()
            for part in order.split(batch):
                total += update(model, optimizer, data, banks, part, group, cot_norm) * len(part)
            metrics, _ = evaluate(model, data, banks, val, group, batch, cot_norm)
            record = {"epoch": epoch, "train_weighted_normalized_ghi_mse": total / len(train), **metrics,
                      "seconds": time.monotonic() - started}
            records.append(record); atomic(out / f"{group}_epoch_{epoch:02d}.json", record)
            print(group, json.dumps(record), flush=True)
            if metrics["station_equal_ghi_rmse"] < best:
                best, best_epoch = metrics["station_equal_ghi_rmse"], epoch
                save(out / f"{group}_best.pt", {"model": model.state_dict(), "group": group,
                                                   "epoch": epoch, "contract": contract})
        model.load_state_dict(torch.load(out / f"{group}_best.pt", map_location="cpu")["model"], strict=True)
        model.cuda().eval(); metrics, pred = evaluate(model, data, banks, val, group, batch, cot_norm, keep=True)
        path = out / f"{group}_validation_predictions.npz"
        np.savez(path, pack_row=val.cpu().numpy(), pred_kt=pred, observed_ghi=data["ghi"][val].cpu().numpy(),
                 clear_sky_ghi=data["clear"][val].cpu().numpy(), station=data["station"][val].cpu().numpy(),
                 lead=data["lead"][val].cpu().numpy())
        report["runs"][group] = {"best_epoch_0based": best_epoch, **metrics, "epochs": records,
                                  "checkpoint_sha256": sha(out / f"{group}_best.pt"), "predictions_sha256": sha(path)}
        atomic(out / "progress.json", report)
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()
    report["state"] = "COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY"
    report["next"] = "Audit saved paired validation outputs before any seed expansion, IFS, or test."
    atomic(out / "complete.json", report)
    print(report["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-unverified-cpp", action="store_true")
    args = parser.parse_args()
    if not args.allow_unverified_cpp:
        raise RuntimeError("requires explicit exploratory CPP authorization")
    main(args.root, args.forecast_dir, args.history_dir, args.out, args.epochs, args.batch, args.seed)
