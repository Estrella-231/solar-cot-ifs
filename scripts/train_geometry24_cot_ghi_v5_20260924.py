"""Paired Hunan GHI heads using geometry24 SimVP and a frozen COT retriever.

The only difference across A_ST, AC_H and AC_ST is which COT values are
zeroed. The head, rows, labels, geometry, optimizer, random seed and causal
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

from causal_cot_trajectory_solarresnet_v5 import CausalCOTTrajectorySolarResNetV5


ROWS, TRAIN_ROWS, VAL_ROWS = 637902, 538053, 99849
GROUPS = ("A_ST", "AC_H", "AC_ST")
STATIONS = ((0, "Sili"), (1, "Zhujia"))


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


def load_geometry24_forecast(directory, split):
    contract = json.loads((directory / "complete.json").read_text())
    if (contract["state"] != "COMPLETE_GEOMETRY24_FORECAST_COT" or contract["test_used"] or
            contract["split"] != split or contract["backbone"] != "simvp" or
            contract["geometry_contract"] != "history_8_then_target_16_v2" or
            contract["geometry_forward_indices"] != list(range(8, 24))):
        raise RuntimeError("invalid geometry24 SimVP forecast bank")
    path = directory / "forecast_cot_log1p.npy"
    if sha(path) != contract["forecast_cot_sha256"]:
        raise RuntimeError("forecast COT bank hash drift")
    return torch.from_numpy(np.load(path, allow_pickle=False)).cuda()


def load_real_future_oracle(directory, split):
    contract = json.loads((directory / "complete.json").read_text())
    if (contract["state"] != "COMPLETE_REAL_FUTURE_R_ORACLE" or contract["test_used"] or
            contract["deployable"] or contract["split"] != split):
        raise RuntimeError("invalid oracle reference bank")
    path = directory / "real_future_cot_log1p.npy"
    if sha(path) != contract["cot_sha256"]:
        raise RuntimeError("real-future oracle bank hash drift")
    return torch.from_numpy(np.load(path, allow_pickle=False)).cuda()


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


def image_batch(image, metadata, ids):
    """Copy one batch from the host-resident v5 sidecar to the selected GPU."""
    rows = ids.detach().cpu().numpy()
    return (torch.from_numpy(np.array(image[rows], copy=True)).cuda(non_blocking=True),
            torch.from_numpy(np.array(metadata[rows], copy=True)).cuda(non_blocking=True))


def predict(model, data, banks, image, metadata, ids, group, cot_norm):
    historic, forecast = batch_trajectory(data, banks, ids, group, cot_norm)
    image_value, metadata_value = image_batch(image, metadata, ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(image_value, metadata_value, historic, forecast, data["lead"][ids]).float()


def update(model, optimizer, data, banks, image, metadata, ids, group, cot_norm):
    optimizer.zero_grad(set_to_none=True)
    pred = predict(model, data, banks, image, metadata, ids, group, cot_norm)
    value = objective(pred, data["ghi"][ids], data["clear"][ids], data["weight"][ids])
    if not torch.isfinite(value):
        raise RuntimeError("non-finite GHI objective")
    value.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("non-finite SolarResNet-v5 gradient")
    optimizer.step()
    return float(value.detach())


@torch.no_grad()
def evaluate(model, data, banks, image, metadata, ids, group, batch, cot_norm, keep=False):
    model.eval()
    sse = torch.zeros((), dtype=torch.float64, device="cuda")
    count = torch.zeros((), dtype=torch.int64, device="cuda")
    output = []
    for part in ids.split(batch):
        pred = predict(model, data, banks, image, metadata, part, group, cot_norm)
        if not torch.isfinite(pred).all():
            raise RuntimeError("non-finite validation prediction")
        error = pred.double() * data["clear"][part] - data["ghi"][part]
        sse += error.square().sum()
        count += len(error)
        if keep:
            output.append(pred.cpu().numpy())
    if int(count) == 0:
        raise RuntimeError("empty validation station cohort")
    return ({"ghi_rmse": float(torch.sqrt(sse / count)), "sample_count": int(count)},
            np.concatenate(output) if keep else None)


def profile(data, banks, image, metadata, train, out, cot_norm):
    trials = []
    for batch in (256, 512, 1024, 1536, 2048):
        model = optimizer = None
        try:
            seed(42)
            model = CausalCOTTrajectorySolarResNetV5().cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            ids = train[:batch]
            for _ in range(2):
                update(model, optimizer, data, banks, image, metadata, ids, "AC_ST", cot_norm)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
            for _ in range(3):
                update(model, optimizer, data, banks, image, metadata, ids, "AC_ST", cot_norm)
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
    # Select by measured throughput under the safe-memory envelope.  The
    # largest batch can be slower when shared-storage reads dominate, as
    # happened in the first pilot (2048 was much slower than 1536).
    chosen = max(usable, key=lambda trial: trial["rows_per_second"])
    result = {"trials": trials, "training_batch": chosen["batch"],
              "selection": "highest measured row throughput below 80 percent allocated-memory threshold"}
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


def main(root, forecast_train_dir, forecast_val_dir, history_dir, image_sidecar, out,
         epochs, batch, run_seed, forecast_kind, cot_normalization_json=None,
         selected_groups=GROUPS):
    root, forecast_train_dir, forecast_val_dir, history_dir, image_sidecar, out = (
        item.resolve() for item in (root, forecast_train_dir, forecast_val_dir, history_dir, image_sidecar, out))
    if out.exists():
        raise RuntimeError("refuse to overwrite trajectory run")
    base = root / "data/head_pack_trainval_20260908_v1"
    audit = json.loads((base / "audit.json").read_text())
    if audit["test_used"]:
        raise RuntimeError("test-contaminated base pack")
    input_audit = json.loads((image_sidecar / "audit.json").read_text())
    if input_audit["state"] != "COMPLETE_GEOMETRY24_SOLARRESNET_V5_INPUT_SIDECAR" or input_audit["test_used"]:
        raise RuntimeError("invalid SolarResNet-v5 input sidecar")
    r_contract = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    head_source = root / "scripts/causal_cot_trajectory_solarresnet_v5.py"
    if sha(head_source) != sha(Path(__file__).with_name(head_source.name)):
        raise RuntimeError("deployed GHI head source differs from recorded project source")
    history_audit = json.loads((history_dir / "audit.json").read_text())
    if (history_audit.get("arm") != "original_R" or
            history_audit.get("checkpoint_sha256") != r_contract["checkpoint_sha256"]):
        raise RuntimeError("historical COT does not come from the locked original R")
    for split_name, bank_dir in (("train", forecast_train_dir), ("val", forecast_val_dir)):
        bank_contract = json.loads((bank_dir / "complete.json").read_text())
        if bank_contract.get("R_checkpoint_sha256") != r_contract["checkpoint_sha256"]:
            raise RuntimeError("future COT and historical COT use different retrievers")
        if forecast_kind == "simvp" and input_audit["bank_complete_sha256"][split_name] != sha(bank_dir / "complete.json"):
            raise RuntimeError("image and COT bank source mismatch")
    keys = ("ghi", "clear", "station", "lead", "sequence", "split", "weight")
    data = {key: torch.from_numpy(np.load(base / f"{key}.npy", allow_pickle=False)).cuda() for key in keys}
    # Read the sidecar into host RAM once; shuffled batches never reread shared storage.
    image = np.load(image_sidecar / "image.npy", allow_pickle=False)
    metadata = np.load(image_sidecar / "metadata.npy", allow_pickle=False)
    if image.shape != (ROWS, 16, 16, 16) or metadata.shape != (ROWS, 5) or len(data["split"]) != ROWS:
        raise RuntimeError("frozen head-pack row contract drift")
    if sha(image_sidecar / "image.npy") != input_audit["image_sha256"] or sha(image_sidecar / "metadata.npy") != input_audit["metadata_sha256"]:
        raise RuntimeError("SolarResNet-v5 input sidecar hash drift")
    train, val = torch.where(data["split"] == 0)[0], torch.where(data["split"] == 1)[0]
    if len(train) != TRAIN_ROWS or len(val) != VAL_ROWS or not bool((data["clear"] > 0).all()):
        raise RuntimeError("train/validation/clear contract drift")
    forecast_dirs = {"train": forecast_train_dir, "val": forecast_val_dir}
    loader = load_geometry24_forecast if forecast_kind == "simvp" else load_real_future_oracle
    if forecast_kind == "oracle" and tuple(selected_groups) != ("AC_ST",):
        raise RuntimeError("oracle arm trains only AC_ST; no deployable A0/AH from future real images")
    banks = {split: {"forecast": loader(forecast_dirs[split], split),
                     "history": load_bank(history_dir, "history", split)} for split in ("train", "val")}
    for split in ("train", "val"):
        required = torch.unique(data["sequence"][data["split"] == (0 if split == "train" else 1)])
        if int(required.max()) >= len(banks[split]["forecast"]) or int(required.max()) >= len(banks[split]["history"]):
            raise RuntimeError("sequence-bank range drift")
        if banks[split]["forecast"].shape[1:] != (2, 16, 1, 16, 16) or banks[split]["history"].shape[1:] != (2, 8, 1, 16, 16):
            raise RuntimeError("trajectory bank shape drift")
    train_sequence = torch.unique(data["sequence"][data["split"] == 0])
    normalization_source = None
    if cot_normalization_json is None:
        cot_norm = fit_train_cot_norm(banks["train"], train_sequence)
    else:
        normalization_path = cot_normalization_json.resolve()
        normalization_source = {"path": str(normalization_path), "sha256": sha(normalization_path)}
        specification = json.loads(normalization_path.read_text())["cot_normalization"]
        mean_value, std_value = float(specification["mean"]), float(specification["std"])
        count = int(specification["count"])
        if not np.isfinite([mean_value, std_value]).all() or std_value <= 0 or count <= 0:
            raise RuntimeError("invalid externally frozen train-only COT normalization")
        cot_norm = {"mean": torch.tensor(mean_value, dtype=torch.float32, device="cuda"),
                    "std": torch.tensor(std_value, dtype=torch.float32, device="cuda"),
                    "count": count, "mean_value": mean_value, "std_value": std_value}
    out.mkdir(parents=True)
    profile_result = profile(data, banks, image, metadata, train[data["station"][train] == 0], out, cot_norm)
    if batch == 0:
        batch = profile_result["training_batch"]
    else:
        profiled = [trial for trial in profile_result["trials"] if trial.get("batch") == batch and
                    "rows_per_second" in trial and trial["peak_GiB"] <
                    .80 * torch.cuda.get_device_properties(0).total_memory / 2**30]
        if not profiled:
            raise RuntimeError("requested fixed comparison batch was not profiled as safe")
    contract = {"state": "EXPLORATORY_CAUSAL_COT_TRAJECTORY_CPP_PROVENANCE_UNRESOLVED", "test_used": False,
                "groups": list(selected_groups), "seed": run_seed, "epochs_max": epochs, "batch": batch,
                "frozen_S": True, "frozen_R": True, "backbone": "geometry24 SimVP best checkpoint",
                "future_cot_kind": forecast_kind, "deployable": forecast_kind == "simvp",
                "group_meanings": {"A_ST": "A0: zero history/future COT", "AC_H": "AH: history COT only",
                                   "AC_ST": "AI: history plus SimVP forecast COT"},
                "head": "SolarResNet image trunk plus causal dilated 3D COT trajectory fusion",
                "forbidden": "no full-patch average; no COT statistics; no post-target forecast map; no reference CPP",
                "cot_normalization": {"fit": "one scalar over only unique training sequences and all COT pixels/times",
                                      "mean": cot_norm["mean_value"], "std": cot_norm["std_value"], "count": cot_norm["count"]},
                "cot_normalization_source": normalization_source or "fitted from this route train split only",
                "loss": "mean(w*((pred_kt*clear-ghi)/1000)^2)", "base_pack_audit_sha256": sha(base / "audit.json"),
                "input_sidecar_audit_sha256": sha(image_sidecar / "audit.json"),
                "forecast_train_complete_sha256": sha(forecast_train_dir / "complete.json"),
                "forecast_val_complete_sha256": sha(forecast_val_dir / "complete.json"),
                "history_sidecar_audit_sha256": sha(history_dir / "audit.json"),
                "head_model_code_sha256": sha(head_source),
                "script_sha256": sha(__file__)}
    atomic(out / "execution_contract.json", contract)
    report = {"state": "RUNNING_EXPLORATORY_CAUSAL_COT_TRAJECTORY", **contract, "profile": profile_result,
              "station_training": "one independently initialized head per station", "runs": {}}
    for station_id, station_name in STATIONS:
        station_train = train[data["station"][train] == station_id]
        station_val = val[data["station"][val] == station_id]
        if len(station_train) == 0 or len(station_val) == 0:
            raise RuntimeError("missing station cohort: " + station_name)
        report["runs"][station_name] = {}
        for group in selected_groups:
            seed(run_seed)
            model = CausalCOTTrajectorySolarResNetV5().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            best, best_epoch, records = float("inf"), None, []
            for epoch in range(epochs):
                model.train()
                generator = torch.Generator(device="cuda").manual_seed(run_seed + epoch)
                order = station_train[torch.randperm(len(station_train), generator=generator, device="cuda")]
                total, started = 0.0, time.monotonic()
                for part in order.split(batch):
                    total += update(model, optimizer, data, banks, image, metadata, part, group, cot_norm) * len(part)
                metrics, _ = evaluate(model, data, banks, image, metadata, station_val, group, batch, cot_norm)
                record = {"epoch": epoch, "train_weighted_normalized_ghi_mse": total / len(station_train), **metrics,
                          "seconds": time.monotonic() - started}
                records.append(record); atomic(out / f"{station_name}_{group}_epoch_{epoch:02d}.json", record)
                print(station_name, group, json.dumps(record), flush=True)
                if metrics["ghi_rmse"] < best:
                    best, best_epoch = metrics["ghi_rmse"], epoch
                    save(out / f"{station_name}_{group}_best.pt", {"model": model.state_dict(), "group": group,
                                                                     "station": station_name, "epoch": epoch, "contract": contract})
            checkpoint = out / f"{station_name}_{group}_best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=True)
            model.cuda().eval(); metrics, pred = evaluate(model, data, banks, image, metadata, station_val, group, batch, cot_norm, keep=True)
            path = out / f"{station_name}_{group}_validation_predictions.npz"
            np.savez(path, pack_row=station_val.cpu().numpy(), pred_kt=pred,
                     observed_ghi=data["ghi"][station_val].cpu().numpy(), clear_sky_ghi=data["clear"][station_val].cpu().numpy(),
                     station=data["station"][station_val].cpu().numpy(), lead=data["lead"][station_val].cpu().numpy())
            report["runs"][station_name][group] = {"best_epoch_0based": best_epoch, **metrics, "epochs": records,
                                                      "checkpoint_sha256": sha(checkpoint), "predictions_sha256": sha(path)}
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
    parser.add_argument("--forecast-train-dir", type=Path, required=True)
    parser.add_argument("--forecast-val-dir", type=Path, required=True)
    parser.add_argument("--forecast-kind", choices=("simvp", "oracle"), default="simvp")
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--image-sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=0, help="0 selects the largest safe profiled batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot-normalization-json", type=Path,
                        help="reuse a frozen train-only COT mean/std from a matched reference run")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS),
                        help="optional subset of paired COT input groups")
    parser.add_argument("--allow-unverified-cpp", action="store_true")
    args = parser.parse_args()
    if not args.allow_unverified_cpp:
        raise RuntimeError("requires explicit exploratory CPP authorization")
    main(args.root, args.forecast_train_dir, args.forecast_val_dir, args.history_dir, args.image_sidecar,
         args.out, args.epochs, args.batch, args.seed, args.forecast_kind,
         args.cot_normalization_json, tuple(args.groups))
