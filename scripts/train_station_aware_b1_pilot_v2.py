"""Validation-only seed-42 pilot for station-aware aligned AGRI+COT fusion."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from station_aware_spatial_head_v2 import StationAwareSpatialHead

ROOT_ROWS, TRAIN_ROWS, VAL_ROWS = 637902, 538053, 99849
GROUPS = ("A_station", "AC_station")
FUSION_DESCRIPTION = "aligned concat(13 AGRI, 1 COT) before first convolution"
AGGREGATION_DESCRIPTION = ("station [8,8] feature + 5x5 station neighbourhood + "
                           "learned position-preserving 4x4 full-patch encoding; "
                           "no COT mean/percentile/statistical feature")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


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


def loss_fn(pred, ghi, clear, weight):
    return ((((pred.float() * clear.float() - ghi.float()) / 1000.0).square()) * weight.float()).mean()


def predict(model, data, ids, group):
    cot = data["cot_map"][ids] if group == "AC_station" else torch.zeros_like(data["cot_map"][ids])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(data["x"][ids], cot, data["g"][ids],
                     data["station"][ids], data["lead"][ids]).float()


def evaluate(model, data, ids, group, batch, keep=False):
    model.eval()
    sse = torch.zeros(2, dtype=torch.float64, device="cuda")
    count = torch.zeros(2, dtype=torch.int64, device="cuda")
    predictions = []
    with torch.no_grad():
        for part in ids.split(batch):
            pred = predict(model, data, part, group)
            if not torch.isfinite(pred).all():
                raise RuntimeError("non-finite validation prediction")
            error = pred.double() * data["clear"][part] - data["ghi"][part]
            for station in (0, 1):
                mask = data["station"][part] == station
                sse[station] += error[mask].square().sum()
                count[station] += mask.sum()
            if keep:
                predictions.append(pred.cpu().numpy())
    if not (count > 0).all():
        raise RuntimeError("missing station validation rows")
    rmse = torch.sqrt(sse / count)
    return ({"station_equal_ghi_rmse": float(rmse.mean()), "station_ghi_rmse": rmse.cpu().tolist(),
             "station_counts": count.cpu().tolist()}, np.concatenate(predictions) if keep else None)


def profile(data, train):
    """Profile the full GPU-resident train pack; profile model is discarded."""
    trials = []
    for batch in (2048, 8192, 16384):
        seed(42)
        model = StationAwareSpatialHead().cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        ids = train[:batch]
        try:
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                pred = predict(model, data, ids, "AC_station")
                value = loss_fn(pred, data["ghi"][ids], data["clear"][ids], data["weight"][ids])
                value.backward()
                opt.step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                pred = predict(model, data, ids, "AC_station")
                value = loss_fn(pred, data["ghi"][ids], data["clear"][ids], data["weight"][ids])
                value.backward()
                opt.step()
            torch.cuda.synchronize()
            seconds = (time.monotonic() - start) / 3
            trial = {"batch": batch, "seconds_per_update": seconds,
                     "rows_per_second": batch / seconds,
                     "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}
            trials.append(trial)
        except torch.cuda.OutOfMemoryError:
            trials.append({"batch": batch, "state": "OOM"})
            torch.cuda.empty_cache()
            break
        finally:
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()
    usable = [x for x in trials if "rows_per_second" in x and x["peak_GiB"] < .80 * torch.cuda.get_device_properties(0).total_memory / 2**30]
    if not usable:
        raise RuntimeError("no profile batch retained 20 percent GPU headroom")
    chosen = max(usable, key=lambda x: x["rows_per_second"])
    return {"trials": trials, "selected_batch": chosen["batch"], "selection": "highest measured rows/s below 80% total memory"}


def main(root, out, batch_override=None):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise RuntimeError("refuse to overwrite station-aware pilot")
    base = root / "data/head_pack_trainval_20260908_v1"
    sidecar = root / "data/head_spatial_cot_trainval_20260917_v2"
    ba, ma = json.loads((base / "audit.json").read_text()), json.loads((sidecar / "audit.json").read_text())
    if ba["test_used"] or ma["test_used"] or ma["state"] != "COMPLETE_SPATIAL_COT_SIDECAR":
        raise RuntimeError("input state is not test-free complete")
    if sha(base / "audit.json") != ma["base_pack_audit_sha256"]:
        raise RuntimeError("base/sidecar contract binding drift")
    for spec in ba["arrays"].values():
        if sha(base / spec["name"]) != spec["sha256"]:
            raise RuntimeError("base array hash drift: " + spec["name"])
    if sha(sidecar / "cot_map.npy") != ma["arrays"]["cot_map.npy"]["sha256"]:
        raise RuntimeError("COT map hash drift")
    keys = ("x", "g", "ghi", "clear", "station", "lead", "split", "weight")
    data = {key: torch.from_numpy(np.load(base / (key + ".npy"), allow_pickle=False)).cuda() for key in keys}
    data["cot_map"] = torch.from_numpy(np.load(sidecar / "cot_map.npy", allow_pickle=False)).cuda()
    train, val = torch.where(data["split"] == 0)[0], torch.where(data["split"] == 1)[0]
    if len(train) != TRAIN_ROWS or len(val) != VAL_ROWS or data["x"].shape != (ROOT_ROWS, 13, 16, 16):
        raise RuntimeError("frozen row or shape contract drift")
    if data["cot_map"].shape != (ROOT_ROWS, 1, 16, 16) or not (data["clear"] > 0).all():
        raise RuntimeError("COT/clear contract drift")
    out.mkdir(parents=True)
    profile_result = profile(data, train)
    batch = batch_override if batch_override is not None else profile_result["selected_batch"]
    if batch not in (2048, 8192, 16384):
        raise RuntimeError("batch override must be a pre-profiled candidate")
    contract = {"state": "EXPLORATORY_STATION_AWARE_B1_CPP_PROVENANCE_UNRESOLVED", "test_used": False,
                "groups": list(GROUPS), "seed": 42, "epochs": 10, "batch": batch,
                "batch_selection": ("forced 2048 to preserve the original B1 optimizer-update budget"
                                    if batch_override is not None else profile_result["selection"]),
                "optimizer": "AdamW(lr=3e-4, weight_decay=1e-4)",
                "loss": "mean(w*((pred_kt*clear-ghi)/1000)^2)",
                "fusion": FUSION_DESCRIPTION,
                "aggregation": AGGREGATION_DESCRIPTION,
                "cot_source": ("frozen COT retriever applied at each target lead to the "
                               "fixed SimVP forecast AGRI; no future reference/CPP COT"),
                "forbidden_cot_features": ["full_patch_mean", "p90", "four_statistics"],
                "base_pack_audit_sha256": sha(base / "audit.json"),
                "sidecar_audit_sha256": sha(sidecar / "audit.json"), "script_sha256": sha(__file__)}
    atomic(out / "execution_contract.json", contract)
    atomic(out / "profile.json", profile_result)
    report = {"state": "RUNNING_EXPLORATORY_STATION_AWARE_B1", **contract, "profile": profile_result, "runs": {}}
    for group in GROUPS:
        seed(42)
        model = StationAwareSpatialHead().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best, best_epoch, epochs = float("inf"), None, []
        for epoch in range(10):
            model.train()
            order = train[torch.randperm(len(train), generator=torch.Generator(device="cuda").manual_seed(42 + epoch), device="cuda")]
            total, started = 0.0, time.monotonic()
            for part in order.split(batch):
                optimizer.zero_grad(set_to_none=True)
                pred = predict(model, data, part, group)
                value = loss_fn(pred, data["ghi"][part], data["clear"][part], data["weight"][part])
                if not torch.isfinite(value):
                    raise RuntimeError("non-finite training loss")
                value.backward()
                optimizer.step()
                total += float(value.detach()) * len(part)
            metrics, _ = evaluate(model, data, val, group, batch)
            record = {"epoch": epoch, "train_weighted_normalized_ghi_mse": total / len(train),
                      **metrics, "seconds": time.monotonic() - started}
            atomic(out / f"{group}_epoch_{epoch:02d}.json", record)
            print(group, json.dumps(record), flush=True)
            epochs.append(record)
            if metrics["station_equal_ghi_rmse"] < best:
                best, best_epoch = metrics["station_equal_ghi_rmse"], epoch
                save(out / f"{group}_best.pt", {"model": model.state_dict(), "group": group,
                                                   "epoch": epoch, "contract": contract})
        checkpoint = torch.load(out / f"{group}_best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.cuda().eval()
        metrics, predictions = evaluate(model, data, val, group, batch, keep=True)
        path = out / f"{group}_validation_predictions.npz"
        np.savez(path, pack_row=val.cpu().numpy(), pred_kt=predictions,
                 observed_ghi=data["ghi"][val].cpu().numpy(), clear_sky_ghi=data["clear"][val].cpu().numpy(),
                 station=data["station"][val].cpu().numpy(), lead=data["lead"][val].cpu().numpy())
        report["runs"][group] = {"best_epoch_0based": best_epoch, **metrics, "epochs": epochs,
                                  "checkpoint_sha256": sha(out / f"{group}_best.pt"),
                                  "predictions_sha256": sha(path)}
        atomic(out / "progress.json", report)
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()
    report["state"] = "COMPLETE_EXPLORATORY_STATION_AWARE_B1"
    report["next"] = "Audit paired saved validation predictions before any seed expansion, IFS, or test."
    atomic(out / "complete.json", report)
    print("STATION_AWARE_B1_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--allow-unverified-cpp", action="store_true")
    args = parser.parse_args()
    if not args.allow_unverified_cpp:
        raise RuntimeError("requires explicit exploratory CPP authorization")
    main(args.root, args.out, args.batch)
