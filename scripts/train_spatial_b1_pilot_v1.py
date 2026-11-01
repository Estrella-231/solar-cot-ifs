"""Exploratory B1 pilot for fixed SimVP/R spatial-COT inputs.

This run is explicitly allowed with an unresolved historical CPP producer
chain.  It is therefore a representation screen only: it cannot be promoted
to a final paper claim without the provenance gate and the preregistered
multi-seed controls.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from spatial_head_smoke_v1 import SpatialHead


ROOT_ROWS = 637902
TRAIN_ROWS = 538053
VAL_ROWS = 99849
GROUPS = ("A_sp", "AC_stat_sp", "AC_map")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def seed(value):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def ghi_loss(pred, truth, clear, weight):
    return ((((pred.float() * clear.float() - truth.float()) / 1000.0).square()) * weight.float()).mean()


def predict(model, data, ids, group):
    c_map = data["cot_map"][ids] if group == "AC_map" else torch.zeros_like(data["cot_map"][ids])
    c_stat = data["c"][ids] if group == "AC_stat_sp" else torch.zeros_like(data["c"][ids])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(data["x"][ids], c_map, c_stat, data["g"][ids], data["station"][ids],
                     data["lead"][ids]).float()


def evaluate(model, data, ids, group, batch, keep=False):
    model.eval()
    sse = torch.zeros(2, dtype=torch.float64, device="cuda")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    outputs = []
    with torch.no_grad():
        for ix in ids.split(batch):
            pred = predict(model, data, ix, group)
            if not torch.isfinite(pred).all():
                raise RuntimeError("nonfinite validation prediction")
            residual = pred.double() * data["clear"][ix] - data["ghi"][ix]
            for station in (0, 1):
                mask = data["station"][ix] == station
                sse[station] += residual[mask].square().sum()
                counts[station] += mask.sum()
            if keep:
                outputs.append(pred.cpu().numpy())
    if not (counts > 0).all():
        raise RuntimeError("missing station in validation")
    rmse = torch.sqrt(sse / counts)
    result = {"station_equal_ghi_rmse": float(rmse.mean()),
              "station_ghi_rmse": rmse.cpu().tolist(), "station_counts": counts.cpu().tolist()}
    return result, np.concatenate(outputs) if keep else None


def main(root, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise RuntimeError("refuse to overwrite pilot directory")
    base = root / "data/head_pack_trainval_20260908_v1"
    sidecar = root / "data/head_spatial_cot_trainval_20260917_v2"
    base_audit = json.loads((base / "audit.json").read_text())
    map_audit = json.loads((sidecar / "audit.json").read_text())
    if base_audit["test_used"] or map_audit["test_used"] or map_audit["state"] != "COMPLETE_SPATIAL_COT_SIDECAR":
        raise RuntimeError("test or incomplete input contract")
    if sha(base / "audit.json") != map_audit["base_pack_audit_sha256"]:
        raise RuntimeError("base/sidecar binding drift")
    for spec in base_audit["arrays"].values():
        if sha(base / spec["name"]) != spec["sha256"]:
            raise RuntimeError("base array hash drift: " + spec["name"])
    if sha(sidecar / "cot_map.npy") != map_audit["arrays"]["cot_map.npy"]["sha256"]:
        raise RuntimeError("COT map hash drift")
    names = ("x", "c", "g", "kt", "ghi", "clear", "station", "lead", "split", "sequence", "weight")
    data = {name: torch.from_numpy(np.load(base / (name + ".npy"), allow_pickle=False)).cuda() for name in names}
    data["cot_map"] = torch.from_numpy(np.load(sidecar / "cot_map.npy", allow_pickle=False)).cuda()
    if data["x"].shape != (ROOT_ROWS, 13, 16, 16) or data["cot_map"].shape != (ROOT_ROWS, 1, 16, 16):
        raise RuntimeError("input tensor shape")
    if not (torch.isfinite(data["cot_map"]).all() and torch.isfinite(data["x"]).all() and
            torch.isfinite(data["c"]).all() and torch.isfinite(data["g"]).all()):
        raise RuntimeError("nonfinite input")
    train, val = torch.where(data["split"] == 0)[0], torch.where(data["split"] == 1)[0]
    if len(train) != TRAIN_ROWS or len(val) != VAL_ROWS or not (data["clear"] > 0).all():
        raise RuntimeError("split or target contract")
    if torch.max(torch.abs(data["kt"] * data["clear"] - data["ghi"])) >= 1e-8:
        raise RuntimeError("GHI reconstruction contract")
    out.mkdir(parents=True)
    contract = {"state": "EXPLORATORY_B1_CPP_PROVENANCE_UNRESOLVED", "test_used": False,
                "user_authorization": "2026-09-17 use current frozen COT model before CPP historical producer closure",
                "groups": list(GROUPS), "seed": 42, "epochs": 10, "batch": 2048, "lr": 0.0003,
                "optimizer": "AdamW(weight_decay=1e-4)", "loss": "mean(w*((pred_kt*clear-ghi)/1000)^2)",
                "base_pack_audit_sha256": sha(base / "audit.json"),
                "sidecar_audit_sha256": sha(sidecar / "audit.json"),
                "source_cpp_gate": "UNRESOLVED; exploratory results are not final scientific claims",
                "script_sha256": sha(__file__)}
    atomic(out / "execution_contract.json", contract)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BFloat16 unavailable")
    report = {"state": "RUNNING_EXPLORATORY_B1", **contract, "runs": {}}
    for group in GROUPS:
        seed(42)
        model = SpatialHead().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best, best_epoch, epochs = float("inf"), None, []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(10):
            model.train()
            generator = torch.Generator(device="cuda").manual_seed(42 + epoch)
            shuffled = train[torch.randperm(len(train), generator=generator, device="cuda")]
            total_loss, start = 0.0, time.monotonic()
            for ix in shuffled.split(2048):
                optimizer.zero_grad(set_to_none=True)
                pred = predict(model, data, ix, group)
                loss = ghi_loss(pred, data["ghi"][ix], data["clear"][ix], data["weight"][ix])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError("nonfinite gradients")
                optimizer.step()
                total_loss += float(loss.detach()) * len(ix)
            metrics, _ = evaluate(model, data, val, group, 2048)
            epoch_record = {"epoch": epoch, "train_weighted_normalized_ghi_mse": total_loss / len(train),
                            **metrics, "seconds": time.monotonic() - start}
            atomic(out / (group + "_epoch_%02d.json" % epoch), epoch_record)
            print(group, json.dumps(epoch_record), flush=True)
            epochs.append(epoch_record)
            if metrics["station_equal_ghi_rmse"] < best:
                best, best_epoch = metrics["station_equal_ghi_rmse"], epoch
                save(out / (group + "_best.pt"), {"model": model.state_dict(), "group": group,
                                                   "epoch": epoch, "contract": contract})
        checkpoint = torch.load(out / (group + "_best.pt"), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.cuda().eval()
        metrics, predictions = evaluate(model, data, val, group, 2048, keep=True)
        prediction_path = out / (group + "_validation_predictions.npz")
        with prediction_path.open("wb") as handle:
            np.savez(handle, pack_row=val.cpu().numpy(), pred_kt=predictions,
                     observed_ghi=data["ghi"][val].cpu().numpy(), clear_sky_ghi=data["clear"][val].cpu().numpy(),
                     station=data["station"][val].cpu().numpy(), lead=data["lead"][val].cpu().numpy())
        report["runs"][group] = {"best_epoch_0based": best_epoch, **metrics, "epochs": epochs,
                                  "checkpoint_sha256": sha(out / (group + "_best.pt")),
                                  "predictions_sha256": sha(prediction_path),
                                  "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}
        atomic(out / "progress.json", report)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    report["state"] = "COMPLETE_EXPLORATORY_B1_VALIDATION_PILOT"
    report["next"] = "Audit saved predictions; no automatic multi-seed, QC1, IFS, or test."
    atomic(out / "complete.json", report)
    print("SPATIAL_B1_EXPLORATORY_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unverified-cpp", action="store_true")
    args = parser.parse_args()
    if not args.allow_unverified_cpp:
        raise RuntimeError("requires explicit --allow-unverified-cpp authorization")
    main(args.root, args.out)
