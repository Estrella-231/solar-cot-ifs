"""Paired fixed-budget GHI heads for four forecast-COT sources.

The pack producer must certify common H/V keys and train-only normalization.
Each station, arm, and seed is trained independently for exactly 30 epochs;
the validation set is scored only after the final epoch and never selects a
checkpoint. No true future AGRI/CPP or test/buffer payload is accepted here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from single_frame_ghi_head import SingleFrameGHIHead

ARMS = ("none", "simvp_r", "p_c", "p_xc")
STATIONS = ("sili", "zhujia")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(station_dir: Path, split: str, arm: str):
    # The pack builder writes common AGRI/geometry/labels and each COT arm
    # together under <station>/<split>; there is no separate "common" tree.
    common = station_dir / split
    agri = np.load(common / "agri.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(common / "geometry.npy", mmap_mode="r", allow_pickle=False)
    ghi = np.load(common / "ghi.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(common / "clear.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(common / "cot" / arm / f"{split}.npy", mmap_mode="r", allow_pickle=False)
    keys_path = common / "keys.jsonl"
    keys = keys_path.read_bytes()
    n = len(ghi)
    expected = ((n, 13, 3, 3), (n, 2, 3, 3), (n, 1, 3, 3), (n,), (n,))
    arrays = (agri, geometry, cot, ghi, clear)
    if tuple(x.shape for x in arrays) != expected:
        raise ValueError(f"{station_dir.name}/{split}/{arm}: array shape contract failed")
    if len(keys.splitlines()) != n:
        raise ValueError(f"{station_dir.name}/{split}: keys row count differs")
    for x in arrays:
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite array in {station_dir.name}/{split}/{arm}")
    if (clear <= 0).any() or (ghi < 0).any():
        raise ValueError(f"invalid GHI/clear-sky values in {station_dir.name}/{split}")
    return {
        "agri": torch.from_numpy(np.asarray(agri, dtype=np.float32).copy()).cuda(),
        "geometry": torch.from_numpy(np.asarray(geometry, dtype=np.float32).copy()).cuda(),
        "cot": torch.from_numpy(np.asarray(cot, dtype=np.float32).copy()).cuda(),
        "ghi": torch.from_numpy(np.asarray(ghi, dtype=np.float32).copy()).cuda(),
        "clear": torch.from_numpy(np.asarray(clear, dtype=np.float32).copy()).cuda(),
        "keys": np.frombuffer(keys, dtype=np.uint8).copy(),
    }


def score(model, data, batch_size: int):
    model.eval()
    out = []
    with torch.inference_mode():
        for lo in range(0, len(data["ghi"]), batch_size):
            sl = slice(lo, min(lo + batch_size, len(data["ghi"])))
            kt = model(data["agri"][sl], data["geometry"][sl], data["cot"][sl])
            out.append((kt.double() * data["clear"][sl].double()).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def run_head(station: str, arm: str, seed: int, train, val,
             output: Path, pack_sha: str, epochs: int, batch_size: int):
    run = output / station / arm / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    # Same seed => byte-identical initial state and epoch orders across arms.
    seed_all(seed)
    model = SingleFrameGHIHead().cuda()
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_hash = hashlib.sha256(b"".join(
        k.encode() + v.contiguous().numpy().tobytes() for k, v in sorted(init_state.items())
    )).hexdigest()
    model.load_state_dict(init_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    n = len(train["ghi"])
    steps = math.ceil(n / batch_size)
    history, updates = [], 0
    start = time.monotonic()
    for epoch in range(epochs):
        # The same seed/epoch permutation is used for every COT arm.
        order = np.random.default_rng(seed + epoch).permutation(n)
        model.train()
        loss_sum = 0.0
        for lo in range(0, n, batch_size):
            ids = torch.as_tensor(order[lo:lo + batch_size], device="cuda", dtype=torch.long)
            pred_kt = model(train["agri"][ids], train["geometry"][ids], train["cot"][ids])
            resid = (pred_kt * train["clear"][ids] - train["ghi"][ids]) / 1000.0
            loss = resid.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {station}/{arm}/seed{seed}/epoch{epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            updates += 1
        scheduler.step()
        rec = {"epoch": epoch + 1, "train_scaled_mse": loss_sum / n,
               "updates": updates, "fixed_budget": True}
        history.append(rec)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps({"station": station, "arm": arm, "seed": seed, **rec}), flush=True)

    # Fixed epoch-30 checkpoint: validation has not selected/changed training.
    model.eval()
    pred = score(model, val, batch_size)
    truth = val["ghi"].double().cpu().numpy()
    delta = pred - truth
    checkpoint = run / f"epoch_{epochs:03d}.pt"
    torch.save({"model": model.state_dict(), "station": station, "arm": arm,
                "seed": seed, "epochs": epochs, "initial_state_sha256": init_hash,
                "pack_manifest_sha256": pack_sha}, checkpoint)
    keys = np.frombuffer(val["keys"], dtype=np.uint8)
    pred_file = run / "validation_predictions.npz"
    np.savez_compressed(pred_file, prediction_ghi_wm2=pred, truth_ghi_wm2=truth,
                        keys_jsonl_bytes=keys)
    metrics = {"station": station, "arm": arm, "seed": seed,
               "n_train": n, "n_val": len(truth), "epochs": epochs,
               "updates": updates, "rmse_wm2": float(np.sqrt(np.mean(delta ** 2))),
               "mae_wm2": float(np.mean(np.abs(delta))), "bias_wm2": float(np.mean(delta)),
               "checkpoint_sha256": sha(checkpoint), "prediction_sha256": sha(pred_file),
               "elapsed_seconds": time.monotonic() - start}
    atomic_json(run / "metrics.json", metrics)
    atomic_json(run / "training_curve.json", {"history": history})
    return metrics, init_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--station", choices=(*STATIONS, "both"), default="both")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GHI-head training")
    if a.epochs != 30 or a.batch_size != 512:
        raise ValueError("production contract is fixed at 30 epochs and effective batch 512")
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError("duplicate seeds")
    manifest_path = a.pack / "pack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_PACK":
        raise ValueError("direct-COT GHI pack is not complete")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer payload closure not certified")
    if manifest.get("future_observed_agri_used_as_input") is not False or manifest.get("cpp_used_as_input") is not False:
        raise ValueError("future observed AGRI/CPP input is forbidden")
    if manifest.get("arms") != list(ARMS):
        raise ValueError("four forecast-COT arms differ from the fixed contract")
    if manifest.get("train_partition") != "H" or manifest.get("validation_partition") != "V":
        raise ValueError("this experiment requires H-only head training and V-only evaluation")
    if manifest.get("split_support_audit_sha256") is None:
        raise ValueError("U/H/V 24-frame support audit is not bound to the pack")
    if manifest.get("normalization_and_common_key_contract") != "PASS_TRAIN_ONLY_NORMALIZATION_AND_COMMON_KEYS":
        raise ValueError("H-only normalization/common-key audit is not certified")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    pack_sha = sha(manifest_path)
    stations = STATIONS if a.station == "both" else (a.station,)
    all_runs, init_hashes, reference_init = [], {}, {}
    for station in stations:
        sd = a.pack / station
        tr_common = load_split(sd, "train", "none")
        va_common = load_split(sd, "val", "none")
        # The non-COT source is required to be exactly zero after its train-only normalization.
        if torch.count_nonzero(tr_common["cot"]).item() or torch.count_nonzero(va_common["cot"]).item():
            raise ValueError(f"no-COT input is not the neutral zero tensor for {station}")
        for arm in ARMS:
            tr = tr_common if arm == "none" else load_split(sd, "train", arm)
            va = va_common if arm == "none" else load_split(sd, "val", arm)
            if not (np.array_equal(tr["keys"], tr_common["keys"])
                    and np.array_equal(va["keys"], va_common["keys"])):
                raise ValueError(f"common key mismatch for {station}/{arm}")
            for seed in a.seeds:
                rec, ih = run_head(station, arm, seed, tr, va, a.output, pack_sha, a.epochs, a.batch_size)
                pair_key = f"{station}/{seed}"
                if pair_key in reference_init and reference_init[pair_key] != ih:
                    raise RuntimeError(f"paired model initialization differs for {pair_key}/{arm}")
                reference_init.setdefault(pair_key, ih)
                init_hashes[f"{station}/{seed}/{arm}"] = ih
                all_runs.append(rec)
    report = {"state": "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_HEADS",
              "arms": list(ARMS), "stations": list(stations), "seeds": list(a.seeds),
              "epochs": a.epochs, "effective_batch": a.batch_size,
              "architecture": "single-frame 3x3; 13 AGRI + 2 geometry + one COT field; all arms use same COT-capable head",
              "normalization_and_common_key_contract": manifest.get("normalization_and_common_key_contract"),
              "pack_manifest_sha256": pack_sha, "initial_state_sha256": init_hashes,
              "trainer_sha256": sha(Path(__file__)),
              "head_definition_sha256": sha(Path(__file__).with_name("single_frame_ghi_head.py")),
              "runs": all_runs, "test_payload_opened": False, "buffer_payload_opened": False,
              "future_observed_agri_used_as_input": False, "cpp_used_as_input": False}
    atomic_json(a.output / "run_manifest.json", report)
    print(json.dumps({"state": report["state"], "runs": len(all_runs), "test_payload_opened": False}), flush=True)


if __name__ == "__main__":
    main()
