"""Equal-budget MSE/temporal/spatial SimVP fine-tuning candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from simvp_supervision_losses_20261003 import (
    base_mse, loss_witness, spatial_gradient_l1, temporal_delta_l1,
)

STEPS = 4000
BATCH = 8
SEED = 42
EXPECTED_MANIFEST = "f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2"
EXPECTED_NORM = "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d"
EXPECTED_MODEL = "9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb"
EXPECTED_SIMVP = "2e0d280f7962a77f0c9b7e4cc817fed21d142c1436a12eb858e3234485313d16"
EXPECTED_DATASET_TRAINER = "9a34309b6af4d7debd2f7343c9266e9225df098e96fe699ab31b07c5d9188992"
EXPECTED_DATA_LOADER = "f4b4aab55101bde575f474d5daa4a184466c16399b401c2ac67bf666c5f28b91"
ARMS = {"A": "mse", "B": "temporal", "C": "spatial"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def seed_all(value=SEED):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def project_modules(source: Path):
    sys.path.insert(0, str(source))
    from train_monthly_simvp_20261002 import MonthlyDataset, model as build_model
    return MonthlyDataset, build_model


def load_contract(args):
    MonthlyDataset, build_model = project_modules(args.source)
    manifest = args.manifests / "hunan_8to16.csv"
    norm_path = args.manifests / "hunan_train_norm.json"
    checkpoint_sha = sha(args.checkpoint)
    assert sha(args.source / "triad_models.py") == EXPECTED_MODEL, "deployed backbone source mismatch"
    assert sha(args.source / "simvp.py") == EXPECTED_SIMVP, "deployed SimVP source mismatch"
    assert sha(args.source / "train_monthly_simvp_20261002.py") == EXPECTED_DATASET_TRAINER, "deployed dataset/training source mismatch"
    assert sha(args.source / "hunan_data.py") == EXPECTED_DATA_LOADER, "deployed frame reader source mismatch"
    assert sha(manifest) == EXPECTED_MANIFEST, "monthly split manifest hash mismatch"
    assert sha(norm_path) == EXPECTED_NORM, "train-only normalization hash mismatch"
    with norm_path.open() as f:
        norm = json.load(f)
    assert norm["manifest_sha256"] == EXPECTED_MANIFEST
    assert norm["validation_test_used"] is False
    dataset = MonthlyDataset(SimpleNamespace(
        source=str(args.source), root=str(args.root), cache=str(args.cache),
        manifests=str(args.manifests)), norm, "train")
    val = MonthlyDataset(SimpleNamespace(
        source=str(args.source), root=str(args.root), cache=str(args.cache),
        manifests=str(args.manifests)), norm, "val")
    assert len(dataset) == 35320 and len(val) == 3140
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    assert int(ckpt.get("epoch", -1)) == 10, "expected new monthly split epoch-10 best"
    assert meta.get("manifest_sha256") == EXPECTED_MANIFEST
    assert meta.get("normalization_sha256") == EXPECTED_NORM
    assert meta.get("model_code_sha256") == EXPECTED_MODEL
    assert meta.get("split_id") == "hunan_monthly_block_2d_buffer1d_seed42_v1"
    assert meta.get("test_payloads_read") == 0
    net = build_model(SimpleNamespace(source=str(args.source)))
    net.load_state_dict(ckpt["model"], strict=True)
    return dataset, val, norm, ckpt, net


def collate(dataset, rows, device="cuda"):
    samples = [dataset[i] for i in rows]
    batch = [torch.from_numpy(np.stack([s[j] for s in samples])).to(device) for j in range(4)]
    return batch[0], batch[1], batch[2], batch[3], [int(s[4]) for s in samples]


def season_index(month: int) -> int:
    return 0 if month in (12, 1, 2) else (1 if month in (3, 4, 5) else (2 if month in (6, 7, 8) else 3))


def eval_model(net, val, batch_size=8, workers=2):
    loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0,
                        prefetch_factor=1 if workers > 0 else None)
    sse = np.zeros((12, 16, 13), np.float64)
    count = np.zeros_like(sse)
    net.eval()
    with torch.no_grad():
        for x, g, y, v, months in loader:
            x, g, y, v = [z.cuda(non_blocking=True) for z in (x, g, y, v)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x, g)
            err = (pred.float() - y.float()).square() * v.float()
            sums = err.sum(dim=(0, 3, 4)).double().cpu().numpy()
            nums = v.sum(dim=(0, 3, 4)).double().cpu().numpy()
            for month, sm, nn in zip(months.tolist(), sums, nums):
                sse[month - 1] += sm
                count[month - 1] += nn
    monthly = np.sqrt(np.divide(sse.sum((1, 2)), count.sum((1, 2)),
                                out=np.full(12, np.nan), where=count.sum((1, 2)) > 0))
    seasons = {}
    groups = {"DJF": (11, 0, 1), "MAM": (2, 3, 4), "JJA": (5, 6, 7), "SON": (8, 9, 10)}
    for name, months in groups.items():
        ix = list(months)
        n = count[ix].sum()
        seasons[name] = float(np.sqrt(sse[ix].sum() / n))
    lead = np.sqrt(np.divide(sse.sum((0, 2)), count.sum((0, 2)),
                             out=np.full(16, np.nan), where=count.sum((0, 2)) > 0))
    channel = np.sqrt(np.divide(sse.sum((0, 1)), count.sum((0, 1)),
                                out=np.full(13, np.nan), where=count.sum((0, 1)) > 0))
    return {"val_equal_season_rmse": float(np.mean(list(seasons.values())),),
            "season_rmse": seasons, "monthly_rmse": {str(i + 1): float(v) for i, v in enumerate(monthly) if np.isfinite(v)},
            "lead_rmse": lead.tolist(), "channel_rmse": channel.tolist(),
            "sse_by_month_lead_channel": sse.tolist(), "count_by_month_lead_channel": count.tolist(),
            "val_windows": len(val)}


def calibration_rows(train, n=32):
    buckets = {m: [] for m in range(1, 13)}
    for i, row in enumerate(train.rows):
        month = int(row["BJT_start"][5:7])
        buckets[month].append(i)
    rng = np.random.default_rng(SEED)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < n:
        progressed = False
        for m in range(1, 13):
            if buckets[m] and len(selected) < n:
                selected.append(buckets[m].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("could not draw 32 distinct train windows")
    return selected


def loss_parts(pred, y, valid):
    l0 = base_mse(pred, y, valid)
    lt, nt = temporal_delta_l1(pred, y, valid)
    ls, ns = spatial_gradient_l1(pred, y, valid)
    return l0, lt, ls, nt, ns


def run_preflight(args):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    seed_all()
    ds, va, norm, ckpt, net = load_contract(args)
    witness = loss_witness("cuda")
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()[:BATCH * STEPS]
    order_sha = hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest()
    atomic_json(args.output / "fixed_training_rows.json", {"seed": SEED, "rows": order,
                "sha256": order_sha, "rows_per_arm": len(order), "test_payloads_read": 0})
    rows = calibration_rows(ds)
    ratios_t, ratios_s, records = [], [], []
    net.eval().cuda()
    for start in range(0, len(rows), BATCH):
        x, g, y, v, months = collate(ds, rows[start:start + BATCH])
        # Only calibrate auxiliary-loss gradient scales at the prediction
        # tensor. Keeping the full SimVP graph here needlessly retains the
        # large 256x256 encoder/translator activations and can OOM before the
        # actual training profile. No parameter gradients are needed.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x, g)
        pred = pred.detach().requires_grad_(True)
        l0, lt, ls, nt, ns = loss_parts(pred, y, v)
        m = v.bool()
        g0 = torch.autograd.grad(l0, pred, retain_graph=True)[0].float()
        gt = torch.autograd.grad(lt, pred, retain_graph=True)[0].float()
        gs = torch.autograd.grad(ls, pred)[0].float()
        n0 = torch.linalg.vector_norm(g0[m]).item()
        nt_norm = torch.linalg.vector_norm(gt[m]).item()
        ns_norm = torch.linalg.vector_norm(gs[m]).item()
        rt = n0 / max(nt_norm, 1e-12)
        rs = n0 / max(ns_norm, 1e-12)
        ratios_t.append(rt); ratios_s.append(rs)
        records.append({"months": months, "L0": float(l0), "Lt": float(lt), "Ls": float(ls),
                        "n_valid": int(m.sum()), "n_temporal_pairs": int(nt), "n_spatial_pairs": int(ns),
                        "gradient_ratio_t": rt, "gradient_ratio_s": rs})
        del x, g, y, v, pred, g0, gt, gs
    raw_t = 0.1 * float(np.median(ratios_t))
    raw_s = 0.1 * float(np.median(ratios_s))
    lam_t = float(np.clip(raw_t, 1e-4, 10.0))
    lam_s = float(np.clip(raw_s, 1e-4, 10.0))
    if lam_t != raw_t or lam_s != raw_s:
        raise RuntimeError(f"auxiliary gradient scale hit a safety clip: raw={raw_t, raw_s}")
    result = {"status": "PASS", "test_payloads_read": 0, "loss_witness": witness,
              "checkpoint_sha256": sha(args.checkpoint), "manifest_sha256": sha(args.manifests / "hunan_8to16.csv"),
              "normalization_sha256": sha(args.manifests / "hunan_train_norm.json"),
              "calibration_train_indices": rows, "calibration_indices_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
              "fixed_training_rows_sha256": order_sha,
              "lambda_temporal": lam_t, "lambda_spatial": lam_s,
              "ratio_medians": {"temporal": float(np.median(ratios_t)), "spatial": float(np.median(ratios_s))},
              "ratio_minmax": {"temporal": [float(min(ratios_t)), float(max(ratios_t))],
                               "spatial": [float(min(ratios_s)), float(max(ratios_s))]},
              "calibration_batches": records, "device": torch.cuda.get_device_name(0),
              "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    atomic_json(args.output / "preflight.json", result)
    print("PREFLIGHT_PASS", json.dumps({k: result[k] for k in ("lambda_temporal", "lambda_spatial", "ratio_medians", "device", "gpu_memory_gib")}), flush=True)


def arm_loss(arm, pred, y, v, lambdas):
    l0, lt, ls, nt, ns = loss_parts(pred, y, v)
    if arm == "A":
        loss = l0
    elif arm == "B":
        loss = l0 + lambdas["temporal"] * lt
    elif arm == "C":
        loss = l0 + lambdas["spatial"] * ls
    else:
        raise ValueError(arm)
    return loss, {"L0": float(l0.detach()), "Lt": float(lt.detach()), "Ls": float(ls.detach()),
                  "lambda_temporal": lambdas["temporal"], "lambda_spatial": lambdas["spatial"],
                  "temporal_pairs": int(nt), "spatial_pairs": int(ns)}


def save_checkpoint(path, net, opt, meta, step, best=None):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "metadata": meta,
                "step": step, "best": best}, tmp)
    tmp.replace(path)


def run_arm(args):
    if args.arm not in ARMS:
        raise ValueError(args.arm)
    preflight = json.loads((args.output / "preflight.json").read_text())
    assert preflight["status"] == "PASS" and preflight["test_payloads_read"] == 0
    lambdas = {"temporal": preflight["lambda_temporal"], "spatial": preflight["lambda_spatial"]}
    seed_all(SEED)
    ds, va, norm, ckpt, net = load_contract(args)
    run = args.output / ARMS[args.arm]
    run.mkdir(parents=True, exist_ok=False)
    state = ckpt["model"]
    del ckpt
    net.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4)
    meta = {"arm": args.arm, "loss": ARMS[args.arm], "initialization": "same monthly split seed42 epoch10 best",
            "checkpoint_sha256": sha(args.checkpoint), "manifest_sha256": EXPECTED_MANIFEST,
            "normalization_sha256": EXPECTED_NORM, "model_code_sha256": EXPECTED_MODEL,
            "seed": SEED, "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
            "train_windows": len(ds), "val_windows": len(va), "test_payloads_read": 0,
            "effective_batch": BATCH, "steps": STEPS, "lambda_temporal": lambdas["temporal"],
            "lambda_spatial": lambdas["spatial"], "pbs_jobid": os.environ.get("PBS_JOBID"),
            "device": torch.cuda.get_device_name(0), "source_code_sha256": sha(Path(__file__))}
    atomic_json(run / "run_metadata.json", meta)
    val0 = eval_model(net, va, batch_size=BATCH, workers=args.workers)
    val0.update({"step": 0, "arm": args.arm})
    atomic_json(run / "validation_step_0000.json", val0)
    save_checkpoint(run / "step_0000.pt", net, opt, meta, 0)

    # Verify real-batch memory/throughput before consuming the fixed budget.
    rows = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()[:BATCH * 4]
    profile = []
    net.train()
    for k in range(4):
        x, g, y, v, _ = collate(ds, rows[k * BATCH:(k + 1) * BATCH])
        torch.cuda.synchronize(); started = time.monotonic()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x, g); loss, parts = arm_loss(args.arm, pred, y, v, lambdas)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite profile loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize()
        profile.append({"step": k + 1, "seconds": time.monotonic() - started,
                        "loss": float(loss.detach()), "grad_norm_preclip": float(grad), **parts})
        del x, g, y, v, pred, loss
    torch.cuda.synchronize()
    profile_meta = {"arm": args.arm, "device": torch.cuda.get_device_name(0),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                    "samples_per_second_steady": BATCH / float(np.mean([r["seconds"] for r in profile[1:]])),
                    "profile_steps_not_counted": 4, "runs_concurrently": True, "records": profile}
    atomic_json(run / "profile.json", profile_meta)
    if profile_meta["peak_allocated_gib"] > 0.85 * profile_meta["total_gib"]:
        raise RuntimeError("profile exceeds 85% memory gate; all arms must use a revised shared batch contract")

    # Discard profiling updates; reload identical epoch-10 start and reset optimizer/RNG.
    net.load_state_dict(state, strict=True)
    del state
    seed_all(SEED)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4)
    order_data = json.loads((args.output / "fixed_training_rows.json").read_text())
    order = order_data["rows"]
    assert order_data["sha256"] == preflight["fixed_training_rows_sha256"]
    assert len(order) == BATCH * STEPS
    loader = DataLoader(ds, batch_size=BATCH, sampler=order, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=1 if args.workers > 0 else None,
                        generator=torch.Generator().manual_seed(SEED))
    net.train(); started_all = time.monotonic(); step = 0; recent = []; step_started = time.monotonic()
    for x, g, y, v, _ in loader:
        if step >= STEPS:
            break
        x, g, y, v = [z.cuda(non_blocking=True) for z in (x, g, y, v)]
        frac = step / STEPS
        lr = 1e-6 + 0.5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * frac))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x, g); loss, parts = arm_loss(args.arm, pred, y, v, lambdas)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step(); step += 1
        recent.append({"step": step, "loss": float(loss.detach()), "lr": lr,
                       "grad_norm_preclip": float(grad), "seconds": time.monotonic() - step_started, **parts})
        step_started = time.monotonic()
        if step % 100 == 0:
            atomic_json(run / "recent_steps.json", {"step": step, "last100": recent[-100:]})
            print("TRAIN_PROGRESS", json.dumps({"arm": args.arm, "step": step, "loss": recent[-1]["loss"],
                                                 "lr": lr, "samples_per_second": BATCH / max(recent[-1]["seconds"], 1e-9)}), flush=True)
        if step in (2000, 4000):
            net.eval()
            metrics = eval_model(net, va, batch_size=BATCH, workers=args.workers)
            metrics.update({"step": step, "arm": args.arm})
            atomic_json(run / f"validation_step_{step:04d}.json", metrics)
            save_checkpoint(run / f"step_{step:04d}.pt", net, opt, meta, step)
            net.train()
        del x, g, y, v, pred, loss
        if step == STEPS:
            break
    assert step == STEPS, f"expected {STEPS} updates, got {step}"
    atomic_json(run / "training_complete.json", {"status": "PASS", "arm": args.arm, "steps": step,
                 "wall_seconds": time.monotonic() - started_all, "test_payloads_read": 0,
                 "last_validation": json.loads((run / "validation_step_4000.json").read_text())["val_equal_season_rmse"]})
    print("ARM_COMPLETE", json.dumps({"arm": args.arm, "steps": step,
                                      "validation_step4000": json.loads((run / "training_complete.json").read_text())["last_validation"]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("preflight", "arm"), required=True)
    p.add_argument("--arm", choices=tuple(ARMS))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_arm(args)


if __name__ == "__main__":
    main()
