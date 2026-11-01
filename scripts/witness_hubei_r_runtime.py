"""GPU-only runtime witness for the Hubei fixed-v2 COT R train/val pack.

This is a bounded forward/backward, save/reload, FP64 metric, and throughput
witness. It does not run epochs, select a checkpoint, or access test payloads.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--profile-batches", default="32,64,128,192")
    a = p.parse_args()
    pack, model_source, output = Path(a.pack), Path(a.model_source), Path(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; CPU fallback is intentionally disabled")
    status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED":
        raise ValueError("Unexpected or incomplete pack state")
    expected = json.loads((pack / "SHA256.json").read_text(encoding="utf-8"))
    actual = {name: sha(pack / name) for name in expected}
    if actual != expected:
        raise ValueError("Train/val pack hash mismatch")
    with (pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Non-train/val row found")
    norm = json.loads((pack / "norm.json").read_text(encoding="utf-8"))
    if norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("Normalization provenance violation")

    spec = importlib.util.spec_from_file_location("hubei_r_model_contract", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls, loss_fn = module.COTUNet, module.loss_fn
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    x_np = np.load(pack / "x_norm.npy", mmap_mode="r", allow_pickle=False)
    target_np = np.load(pack / "target.npy", mmap_mode="r", allow_pickle=False)
    mask_np = np.load(pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    train_ids = [i for i, r in enumerate(rows) if r["split"] == "train" and mask_np[i].any()]
    val_ids = [i for i, r in enumerate(rows) if r["split"] == "val" and mask_np[i].any()]
    if len(train_ids) < a.batch or not val_ids:
        raise ValueError("Insufficient supervised train/validation rows")
    ids = train_ids[:a.batch]
    x = torch.as_tensor(np.asarray(x_np[ids]).copy(), device=device)
    y = torch.as_tensor(np.asarray(target_np[ids]).copy() * 100.0, device=device)
    m = torch.as_tensor(np.asarray(mask_np[ids]).copy(), device=device)
    if not bool(m.reshape(len(ids), -1).any(1).all()):
        raise ValueError("Selected train batch contains an empty CPP mask")
    ylog = torch.log1p(y)

    torch.manual_seed(20261015)
    torch.cuda.manual_seed_all(20261015)
    model = model_cls().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model(x)
        loss = loss_fn(pred, ylog, m)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite mixed-precision loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    if not all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters()):
        raise RuntimeError("Missing or nonfinite gradient")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        probe = torch.as_tensor(np.asarray(x_np[[val_ids[0]]]).copy(), device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred_before = model(probe).float()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    state_bytes = buffer.getvalue()
    reloaded = model_cls().to(device).eval()
    reloaded.load_state_dict(torch.load(io.BytesIO(state_bytes), map_location=device, weights_only=True), strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred_after = reloaded(probe).float()
    reload_max_abs = float((pred_before - pred_after).abs().max().cpu())
    if reload_max_abs > 1e-6:
        raise RuntimeError(f"Save/reload mismatch: {reload_max_abs}")

    vindex = val_ids[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        val_pred_log = reloaded(probe).float().cpu().numpy().astype(np.float64)
    val_truth = np.asarray(target_np[[vindex]], dtype=np.float64) * 100.0
    val_mask = np.asarray(mask_np[[vindex]], dtype=bool)
    pred_cot = np.expm1(val_pred_log)
    err = pred_cot[val_mask] - val_truth[val_mask]
    if not err.size or not np.isfinite(err).all():
        raise RuntimeError("No finite validation pixels for independent FP64 metric")
    fp64 = {"valid_pixels": int(err.size), "rmse": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
            "mae": float(np.mean(np.abs(err), dtype=np.float64)),
            "bias": float(np.mean(err, dtype=np.float64))}

    # Bounded single-GPU profile with a preregistered >=12-updates/epoch cap.
    profile = []
    candidates = [int(v) for v in a.profile_batches.split(",") if v.strip()]
    for batch in candidates:
        if batch <= 0 or batch > len(train_ids) or len(train_ids) // batch < 12:
            profile.append({"batch": batch, "skipped": "would provide fewer than 12 updates per epoch"})
            continue
        bx = torch.as_tensor(np.asarray(x_np[train_ids[:batch]]).copy(), device=device)
        by = torch.as_tensor(np.asarray(target_np[train_ids[:batch]]).copy() * 100.0, device=device)
        bm = torch.as_tensor(np.asarray(mask_np[train_ids[:batch]]).copy(), device=device)
        bylog = torch.log1p(by)
        if not bool(bm.reshape(batch, -1).any(1).all()):
            raise ValueError(f"Profile batch {batch} contains an empty CPP mask")
        bench_model = model_cls().to(device).train()
        bench_opt = torch.optim.AdamW(bench_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        torch.cuda.reset_peak_memory_stats()
        def profile_step():
            bench_opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                bench_loss = loss_fn(bench_model(bx), bylog, bm)
            if not torch.isfinite(bench_loss):
                raise RuntimeError(f"Nonfinite profile loss at batch {batch}")
            scaler.scale(bench_loss).backward()
            scaler.unscale_(bench_opt)
            torch.nn.utils.clip_grad_norm_(bench_model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(bench_opt)
            scaler.update()
        torch.cuda.synchronize()
        for _ in range(3):
            profile_step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            profile_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        profile.append({"batch": batch, "updates_per_epoch": len(train_ids) // batch,
                        "samples_per_second": float(batch * 10 / elapsed),
                        "peak_memory_bytes": int(torch.cuda.max_memory_allocated())})
        del bench_model, bench_opt, scaler, bx, by, bm, bylog
        torch.cuda.empty_cache()
    valid_profiles = [item for item in profile if "samples_per_second" in item]
    if not valid_profiles:
        raise RuntimeError("No eligible single-GPU profile batch")
    best_speed = max(item["samples_per_second"] for item in valid_profiles)
    selected_batch = min(item["batch"] for item in valid_profiles
                         if item["samples_per_second"] >= 0.97 * best_speed)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "state": "HUBEI_R_RUNTIME_WITNESS_PASSED_NO_FORMAL_TRAINING",
        "test_payloads_read": 0,
        "pack_hashes_match": True,
        "pack_hashes": actual,
        "train_rows": len(train_ids), "val_rows": len(val_ids), "batch_samples": len(ids),
        "batch_valid_cpp_pixels": int(m.sum().cpu()), "loss_fp16_autocast": float(loss.detach().float().cpu()),
        "gradient_norm_before_clip": float(grad_norm.detach().float().cpu()),
        "save_reload_max_abs_prediction_difference": reload_max_abs,
        "single_validation_sample_fp64_metrics_not_skill_claim": fp64,
        "profile_candidates": profile, "profile_selected_batch_within_3pct_best": selected_batch,
        "profile_scope": "single-card microprofile; selected batch remains provisional until formal trainer is separately audited",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "torch": torch.__version__, "model_source_sha256": sha(model_source),
        "note": "Bounded runtime witness only; no epoch training, checkpoint selection, test access, or performance claim.",
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
