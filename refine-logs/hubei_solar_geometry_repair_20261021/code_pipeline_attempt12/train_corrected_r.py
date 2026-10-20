"""Train the independent Hubei COT-R on corrected-UTC train/validation only."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
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


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_step(model, optimizer, scaler, x, y, mask, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_fn(model(x), y, mask)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite Hubei COT loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Missing/nonfinite gradient in Hubei R")
    scaler.step(optimizer)
    scaler.update()
    return loss.detach(), grad_norm.detach()


@torch.no_grad()
def evaluate(model, x, y, mask, indices, batch, loss_fn, save_predictions=False):
    model.eval()
    loss_sum = torch.zeros((), device=x.device, dtype=torch.float64)
    pixel_count = torch.zeros((), device=x.device, dtype=torch.float64)
    predictions, targets, masks = [], [], []
    for ids in indices.split(batch):
        with torch.autocast("cuda", dtype=torch.float16):
            pred_log = model(x[ids]).float()
        if not torch.isfinite(pred_log).all():
            raise RuntimeError("Nonfinite validation prediction")
        count = mask[ids].sum()
        if count <= 0:
            raise RuntimeError("Empty-mask sample reached R validation")
        loss_sum += loss_fn(pred_log, y[ids], mask[ids]).double() * count.double()
        pixel_count += count.double()
        if save_predictions:
            predictions.append(torch.expm1(pred_log).cpu().numpy().astype(np.float64))
            targets.append(torch.expm1(y[ids].float()).cpu().numpy().astype(np.float64))
            masks.append(mask[ids].cpu().numpy().astype(bool))
    result = {"valid_pixel_huber_log1p": float((loss_sum / pixel_count).cpu()),
              "valid_cpp_pixels": int(pixel_count.cpu())}
    if save_predictions:
        pred = np.concatenate(predictions, axis=0)
        truth = np.concatenate(targets, axis=0)
        valid = np.concatenate(masks, axis=0)
        err = pred[valid] - truth[valid]
        truth_valid = truth[valid]
        ss_tot = np.sum((truth_valid - truth_valid.mean(dtype=np.float64)) ** 2, dtype=np.float64)
        result.update({"reference": "cpp_16_fixed_v2 retrieval, not independent truth",
                       "rmse_cot": float(np.sqrt(np.mean(err ** 2, dtype=np.float64))),
                       "mae_cot": float(np.mean(np.abs(err), dtype=np.float64)),
                       "bias_cot": float(np.mean(err, dtype=np.float64)),
                       "r2_cot": float(1.0 - np.sum(err ** 2, dtype=np.float64) / ss_tot) if ss_tot > 0 else None})
        result["predictions_cot"] = pred
        result["targets_cot"] = truth
        result["mask"] = valid
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--allow-variable-support", action="store_true",
                    help="For date-fold OOF packs, derive supervision counts from this fold's explicit mask instead of the original full-pack counts")
    a = p.parse_args()
    pack, model_source, output = Path(a.pack), Path(a.model_source), Path(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Hubei R training requires the allocated GPU; CPU fallback disabled")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output is not empty; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "run.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
    assert status.get("solar_geometry_corrected") is True, "corrected geometry pack required"
    if status.get("state") != "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED":
        raise ValueError("Pack is not the corrected-UTC closed train/val pack")
    expected_hashes = json.loads((pack / "SHA256.json").read_text(encoding="utf-8"))
    actual_hashes = {name: sha(pack / name) for name in expected_hashes}
    if expected_hashes != actual_hashes:
        raise ValueError("Train/validation source pack SHA mismatch")
    with (pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(row["split"] not in ("train", "val") for row in rows):
        raise ValueError("Pack contains an unexpected split")
    norm = json.loads((pack / "norm.json").read_text(encoding="utf-8"))
    if norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("Train-only input normalization contract failed")
    if len(norm.get("input_channels", [])) != 16:
        raise ValueError("Expected the frozen 16-channel Hubei input contract")

    spec = importlib.util.spec_from_file_location("hubei_correctutc_cot_model", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls, loss_fn = module.COTUNet, module.loss_fn
    x_np = np.load(pack / "x_norm.npy", allow_pickle=False)
    target_scaled = np.load(pack / "target.npy", allow_pickle=False)
    mask_np = np.load(pack / "mask.npy", allow_pickle=False)
    if x_np.shape != (len(rows), 16, 16, 16) or target_scaled.shape != (len(rows), 1, 16, 16):
        raise ValueError("Input/target shape mismatch")
    if mask_np.shape != target_scaled.shape or mask_np.dtype != np.bool_:
        raise ValueError("Mask shape/dtype mismatch")
    if not np.isfinite(x_np).all() or not np.isfinite(target_scaled).all():
        raise ValueError("Nonfinite packed data")
    valid_rows = [i for i in range(len(rows)) if mask_np[i].any()]
    train_ids_np = np.asarray([i for i in valid_rows if rows[i]["split"] == "train"], dtype=np.int64)
    val_ids_np = np.asarray([i for i in valid_rows if rows[i]["split"] == "val"], dtype=np.int64)
    all_train_rows = sum(row["split"] == "train" for row in rows)
    all_val_rows = sum(row["split"] == "val" for row in rows)
    if not a.allow_variable_support and (len(train_ids_np) != 2811 or len(val_ids_np) != 259):
        raise ValueError("Supervision support changed from the audited pack")
    if a.allow_variable_support and (len(train_ids_np) == 0 or len(val_ids_np) == 0):
        raise ValueError("Date-fold R pack has no valid train or fixed-validation supervision")
    if a.batch <= 0 or len(train_ids_np) // a.batch < 12:
        raise ValueError("Batch violates the preregistered >=12 updates/epoch rule")

    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_all(a.seed)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(target_scaled * 100.0).to(device)
    if ((y < 0) | (y > 100)).any():
        raise ValueError("COT target out of frozen [0,100] range")
    y = torch.log1p(y)
    mask = torch.from_numpy(mask_np).to(device)
    train_ids = torch.as_tensor(train_ids_np, device=device)
    val_ids = torch.as_tensor(val_ids_np, device=device)
    meta = {"model": "hubei_R_station16_solar_corrected_v1", "seed": a.seed,
            "input_channels": norm["input_channels"], "target": "log1p(COT), COT in [0,100]",
            "loss": "valid-pixel masked Huber delta=0.1", "optimizer": "AdamW lr=1e-3 wd=1e-4",
            "gradient_clip": 5.0, "epochs_max": a.epochs, "patience": a.patience,
            "pack_samples": len(rows), "all_train_rows_for_bank": all_train_rows,
            "all_val_rows_for_bank": all_val_rows, "train_supervision_rows": len(train_ids_np),
            "val_supervision_rows": len(val_ids_np), "train_no_valid_cpp_rows": all_train_rows-len(train_ids_np),
            "val_no_valid_cpp_rows": all_val_rows-len(val_ids_np), "batch": a.batch,
            "updates_per_epoch_floor": len(train_ids_np)//a.batch, "test_used": False,
            "allow_variable_support": bool(a.allow_variable_support),
            "normalization": norm, "source_pack_hashes": actual_hashes,
            "pack_status_sha256": sha(pack / "pack_status.json"),
            "model_source_sha256": sha(model_source), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "pbs_job_id": os.environ.get("PBS_JOBID"),
            "reference": "cpp_16_fixed_v2 retrieval, not independent truth",
            "empty_cpp_rows_policy": "excluded from R loss/selection only; retained for train/val AGRI COT-bank inference"}
    atomic(output / "run_metadata.json", meta)

    # Fixed eight-sample overfit gate, then discard these weights.
    overfit_ids = train_ids[:8]
    overfit_x, overfit_y, overfit_m = x[overfit_ids], y[overfit_ids], mask[overfit_ids]
    seed_all(a.seed)
    smoke_model = model_cls().to(device).train()
    smoke_opt = torch.optim.AdamW(smoke_model.parameters(), lr=1e-3, weight_decay=1e-4)
    smoke_scaler = torch.cuda.amp.GradScaler()
    initial = float(evaluate(smoke_model, x, y, mask, overfit_ids, 8, loss_fn)["valid_pixel_huber_log1p"])
    for _ in range(400):
        run_step(smoke_model, smoke_opt, smoke_scaler, overfit_x, overfit_y, overfit_m, loss_fn)
    final = float(evaluate(smoke_model, x, y, mask, overfit_ids, 8, loss_fn)["valid_pixel_huber_log1p"])
    overfit = {"initial_huber": initial, "final_huber": final, "updates": 400,
               "indices": overfit_ids.cpu().tolist(), "passed": final <= 0.5 * initial}
    atomic(output / "smoke_overfit.json", overfit)
    if not overfit["passed"]:
        raise RuntimeError("Fixed-sample overfit gate failed; formal R training stopped")
    del smoke_model, smoke_opt, smoke_scaler
    torch.cuda.empty_cache()

    # Formal fit starts from a fresh initialization; smoke weights are never reused.
    seed_all(a.seed)
    model = model_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    best, best_epoch, stale = float("inf"), -1, 0
    start_time = time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        shuffled = train_ids[torch.randperm(len(train_ids), device=device)]
        loss_sum, pixels = 0.0, 0
        for current in shuffled.split(a.batch):
            loss, _ = run_step(model, optimizer, scaler, x[current], y[current], mask[current], loss_fn)
            count = int(mask[current].sum())
            loss_sum += float(loss.float()) * count
            pixels += count
        val = evaluate(model, x, y, mask, val_ids, a.batch, loss_fn)
        score = val["valid_pixel_huber_log1p"]
        if not np.isfinite(score) or pixels <= 0:
            raise RuntimeError("Invalid train/validation epoch statistic")
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "best_val": best,
                      "best_epoch": best_epoch, "metadata": meta,
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            tmp = output / (name + ".tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(output / name)
        record = {"epoch": epoch, "train_huber": loss_sum/pixels,
                  "validation_huber": score, "validation_valid_cpp_pixels": val["valid_cpp_pixels"],
                  "best_epoch": best_epoch, "elapsed_seconds": time.monotonic()-start_time}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        atomic(output / "status.json", {"state": "TRAINING", **record})
        print("EPOCH", json.dumps(record), flush=True)
        if stale >= a.patience:
            break

    ckpt = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    fp64 = evaluate(model, x, y, mask, val_ids, a.batch, loss_fn, save_predictions=True)
    pred = fp64.pop("predictions_cot")
    truth = fp64.pop("targets_cot")
    valid = fp64.pop("mask")
    np.save(output / "val_prediction_cot.npy", pred.astype(np.float32), allow_pickle=False)
    np.save(output / "val_reference_cpp_cot.npy", truth.astype(np.float32), allow_pickle=False)
    np.save(output / "val_valid_mask.npy", valid, allow_pickle=False)
    val_rows = [rows[i] for i in val_ids_np]
    with (output / "val_keys.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pack_index", "time_bjt", "split"])
        writer.writerows([[r["index"], r["time_bjt"], r["split"]] for r in val_rows])
    atomic(output / "validation_fp64.json", {"status": "VALIDATION_ONLY_NOT_TEST",
           "metrics": fp64, "model_selection_metric": "valid-pixel Huber on log1p(COT)",
           "best_epoch": int(ckpt["best_epoch"]), "best_validation_huber": float(ckpt["best_val"]),
           "test_payloads_read": 0, "reference": "cpp_16_fixed_v2 retrieval, not independent truth"})
    atomic(output / "status.json", {"state": "COMPLETE_VALIDATION_ONLY_TEST_CLOSED",
           "best_epoch": int(ckpt["best_epoch"]), "best_validation_huber": float(ckpt["best_val"]),
           "epochs_completed": len((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()),
           "test_payloads_read": 0})
    print("HUBEI_R_TRAIN_COMPLETE_VALIDATION_ONLY", flush=True)


if __name__ == "__main__":
    main()
