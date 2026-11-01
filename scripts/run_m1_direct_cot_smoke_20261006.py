"""GPU smoke on eight new-contract, train-only windows; no validation/test data."""
import argparse
import csv
import hashlib
import inspect
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from direct_cot_model_20261005 import DirectCOTForecaster, masked_log_cot_mse


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--hunan-data-code", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--cpp-labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the PBS GPU job")
    sys.path.insert(0, str(a.hunan_data_code.parent))
    import hunan_data

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda")
    norm = json.loads(a.norm.read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"][:13], np.float32)[:, None, None]
    std = np.asarray(norm["std"][:13], np.float32)[:, None, None]
    with a.windows.open(newline="", encoding="utf-8-sig") as f:
        windows = list(csv.DictReader(f))
    with np.load(a.cpp_labels, allow_pickle=False) as z:
        cot = np.asarray(z["cot"], np.float32)
        mask = np.asarray(z["mask"], bool)
        ids = [x.decode() for x in z["seq_ids"]]
    if ids != [r["seq_id"] for r in windows]:
        raise RuntimeError("CPP labels and smoke window order differ")
    if cot.shape != mask.shape or cot.shape != (len(windows), 16, 1, 256, 256):
        raise RuntimeError(f"unexpected label/mask shapes {cot.shape}/{mask.shape}")

    histories, geos = [], []
    for row in windows:
        hist, hist_geo = [], []
        future_geo = []
        for stamp in row["history_utc"].split("|"):
            rel = f"{stamp[:4]}/{stamp[:6]}/{stamp[:8]}/{stamp}.npy"
            agri, valid, geom = hunan_data.load_frame(a.data_root, rel)
            hist.append(np.where(valid, (agri - mean) / std, 0.).astype(np.float32))
            hist_geo.append(geom.astype(np.float32))
        for stamp in row["future_utc"].split("|"):
            bjt = datetime.strptime(stamp, "%Y%m%d%H%M") + timedelta(hours=8)
            aux_path = hunan_data.resolve_aux(a.data_root, bjt)
            with np.load(aux_path, allow_pickle=False) as aux:
                g = np.stack([aux[k] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
            if not np.isfinite(g).all():
                raise ValueError(f"non-finite known target geometry: {aux_path}")
            future_geo.append(g)
        histories.append(np.stack(hist))
        geos.append(np.concatenate((np.stack(hist_geo), np.stack(future_geo)), axis=0))
    x = torch.from_numpy(np.stack(histories)).to(device)
    g = torch.from_numpy(np.stack(geos)).to(device)
    y = torch.from_numpy(cot).to(device)
    m = torch.from_numpy(mask).to(device)
    if x.shape != (len(windows), 8, 13, 256, 256) or g.shape != (len(windows), 24, 3, 256, 256):
        raise RuntimeError(f"unexpected inputs {tuple(x.shape)} {tuple(g.shape)}")
    if not bool(torch.isfinite(x).all() and torch.isfinite(g).all() and torch.isfinite(y).all()):
        raise RuntimeError("nonfinite smoke tensor")

    torch.cuda.reset_peak_memory_stats()
    model = DirectCOTForecaster(width=96).to(device)
    params = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        pred = model(x[:1], g[:1])
        if pred.shape != (1, 16, 1, 256, 256) or not bool(torch.isfinite(pred).all()):
            raise RuntimeError("forward shape or finite-value check failed")
        loss_a = masked_log_cot_mse(pred, y[:1], m[:1])
        y_perturbed = y[:1].clone()
        y_perturbed[~m[:1]] = 10000.
        loss_b = masked_log_cot_mse(pred, y_perturbed, m[:1])
        mask_invariance_delta = float((loss_a - loss_b).abs().cpu())
        if mask_invariance_delta != 0.0:
            raise RuntimeError(f"masked invalid target perturbation changed loss: {mask_invariance_delta}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    a.output.mkdir(parents=True, exist_ok=False)
    events_path = a.output / "events.jsonl"
    t0 = time.time()
    initial_loss = None
    final_loss = None
    for step in range(1, a.steps + 1):
        idx0 = (2 * (step - 1)) % len(windows)
        ids_batch = [(idx0 + j) % len(windows) for j in range(a.batch)]
        optimizer.zero_grad(set_to_none=True)
        out = model(x[ids_batch], g[ids_batch])
        loss = masked_log_cot_mse(out, y[ids_batch], m[ids_batch])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at update {step}")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"nonfinite gradient at update {step}")
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 10 == 0 or step == a.steps:
            event = {"step": step, "loss": final_loss,
                     "lr": optimizer.param_groups[0]["lr"],
                     "gpu_memory_allocated": torch.cuda.memory_allocated(),
                     "gpu_peak_memory_allocated": torch.cuda.max_memory_allocated(),
                     "elapsed_seconds": time.time() - t0}
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

    model.eval()
    ckpt = a.output / "smoke_last.pt"
    torch.save({"state_dict": model.state_dict(), "seed": a.seed,
                "steps": a.steps, "params": params}, ckpt)
    reloaded = DirectCOTForecaster(width=96).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    reloaded.load_state_dict(state["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        p1 = model(x[:1], g[:1])
        p2 = reloaded(x[:1], g[:1])
    reload_max_abs = float((p1 - p2).abs().max().cpu())
    if reload_max_abs > 1e-7:
        raise RuntimeError(f"checkpoint reload changed GPU prediction: {reload_max_abs}")
    status = {"state": "COMPLETE_M1_REAL_TRAIN_SMOKE", "test_used": False,
              "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
              "windows": ids, "windows_count": len(windows), "updates": a.steps,
              "batch": a.batch, "seed": a.seed, "model": "DirectCOTForecaster-width96",
              "parameter_count": params, "input_contract": "history AGRI [B,8,13,256,256] plus known geometry [B,24,3,256,256]",
              "future_agri_loaded": False, "future_cot_used_as_input": False,
              "cpp_archive_sha256": sha(a.cpp_labels), "windows_sha256": sha(a.windows),
              "norm_sha256": sha(a.norm), "model_code_sha256": sha(Path(__file__).with_name("direct_cot_model_20261005.py")),
              "initial_loss": initial_loss, "final_training_loss": final_loss,
              "mask_valid_fraction": float(m.float().mean().cpu()),
              "mask_invariance_delta": mask_invariance_delta,
              "reload_max_abs_gpu": reload_max_abs,
              "elapsed_seconds": time.time() - t0,
              "gpu_name": torch.cuda.get_device_name(0),
              "gpu_peak_memory_allocated": torch.cuda.max_memory_allocated(),
              "criterion": "equal-weight per-lead masked MSE(log1p(COT))",
              "scope": "engineering smoke only; no validation metric or skill claim"}
    (a.output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
