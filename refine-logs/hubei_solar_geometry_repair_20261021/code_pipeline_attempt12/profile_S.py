#!/usr/bin/env python3
"""One-real-batch GPU profile for the frozen Hubei SimVP contract."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(os.environ.get("SIMVP_REPO", "/home/Data_Pool_3/chenyi/GHI/SimVPv2"))
ROOT = Path("/home/Data_Pool/zjnu/hubei_cot_independent_20261016")
INDEX = ROOT / "manifests/simvp_frozen_trainval_20261016.csv"
NORM = ROOT / "norm/simvp_train_only_norm_20261016.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--index-csv", type=Path, default=INDEX)
    parser.add_argument("--norm-npz", type=Path, default=NORM)
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/zjnu_hubei_simvp_stage_20261016"))
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    batch_size = args.batch_size
    index_path, norm_path, data_root = args.index_csv, args.norm_npz, args.data_root
    if batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    OUT = args.report_json or ROOT / f"logs/profile_simvp_batch{batch_size}_20261016.json"
    CKPT = args.checkpoint or ROOT / f"runs/profile_simvp_batch{batch_size}_20261016/state.pt"
    if OUT.exists() or CKPT.exists():
        raise FileExistsError("Profile outputs already exist; use a new run id")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profile")
    sys.path.insert(0, str(REPO))
    from openstl.datasets import dataloader_hbagri
    from openstl.models import SimVP_Model

    def frozen_stats(self) -> None:
        with np.load(norm_path, allow_pickle=False) as z:
            self.mean = np.asarray(z["agri_mean"], dtype=np.float32).copy()
            self.std = np.asarray(z["agri_std"], dtype=np.float32).copy()
            self.input_mean = np.asarray(z["input_mean"], dtype=np.float32).copy()
            self.input_std = np.asarray(z["input_std"], dtype=np.float32).copy()
        if self.mean.shape != (13,) or self.input_mean.shape != (16,):
            raise ValueError("Unexpected frozen normalization dimensions")

    dataloader_hbagri.HBAGRIDataset._compute_stats = frozen_stats
    train_loader, _, _ = dataloader_hbagri.load_data(
        batch_size=batch_size, val_batch_size=1,
        data_root=str(data_root), num_workers=4, pre_seq_length=8,
        aft_seq_length=16, in_shape=[8, 16, 256, 256],
        index_csv=str(index_path), data_dir="", aux_dir="", drop_last=False)
    started = time.perf_counter()
    batch = next(iter(train_loader))
    load_seconds = time.perf_counter() - started
    x, y, aux = [value.to("cuda", non_blocking=True) for value in batch]
    if tuple(x.shape) != (batch_size, 8, 16, 256, 256) or tuple(y.shape) != (batch_size, 16, 13, 256, 256):
        raise ValueError(f"Unexpected batch shapes: {tuple(x.shape)}, {tuple(y.shape)}")
    model = SimVP_Model(in_shape=[8, 16, 256, 256], hid_S=32, hid_T=256,
                        N_S=4, N_T=8, model_type="gSTA", drop_path=0.1,
                        spatio_kernel_enc=3, spatio_kernel_dec=3,
                        out_channels=13).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def predict() -> torch.Tensor:
        pred = []
        current = x
        for block in range(2):
            block_pred = model(current)
            pred.append(block_pred)
            if block == 0:
                current = torch.cat([block_pred, aux[:, 8:16]], dim=2)
        return torch.cat(pred, dim=1)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    step_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    pred = predict()
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    grads_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not (torch.isfinite(loss).item() and grads_finite):
        raise FloatingPointError("Non-finite loss or gradient in real-batch witness")
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_start
    peak_bytes = torch.cuda.max_memory_allocated()

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    model.eval()
    with torch.no_grad():
        expected = predict().detach().cpu()
    reloaded = SimVP_Model(in_shape=[8, 16, 256, 256], hid_S=32, hid_T=256,
                           N_S=4, N_T=8, model_type="gSTA", drop_path=0.1,
                           spatio_kernel_enc=3, spatio_kernel_dec=3,
                           out_channels=13).cuda().eval()
    state = torch.load(CKPT, map_location="cuda", weights_only=True)
    reloaded.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = []
        current = x
        for block in range(2):
            block_pred = reloaded(current)
            actual.append(block_pred)
            if block == 0:
                current = torch.cat([block_pred, aux[:, 8:16]], dim=2)
        actual = torch.cat(actual, dim=1).cpu()
    reload_max_abs = float((expected - actual).abs().max())
    if reload_max_abs > 1e-5:
        raise AssertionError(f"Checkpoint reload mismatch: {reload_max_abs}")
    report = {
        "status": "REAL_TRAIN_BATCH_FORWARD_BACKWARD_RELOAD_PASS_TEST_CLOSED",
        "device": torch.cuda.get_device_name(0),
        "visible_cuda_devices": torch.cuda.device_count(),
        "index_sha256": sha256(index_path), "norm_sha256": sha256(norm_path),
        "checkpoint": str(CKPT), "checkpoint_sha256": sha256(CKPT),
        "train_batch_shapes": {"input": list(x.shape), "target": list(y.shape), "aux": list(aux.shape)},
        "batch_size": batch_size,
        "data_load_seconds": load_seconds, "forward_backward_optimizer_step_seconds": step_seconds,
        "peak_allocated_bytes": int(peak_bytes), "loss_normalized_mse": float(loss.detach().cpu()),
        "gradients_finite": grads_finite, "reload_prediction_max_abs": reload_max_abs,
        "test_or_buffer_payloads_read": 0,
        "interpretation": "one-step engineering profile only; not formal training and no forecast skill claim",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
