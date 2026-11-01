#!/usr/bin/env python3
"""One allocated-A800 forward/backward profile for the exact AGRI+frozen-R step."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--s-norm", type=Path, required=True)
    p.add_argument("--r-pack", type=Path, required=True)
    p.add_argument("--r-checkpoint", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.report.exists():
        raise FileExistsError(a.report)
    if not torch.cuda.is_available():
        raise RuntimeError("Attempt13 joint profile requires an allocated CUDA device")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    sys.path.insert(0, str(a.repo))
    sys.path.insert(0, str(a.code_dir))
    sys.path.insert(0, str(a.code_dir.parent / "code_pipeline_attempt13"))

    from openstl.datasets import dataloader_hbagri
    from openstl.models import SimVP_Model
    from train_cot_repaired import COTUNet
    import cot_aux_patch_attempt13 as cotpatch

    with np.load(a.s_norm, allow_pickle=False) as z:
        agri_mean = np.asarray(z["agri_mean"], dtype=np.float32).copy()
        agri_std = np.asarray(z["agri_std"], dtype=np.float32).copy()
        input_mean = np.asarray(z["input_mean"], dtype=np.float32).copy()
        input_std = np.asarray(z["input_std"], dtype=np.float32).copy()

    def frozen_train_stats(self):
        self.mean = agri_mean.copy()
        self.std = agri_std.copy()
        self.input_mean = input_mean.copy()
        self.input_std = input_std.copy()

    cotpatch.patch_dataset_index(dataloader_hbagri.HBAGRIDataset)
    dataloader_hbagri.HBAGRIDataset._compute_stats = frozen_train_stats
    train_loader, _, _ = dataloader_hbagri.load_data(
        batch_size=a.batch, val_batch_size=1, data_root=str(a.data_root),
        num_workers=4, pre_seq_length=8, aft_seq_length=16,
        in_shape=[8, 16, 256, 256], index_csv=str(a.index),
        data_dir="", aux_dir="", drop_last=False,
    )
    batch = None
    for candidate in train_loader:
        if np.asarray(np.load(a.sidecar / "train_mask.npy", mmap_mode="r")[candidate[3].numpy()]).any():
            batch = candidate
            break
    if batch is None:
        raise RuntimeError("No CPP-supported batch found in train loader")
    x, y, aux, indices = [v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v for v in batch]
    model = SimVP_Model(
        in_shape=[8, 16, 256, 256], hid_S=32, hid_T=256,
        N_S=4, N_T=8, model_type="gSTA", drop_path=0.1,
        spatio_kernel_enc=3, spatio_kernel_dec=3, out_channels=13,
    ).cuda().train()
    r = COTUNet().cuda().eval()
    payload = torch.load(a.r_checkpoint, map_location="cpu")
    r.load_state_dict(payload["model"], strict=True)
    for param in r.parameters():
        param.requires_grad_(False)
    sidecar_target = np.load(a.sidecar / "train_target_physical.npy", mmap_mode="r", allow_pickle=False)
    sidecar_mask = np.load(a.sidecar / "train_mask.npy", mmap_mode="r", allow_pickle=False)
    rnorm = json.loads((a.r_pack / "norm.json").read_text(encoding="utf-8"))
    state = {
        "targets": {"train": sidecar_target, "val": sidecar_target},
        "masks": {"train": sidecar_mask, "val": sidecar_mask},
        "r_model": r,
        "agri_mean": torch.from_numpy(agri_mean).cuda().view(1, 1, 13, 1, 1),
        "agri_std": torch.from_numpy(agri_std).cuda().view(1, 1, 13, 1, 1),
        "geom_mean": torch.from_numpy(input_mean[13:]).cuda().view(1, 1, 3, 1, 1),
        "geom_std": torch.from_numpy(input_std[13:]).cuda().view(1, 1, 3, 1, 1),
        "r_mean": torch.tensor(rnorm["mean"], dtype=torch.float32, device="cuda").view(1, 16, 1, 1),
        "r_std": torch.tensor(rnorm["std"], dtype=torch.float32, device="cuda").view(1, 16, 1, 1),
    }
    holder = type("Holder", (), {})()
    object.__setattr__(holder, "_attempt13_cot_state", state)
    cotpatch.patch_dataset_index(dataloader_hbagri.HBAGRIDataset)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    # Match the checked OpenSTL SimVP.forward contract: two recursive 8-frame
    # predictions form the 16-frame forecast, with the first 8 target-time
    # solar-geometry frames appended between recursive blocks.
    predictions = []
    current = x
    for block_idx in range(2):
        prediction = model(current)
        if prediction.shape[1] != 8:
            raise RuntimeError(f"SimVP recursive block must output 8 frames, got {tuple(prediction.shape)}")
        predictions.append(prediction)
        if block_idx == 0:
            future_geom_8 = aux[:, 8:16]
            if future_geom_8.shape[1] != prediction.shape[1]:
                raise RuntimeError("Solar-geometry recursion slice is not aligned to first 8 predictions")
            current = torch.cat((prediction, future_geom_8), dim=2)
    pred = torch.cat(predictions, dim=1)
    if pred.shape != y.shape:
        raise RuntimeError(f"SimVP 8-to-16 profile shape mismatch: pred={tuple(pred.shape)}, target={tuple(y.shape)}")
    loss_agri = F.mse_loss(pred, y)
    # Profiling scale is deliberately small; the real train-only frozen lambda
    # is measured once and used identically in both arms before the training loop.
    loss_cot = cotpatch._cot_loss(holder, pred, aux, indices.cpu(), "train")
    if not torch.isfinite(loss_cot) or loss_cot.detach().item() <= 0:
        raise RuntimeError("Joint profile did not produce a finite nonzero CPP COT loss")
    (loss_agri + 1e-3 * loss_cot).backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not all(p.grad is None for p in r.parameters()):
        raise RuntimeError("Frozen COT R unexpectedly accumulated parameter gradients")
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite SimVP gradient in joint profile")
    report = {
        "state": "PASS_JOINT_AGRI_FROZEN_R_CPP_COT_STEP_PROFILE",
        "batch": a.batch,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "agri_mse": float(loss_agri.detach().cpu()),
        "cot_log1p_mse": float(loss_cot.detach().cpu()),
        "simvp_grad_finite": True,
        "frozen_r_parameter_grads_none": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        print(f"PROFILE_OOM {exc}", flush=True)
        raise SystemExit(20)
