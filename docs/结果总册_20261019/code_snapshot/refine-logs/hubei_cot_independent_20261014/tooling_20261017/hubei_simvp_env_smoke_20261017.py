#!/usr/bin/env python3
"""One allocated-GPU runtime witness using one train-only Hubei sequence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    stage_receipt = json.loads((a.root / "stage_verification_receipt.json").read_text())
    fold_receipt = json.loads((a.root / "control/pbs_fold_input_receipt.json").read_text())
    if stage_receipt.get("status") != "PBS_TRAINVAL_STAGE_SHA256_AND_INDEX_REBASE_PASSED_TEST_CLOSED":
        raise RuntimeError("train/val stage receipt not passed")
    if stage_receipt.get("test_sequences") != 0 or stage_receipt.get("test_or_buffer_payloads_staged") != 0:
        raise RuntimeError("test/buffer closure receipt failed")
    if fold_receipt.get("status") != "PBS_FOLD_INPUTS_REBASED_TEST_CLOSED":
        raise RuntimeError("fold inputs receipt not passed")
    if a.fold not in {int(x["fold"]) for x in fold_receipt.get("folds", [])}:
        raise RuntimeError(f"fold {a.fold} missing from input receipt")

    control = a.root / "control" / f"fold_{a.fold}"
    index = control / "simvp_trainval_index_pbs.csv"
    norm_path = control / "norm/simvp_train_only_norm.npz"
    if not index.is_file() or not norm_path.is_file():
        raise FileNotFoundError("fold index or train-only normalization missing")
    with index.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    allowed = {"train", "val"}
    if any(row["split"] not in allowed for row in rows):
        raise RuntimeError("fold index contains non-train/val rows")
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise RuntimeError("no train rows available for smoke witness")

    norm_file = np.load(norm_path, allow_pickle=False)
    norm = {k: norm_file[k].copy() for k in norm_file.files}
    sys.path.insert(0, str(a.repo))
    from openstl.datasets.dataloader_hbagri import HBAGRIDataset
    from openstl.methods.simvp import SimVP

    def use_frozen_train_statistics(self) -> None:
        self.mean = norm["agri_mean"].copy()
        self.std = norm["agri_std"].copy()
        self.input_mean = norm["input_mean"].copy()
        self.input_std = norm["input_std"].copy()

    HBAGRIDataset._compute_stats = use_frozen_train_statistics
    ds = HBAGRIDataset(
        data_root=str(control), pre_seq_length=8, aft_seq_length=16,
        temporal_stride=1, split="train", data_name="hbagri", index_csv=str(index),
    )
    if len(ds) != len(train_rows):
        raise RuntimeError(f"dataset/index train count mismatch: {len(ds)} != {len(train_rows)}")

    x, y, aux = ds[0]
    if x.shape != (8, 16, 256, 256) or y.shape != (16, 13, 256, 256) or aux.shape != (24, 3, 256, 256):
        raise RuntimeError(f"unexpected real batch shapes: x={tuple(x.shape)}, y={tuple(y.shape)}, aux={tuple(aux.shape)}")
    if not (torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(aux).all()):
        raise RuntimeError("non-finite value in train-only sample")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside PBS allocation")
    cfg = {
        "method": "simvp", "dataname": "hbagri", "data_name": "hbagri",
        "pre_seq_length": 8, "aft_seq_length": 16, "total_length": 24,
        "in_shape": [8, 16, 256, 256], "out_channels": 13,
        "model_type": "gSTA", "hid_S": 32, "hid_T": 256,
        "N_S": 4, "N_T": 8, "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3, "drop_path": 0.1, "mlp_ratio": 8.0,
        "drop": 0.0, "metrics": ["mse", "mae"],
        "steps_per_epoch": 1, "test_mean": norm["agri_mean"],
        "test_std": norm["agri_std"], "save_dir": str(a.out),
        "seed": 42, "epoch": 50, "opt": "adamw", "lr": 1e-3,
        "weight_decay": 1e-4, "sched": "cosine", "min_lr": 1e-5,
        "warmup_epoch": 5,
    }
    torch.manual_seed(42)
    model = SimVP(**cfg).to(device)
    model.train()
    xb, yb, ab = (t.unsqueeze(0).to(device, non_blocking=True) for t in (x, y, aux))
    pred = model(xb, batch_aux=ab)
    if pred.shape != yb.shape or not torch.isfinite(pred).all():
        raise RuntimeError(f"invalid forward output {tuple(pred.shape)}")
    loss = F.mse_loss(pred, yb)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError("missing or non-finite model gradients")
    model.eval()
    with torch.no_grad():
        baseline = model(xb, batch_aux=ab).detach()

    a.out.mkdir(parents=True, exist_ok=False)
    ckpt_path = a.out / "smoke_state.pt"
    torch.save(model.state_dict(), ckpt_path)
    reloaded = SimVP(**cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(state, strict=True)
    reloaded.eval()
    with torch.no_grad():
        restored = reloaded(xb, batch_aux=ab)
    if not torch.allclose(baseline, restored, rtol=0, atol=1e-6):
        raise RuntimeError("save/reload prediction mismatch")

    sample_paths = train_rows[0]["data_paths"].split("|")
    result = {
        "state": "HUBEI_SIMVP_PBS_GPU_ENV_REAL_TRAIN_BATCH_FORWARD_BACKWARD_RELOAD_PASS",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "split_payload_read": "train only, one sequence",
        "fold": a.fold,
        "train_rows_in_index": len(train_rows),
        "sample_first_frame": Path(sample_paths[0]).stem,
        "shapes": {"x": list(x.shape), "y": list(y.shape), "aux": list(aux.shape), "pred": list(pred.shape)},
        "loss": float(loss.detach().cpu()),
        "gradient_tensors_finite": len(grads),
        "reload_max_abs_diff": float((baseline - restored).abs().max().cpu()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "stage_manifest_sha256": stage_receipt.get("source_manifest_sha256"),
        "fold_index_sha256": sha256(index),
        "normalization_sha256": sha256(norm_path),
        "env_spec_sha256": sha256(a.env),
        "checkpoint_sha256": sha256(ckpt_path),
    }
    report_path = a.out / "ENV_SMOKE_REPORT.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("HUBEI_SIMVP_ENV_GPU_WITNESS_PASS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
