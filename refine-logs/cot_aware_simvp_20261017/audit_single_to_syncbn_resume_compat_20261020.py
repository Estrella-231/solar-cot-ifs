#!/usr/bin/env python3
"""Read-only compatibility audit for single-GPU OOF checkpoints in a SyncBN model.

This checks state-dict and AdamW optimizer loading only. It never reads image,
label, validation, test, or buffer payloads and never trains or writes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_one(checkpoint: Path, source: Path) -> dict:
    sys.path.insert(0, str(source))
    from triad_models import SimVPBackbone

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "metadata", "epoch", "step"}
    missing = required.difference(payload)
    if missing:
        raise AssertionError(f"checkpoint missing fields: {sorted(missing)}")

    base_model = SimVPBackbone(str(source))
    source_params = [(n, tuple(p.shape)) for n, p in base_model.named_parameters()]
    source_buffers = [(n, tuple(b.shape)) for n, b in base_model.named_buffers()]
    model_state = payload["model"]
    model_load = base_model.load_state_dict(model_state, strict=True)
    if model_load.missing_keys or model_load.unexpected_keys:
        raise AssertionError("single-GPU model state did not load strictly")

    sync_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
    sync_params = [(n, tuple(p.shape)) for n, p in sync_model.named_parameters()]
    sync_buffers = [(n, tuple(b.shape)) for n, b in sync_model.named_buffers()]
    if source_params != sync_params:
        raise AssertionError("parameter names/order/shapes changed during SyncBN conversion")
    if source_buffers != sync_buffers:
        raise AssertionError("buffer names/order/shapes changed during SyncBN conversion")
    sync_load = sync_model.load_state_dict(model_state, strict=True)
    if sync_load.missing_keys or sync_load.unexpected_keys:
        raise AssertionError("single-GPU model state did not load into SyncBN strictly")

    opt = torch.optim.AdamW(sync_model.parameters(), lr=1e-4, weight_decay=1e-4)
    opt.load_state_dict(payload["optimizer"])
    if len(opt.param_groups) != len(payload["optimizer"]["param_groups"]):
        raise AssertionError("optimizer parameter-group count changed")
    state_count = len(opt.state)
    if state_count == 0:
        raise AssertionError("checkpoint AdamW state is empty")
    state_tensor_mismatches = []
    for p, state in opt.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            t = state.get(key)
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != tuple(p.shape):
                state_tensor_mismatches.append(key)
    if state_tensor_mismatches:
        raise AssertionError("AdamW moment tensors do not match parameter shapes")

    meta = payload["metadata"]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": int(payload["epoch"]),
        "step": int(payload["step"]),
        "fold": meta.get("fold"),
        "seed": meta.get("seed"),
        "model_parameter_count": len(source_params),
        "model_buffer_count": len(source_buffers),
        "strict_single_model_load": True,
        "strict_syncbn_model_load": True,
        "syncbn_param_order_and_shapes_identical": True,
        "syncbn_buffer_order_and_shapes_identical": True,
        "adamw_state_entries_loaded": state_count,
        "adamw_moment_shapes_match": True,
        "data_payloads_read": 0,
        "training_performed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = [audit_one(c, a.source) for c in a.checkpoints]
    report = {"state": "PASS_SINGLE_GPU_TO_SYNCBN_CHECKPOINT_AND_ADAMW_LOAD",
              "scope": "read-only state-dict and optimizer load compatibility; not a DDP training-resume proof",
              "results": results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
