#!/usr/bin/env python3
"""One-batch CUDA compatibility and frozen-A identity witness."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from head_candidate_v0 import LocalGHIHead
from signed_residual_head import SignedResidualHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = False
    manifest = json.loads((a.pack / "pack_manifest.json").read_text())
    rows = list(csv.DictReader((a.pack / "rows.csv").open(encoding="utf-8")))
    indices = [i for i, r in enumerate(rows) if r["station"] == "sili" and r["split"] == "val"][:64]
    am = np.asarray(manifest["AGRI_norm_mean"], np.float32)[None, :, None, None]
    ast = np.asarray(manifest["AGRI_norm_std"], np.float32)[None, :, None, None]
    raw = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)[indices]
    valid = np.load(a.pack / "agri_valid.npy", mmap_mode="r", allow_pickle=False)[indices].astype(bool)
    x = np.where(valid, (raw - am) / ast, 0).astype(np.float32)
    g = np.array(np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)[indices], dtype=np.float32)
    c0 = np.load(a.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)[indices].astype(np.float32)
    c = (c0 - np.float32(manifest["COT_norm_mean"])) / np.float32(manifest["COT_norm_std"])
    clear = np.asarray([float(rows[i]["clear"]) for i in indices], np.float32)
    device = torch.device("cuda")
    X, G, C, CL = [torch.from_numpy(v).to(device) for v in (x, g, c, clear)]
    base_path = a.baseline / "A_best.pt"
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    base = LocalGHIHead("A", 2).to(device)
    base.load_state_dict(state)
    base.eval()
    with torch.inference_mode():
        expected = base(X, G) * CL
    saved_a = np.load(a.baseline / "A_val_prediction_ghi_wm2.npy", allow_pickle=False)[:len(indices)]
    baseline_delta = float(np.max(np.abs(saved_a - expected.cpu().numpy())))
    results = {}
    for variant in ("AGRI", "AGRI_COT"):
        model = SignedResidualHead(variant).to(device)
        model.load_frozen_a(state)
        pred, frozen_base, residual = model(X, G, C, CL)
        err = float((pred - expected).abs().max().item())
        assert err == 0.0 and float(residual.abs().max().item()) == 0.0
        loss = ((pred - (expected + 1.0)) / 1000).square().mean()
        loss.backward()
        assert all(p.grad is None for p in model.features.parameters())
        results[variant] = {"step_zero_max_abs_error": err,
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "frozen_A_gradients_absent": True}
    torch.cuda.synchronize()
    assert results["AGRI"]["trainable_parameters"] == results["AGRI_COT"]["trainable_parameters"]
    print(json.dumps({"state": "PASS", "device": torch.cuda.get_device_name(),
                      "cudnn_enabled": torch.backends.cudnn.enabled,
                      "saved_A_max_abs_delta_wm2_first64": baseline_delta,
                      "variants": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
