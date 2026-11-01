#!/usr/bin/env python3
"""Train date-fold SimVP models with a fixed epoch checkpoint, no heldout selection."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def load_base(source: Path):
    sys.path.insert(0, str(source))
    import train_monthly_simvp_20261002 as base
    return base


def build_train_dataset(base, a, norm):
    """The OOF fitting path intentionally cannot construct/read a val dataset."""
    return base.MonthlyDataset(a, norm, "train")


def make_epoch_loader(dataset, batch: int, workers: int, seed: int):
    """Match fold-0's per-epoch DataLoader and shuffle semantics exactly."""
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset, batch_size=batch, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers, pin_memory=True,
        persistent_workers=False,
    )


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(a) -> dict:
    import torch
    base = load_base(Path(a.source))
    norm_path = Path(a.manifests) / "hunan_train_norm.json"
    norm = json.loads(norm_path.read_text())
    assert norm["manifest_sha256"] == base.sha(Path(a.manifests) / "hunan_8to16.csv")
    assert norm["train_frames_sha256"] == base.sha(Path(a.manifests) / "train_frames.txt")
    profile = json.loads((Path(a.output) / "profile.json").read_text())
    batch = int(profile["selected_batch"])
    oof_ids = Path(a.manifests) / "fold_oof_ids.txt"
    assert oof_ids.is_file() and len(oof_ids.read_text().splitlines()) > 0
    output = Path(a.output) / "training"
    output.mkdir(exist_ok=False)
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    tr = build_train_dataset(base, a, norm)
    assert len(tr) > 0
    net = base.model(a)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    meta = {
        "region": "Hunan", "initialization": "random", "seed": a.seed,
        "fold": a.fold, "epochs_fixed": 12, "batch": batch,
        "train_samples": len(tr), "heldout_payload_read": False,
        "test_payloads_read": 0,
        "checkpoint_policy": "use_epoch_12_fixed; no heldout-fold scoring or checkpoint selection",
        "geometry_contract": "history_8_then_target_16_v2", "input_agri_channels": 13,
        "history_frames": 8, "target_frames": 16,
        "training_loss": "pooled-valid-pixel normalized AGRI MSE",
        "manifest_sha256": base.sha(Path(a.manifests) / "hunan_8to16.csv"),
        "fold_oof_ids_sha256": base.sha(oof_ids),
        "fold_manifest_audit_sha256": base.sha(Path(a.manifests) / "manifest_audit.json"),
        "normalization_sha256": base.sha(norm_path),
        "model_code_sha256": base.sha(Path(a.source) / "triad_models.py"),
        "data_code_sha256": base.sha(Path(a.source) / "hunan_data.py"),
        "simvp_code_sha256": base.sha(Path(a.source) / "simvp.py"),
        "training_code_sha256": base.sha(Path(__file__)),
        "pbs": os.environ.get("PBS_JOBID"), "node_cache": a.cache,
        "old_checkpoint_loaded": False,
    }
    base.write_json(output / "run_metadata.json", meta)
    step = 0
    for epoch in range(12):
        started = time.monotonic()
        net.train()
        losses = []
        # Recreate the loader per epoch to preserve fold-0's exact sample-order
        # semantics. Persistent workers changed RandomSampler ordering on reset.
        loader = make_epoch_loader(tr, batch, a.workers, a.seed + epoch)
        for k, (x, g, y, v, _) in enumerate(loader):
            x, g, y, v = [z.cuda(non_blocking=True) for z in (x, g, y, v)]
            progress = (epoch + k / len(loader)) / 12
            opt.param_groups[0]["lr"] = 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = base.loss_fn(net(x, g), y, v)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss))
            step += 1
            if step % 100 == 0:
                print("TRAIN_STEP", json.dumps({"fold": a.fold, "epoch": epoch + 1,
                                                  "step": step, "loss": losses[-1]}), flush=True)
        rec = {"fold": a.fold, "epoch": epoch + 1, "step": step,
               "train_loss_mean": float(np.mean(losses)),
               "elapsed_s": time.monotonic() - started,
               "heldout_payload_read": False, "checkpoint_selection": "fixed epoch schedule"}
        base.write_json(output / f"epoch_{epoch + 1:03d}.json", rec)
        payload = {"model": net.state_dict(), "optimizer": opt.state_dict(),
                   "metadata": meta, "epoch": epoch + 1, "step": step}
        torch.save(payload, output / "last.tmp")
        (output / "last.tmp").replace(output / "last.pt")
        if epoch == 11:
            torch.save(payload, output / "epoch_012.tmp")
            (output / "epoch_012.tmp").replace(output / "epoch_012.pt")
        print("EPOCH_END", json.dumps(rec), flush=True)
    result = {"state": "COMPLETE_OOF_SIMVP_FIXED_EPOCH_TRAINING", "fold": a.fold,
              "epochs": 12, "steps": step, "selected_checkpoint": "epoch_012.pt",
              "checkpoint_sha256": base.sha(output / "epoch_012.pt"),
              "heldout_payload_read": False, "test_payloads_read": 0}
    base.write_json(output / "training_complete.json", result)
    print("OOF_SIMVP_TRAIN_COMPLETE", json.dumps(result), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("normalize", "profile", "train"), required=True)
    for name in ("root", "source", "cache", "manifests", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, required=True)
    a = p.parse_args()
    Path(a.output).mkdir(parents=True, exist_ok=True)
    base = load_base(Path(a.source))
    if a.mode == "normalize":
        base.normalize(a)
    elif a.mode == "profile":
        norm = json.loads((Path(a.manifests) / "hunan_train_norm.json").read_text())
        assert norm["manifest_sha256"] == base.sha(Path(a.manifests) / "hunan_8to16.csv")
        base.profile(a, norm)
    else:
        train(a)


if __name__ == "__main__":
    main()
