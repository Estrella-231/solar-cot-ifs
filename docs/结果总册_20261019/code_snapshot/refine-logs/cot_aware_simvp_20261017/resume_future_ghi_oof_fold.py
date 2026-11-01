#!/usr/bin/env python3
"""Resume a validated OOF SimVP fold from its last atomic epoch checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from train_monthly_simvp_oof_20261019 import (
    build_train_dataset,
    load_base,
    make_epoch_loader,
)


def validate_resume(checkpoint: dict, saved_meta: dict, expected: dict) -> None:
    assert checkpoint["metadata"] == saved_meta
    assert checkpoint["epoch"] == expected["checkpoint_epoch"]
    assert checkpoint["step"] == expected["checkpoint_step"]
    for key in ("fold", "seed", "batch", "train_samples", "manifest_sha256",
                "normalization_sha256", "model_code_sha256", "data_code_sha256",
                "simvp_code_sha256"):
        assert saved_meta[key] == expected[key], (key, saved_meta.get(key), expected[key])
    assert saved_meta["initialization"] == "random"
    assert saved_meta["heldout_payload_read"] is False
    assert saved_meta["test_payloads_read"] == 0


def run(a) -> dict:
    import torch

    base = load_base(Path(a.source))
    output = Path(a.output)
    training = output / "training"
    norm_path = Path(a.manifests) / "hunan_train_norm.json"
    manifest_path = Path(a.manifests) / "hunan_8to16.csv"
    frames_path = Path(a.manifests) / "train_frames.txt"
    norm = json.loads(norm_path.read_text())
    assert norm["manifest_sha256"] == base.sha(manifest_path)
    assert norm["train_frames_sha256"] == base.sha(frames_path)
    profile = json.loads((output / "profile.json").read_text())
    batch = int(profile["selected_batch"])
    saved_meta = json.loads((training / "run_metadata.json").read_text())
    checkpoint_path = Path(a.resume_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "fold": a.fold, "seed": a.seed, "batch": batch,
        "checkpoint_epoch": a.checkpoint_epoch,
        "checkpoint_step": a.checkpoint_step,
        "train_samples": int(saved_meta["train_samples"]),
        "manifest_sha256": base.sha(manifest_path),
        "normalization_sha256": base.sha(norm_path),
        "model_code_sha256": base.sha(Path(a.source) / "triad_models.py"),
        "data_code_sha256": base.sha(Path(a.source) / "hunan_data.py"),
        "simvp_code_sha256": base.sha(Path(a.source) / "simvp.py"),
    }
    validate_resume(checkpoint, saved_meta, expected)
    assert not (training / "training_complete.json").exists()
    assert len(list(training.glob("epoch_*.json"))) == a.checkpoint_epoch

    recovery = {
        "state": "RESUME_FROM_ATOMIC_EPOCH_CHECKPOINT",
        "fold": a.fold, "seed": a.seed, "resume_checkpoint": str(checkpoint_path),
        "resume_checkpoint_sha256": base.sha(checkpoint_path),
        "interrupted_training_code_sha256": saved_meta["training_code_sha256"],
        "recovery_code_sha256": base.sha(Path(__file__)),
        "resume_epoch": a.checkpoint_epoch, "resume_step": a.checkpoint_step,
        "remaining_epochs": 12 - a.checkpoint_epoch,
        "workers": a.workers,
        "rng_note": "model/dropout RNG restarted deterministically from seed+1000003*resume_epoch; sampler remains seed+epoch",
        "heldout_payload_read": False, "test_payloads_read": 0,
    }
    base.write_json(training / "resume_recovery.json", recovery)

    seed = a.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    train_ds = build_train_dataset(base, a, norm)
    assert len(train_ds) == expected["train_samples"]
    net = base.model(a)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    net.load_state_dict(checkpoint["model"], strict=True)
    opt.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint["step"])

    # The interrupted process did not save per-step RNG state. Restart dropout
    # randomness from a deterministic epoch boundary while preserving model,
    # optimizer, epoch schedule, and the exact epoch-specific sampler order.
    continuation_seed = a.seed + 1_000_003 * a.checkpoint_epoch
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    torch.manual_seed(continuation_seed)
    torch.cuda.manual_seed_all(continuation_seed)

    for epoch in range(a.checkpoint_epoch, 12):
        started = time.monotonic()
        net.train()
        losses = []
        loader = make_epoch_loader(train_ds, batch, a.workers, a.seed + epoch)
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
                print("RESUME_TRAIN_STEP", json.dumps({"fold": a.fold, "epoch": epoch + 1,
                                                        "step": step, "loss": losses[-1]}), flush=True)
        rec = {"fold": a.fold, "epoch": epoch + 1, "step": step,
               "train_loss_mean": float(np.mean(losses)),
               "elapsed_s": time.monotonic() - started,
               "heldout_payload_read": False, "checkpoint_selection": "fixed epoch schedule",
               "resumed_from_epoch": a.checkpoint_epoch}
        base.write_json(training / f"epoch_{epoch + 1:03d}.json", rec)
        payload = {"model": net.state_dict(), "optimizer": opt.state_dict(),
                   "metadata": saved_meta, "epoch": epoch + 1, "step": step}
        torch.save(payload, training / "last.tmp")
        (training / "last.tmp").replace(training / "last.pt")
        if epoch == 11:
            torch.save(payload, training / "epoch_012.tmp")
            (training / "epoch_012.tmp").replace(training / "epoch_012.pt")
        print("RESUME_EPOCH_END", json.dumps(rec), flush=True)

    result = {"state": "COMPLETE_OOF_SIMVP_FIXED_EPOCH_TRAINING",
              "fold": a.fold, "epochs": 12, "steps": step,
              "selected_checkpoint": "epoch_012.pt",
              "checkpoint_sha256": base.sha(training / "epoch_012.pt"),
              "heldout_payload_read": False, "test_payloads_read": 0,
              "resumed_from_epoch": a.checkpoint_epoch,
              "recovery_rng_protocol": recovery["rng_note"]}
    base.write_json(training / "training_complete.json", result)
    print("OOF_SIMVP_RESUMED_TRAIN_COMPLETE", json.dumps(result), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("root", "source", "cache", "manifests", "output", "resume-checkpoint"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint-epoch", type=int, required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
