"""Matched Sili validation-only ablation of the extra low-COT false-COT penalty.
Both loss arms use the same fixed train/validation bank, seed, initialization,
minibatch order, optimizer, schedule, and step-0 candidate selection. Test is closed.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_sili_transport_thickness_pilot_20260927 import (
    TransportThicknessNet, make_input, prepare, evaluate,
)


def loss_fn(prediction, target, mask, use_lowcot_penalty):
    pred = prediction[:, 0]
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f"loss shape mismatch: {pred.shape}, {target.shape}, {mask.shape}")
    cot = torch.expm1(target)
    pred_cot = torch.expm1(pred)
    weight = 1 + 2 * (cot >= 30).float() + (cot < 5).float()
    huber = F.smooth_l1_loss(pred, target, reduction="none")
    result = (huber * weight * mask).sum() / torch.clamp((weight * mask).sum(), min=1)
    if use_lowcot_penalty:
        clear_ref = mask & (cot < 5)
        excess = F.relu(pred_cot - 10)
        result = result + 0.015 * (excess.square() * clear_ref).sum() / torch.clamp(clear_ref.sum(), min=1)
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_arm(label, use_penalty, tr, va, output, epochs, batch_size):
    arm_dir = output / label
    arm_dir.mkdir(parents=True, exist_ok=False)
    seed_all(42)
    model = TransportThicknessNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    started = time.time()

    # The zero-initialized residual is an explicit step-0 candidate. Its
    # validation score is measured with the same frozen evaluator and mask.
    step0 = evaluate(model, va, np.arange(512))
    best_score = float(step0["learned"]["thick_rmse"])
    best_name = "step0_zero_residual_transport"
    torch.save({"model": model.state_dict(), "epoch": 0, "seed": 42,
                "candidate": best_name, "val": step0}, arm_dir / "best.pt")
    (arm_dir / "step0.json").write_text(json.dumps(step0, indent=2) + "\n")
    stale = 0
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(512)
        total, steps = 0.0, 0
        for start in range(0, 512, batch_size):
            ids = order[start:start + batch_size]
            h, f = tr["h"][ids], tr["f"][ids]
            y, mask = tr["y"][ids], tr["mask"][ids]
            motion, adv = tr["motion"][ids], tr["adv"][ids]
            pred = model(make_input(h, f, adv, motion), adv)
            loss = loss_fn(pred, y, mask, use_penalty)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        val = evaluate(model, va, np.arange(512))
        score = float(val["learned"]["thick_rmse"])
        rec = {"epoch": epoch, "train_loss": total / steps,
               "selection_score_thick_rmse": score, "val": val,
               "elapsed_seconds": time.time() - started}
        logs.append(rec)
        with (arm_dir / "epochs.jsonl").open("a", encoding="utf-8") as fobj:
            fobj.write(json.dumps(rec) + "\n")
        print(json.dumps({"arm": label, **rec}), flush=True)
        if score < best_score:
            best_score = score
            best_name = f"learned_epoch_{epoch}"
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": 42,
                        "candidate": best_name, "val": val}, arm_dir / "best.pt")
        else:
            stale += 1
        if stale >= 8:
            break
    report = {
        "state": "COMPLETE_SILI_LOW_COT_PENALTY_ABLATION_ARM",
        "test_used": False, "split": "fixed 512/512 validation bank",
        "loss_arm": label, "extra_lowcot_false_cot_penalty": bool(use_penalty),
        "seed": 42, "epochs_run": len(logs), "selected_candidate": best_name,
        "selected_thick_rmse": best_score,
        "step0_thick_rmse": float(step0["learned"]["thick_rmse"]),
        "step0_val": step0,
        "selection": "minimum CPP-valid intersect transport-supported thick-COT RMSE; includes step-0 candidate",
        "cpp_is_retrieval_reference_not_independent_truth": True,
        "notes": "Exploratory validation only; overlapping windows; no test or GHI head used.",
    }
    (arm_dir / "training_complete.json").write_text(json.dumps(report, indent=2) + "\n")
    del opt, model
    torch.cuda.empty_cache()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    meta = json.loads((a.bank / "complete.json").read_text())
    if meta.get("state") != "COMPLETE_SILI_COT_DYNAMICS_BANK" or meta.get("test_used") is not False:
        raise ValueError("Unexpected or test-contaminated COT bank")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    print("PREPARE_FIXED_TRAIN", flush=True)
    tr = prepare(a.bank, "train", np.arange(512), "cuda")
    print("PREPARE_FIXED_VALIDATION", flush=True)
    va = prepare(a.bank, "val", np.arange(512), "cuda")
    results = []
    # Paired arms are serialized on one GPU and start from identical RNG states.
    results.append(run_arm("penalty_on_seed42", True, tr, va, a.output, a.epochs, a.batch_size))
    results.append(run_arm("penalty_off_seed42", False, tr, va, a.output, a.epochs, a.batch_size))
    summary = {"state": "COMPLETE_SILI_LOW_COT_PENALTY_ABLATION",
               "test_used": False, "results": results,
               "only_intended_difference": "extra low-COT false-COT penalty on vs off"}
    (a.output / "training_complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SILI_LOW_COT_PENALTY_ABLATION_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
