"""Train paired GHI heads for an oracle-only future-COT information ceiling.

The oracle pack replaces only future COT context with frozen-R retrievals from
observed future AGRI. The observed images are not deployable inputs; this is a
validation diagnostic to localize the remaining bottleneck.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from train_future_ghi_heads import IntervalHead, cot_for_arm, load_station, score, sha


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1792)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pack_manifest_path = args.pack / "pack_manifest.json"
    manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_ORACLE_FUTURE_COT_GHI_PACK" or manifest.get("oracle_only") is not True:
        raise ValueError("oracle-only pack is incomplete or not explicitly declared")
    if manifest.get("deployable") is not False or manifest.get("future_observed_agri_used_as_input") is not True:
        raise ValueError("oracle provenance flags are inconsistent")
    if manifest.get("future_cpp_used_as_input") is not False:
        raise ValueError("CPP is not an allowed oracle input")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer closure is not certified")
    if manifest.get("forecast_bank_oof_fold_count") != 5 or manifest.get("train_forecast_oof") is not True:
        raise ValueError("forecast AGRI must retain five-fold OOF provenance")

    args.output.mkdir(parents=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    pack_sha = sha(pack_manifest_path)
    arm_description = {"A": "no COT; same forecast AGRI and geometry",
                       "AH": "observed-history COT only; future COT zeroed",
                       "AC": "observed-history plus frozen-R COT from observed future AGRI; oracle only"}
    report = {"state": "RUNNING_ORACLE_FUTURE_COT_GHI_SEED",
              "scope": "validation information-ceiling diagnostic; not deployable forecast evidence",
              "oracle_only": True, "deployable": False,
              "seed": args.seed, "stations": ["sili", "zhujia"],
              "epochs": args.epochs, "batch_size": args.batch_size,
              "architecture": "established IntervalHead unchanged; 3x3, three frames T-30/T-15/T",
              "arms": arm_description,
              "primary_contrasts": ["AC minus AH: oracle future-COT increment over observed-history COT",
                                    "AC minus A: all-COT oracle versus no-COT head"],
              "future_observed_agri_used_as_input": True,
              "future_cpp_used_as_input": False,
              "test_payload_opened": False, "buffer_payload_opened": False,
              "pack_manifest_sha256": pack_sha,
              "source_sha256": sha(Path(__file__))}
    (args.output / "seed_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summaries = []
    for station in ("sili", "zhujia"):
        train = load_station(args.pack, "train", station, device, manifest)
        val = load_station(args.pack, "val", station, device, manifest)
        n = len(train["ghi"])
        if n // args.batch_size < 12:
            raise ValueError(f"insufficient updates per epoch for {station}: {n}")
        seed_all(args.seed)
        base = IntervalHead().to(device)
        initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
        rng = np.random.default_rng(args.seed)
        orders = [rng.permutation(n) for _ in range(args.epochs)]
        # Both arms receive the same initialization and row order.
        for arm in ("A", "AH", "AC"):
            run = args.output / station / arm
            run.mkdir(parents=True)
            model = IntervalHead().to(device)
            model.load_state_dict(initial, strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
            cot = cot_for_arm(train, arm)
            best_rmse = float("inf")
            best_state = None
            history = []
            updates = 0
            started = time.monotonic()
            for epoch, order in enumerate(orders):
                model.train()
                weighted_loss = 0.0
                for lo in range(0, n, args.batch_size):
                    ix = torch.as_tensor(order[lo:lo + args.batch_size], device=device)
                    pred_kt = model(train["agri"][ix], train["geometry"][ix], cot[ix])
                    loss = (((pred_kt * train["clear"][ix] - train["ghi"][ix]) / 1000.0) ** 2).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss {station}/{arm}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    weighted_loss += float(loss.detach()) * len(ix)
                scheduler.step()
                pred = score(model, val, arm, args.batch_size)
                truth = val["ghi"].detach().cpu().numpy().astype(np.float64)
                val_rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
                history.append({"epoch": epoch + 1, "updates": updates,
                                "train_scaled_mse": weighted_loss / n,
                                "val_rmse_wm2": val_rmse})
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    print(json.dumps({"seed": args.seed, "station": station, "arm": arm,
                                      **history[-1]}, separators=(",", ":")), flush=True)
            if best_state is None:
                raise RuntimeError("no validation checkpoint selected")
            model.load_state_dict(best_state, strict=True)
            pred = score(model, val, arm, args.batch_size)
            truth = val["ghi"].detach().cpu().numpy().astype(np.float64)
            best_rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
            prediction_path = run / "validation_predictions.npz"
            np.savez_compressed(prediction_path, prediction_ghi_wm2=pred,
                truth_ghi_wm2=truth,
                keys=np.asarray([json.dumps(x, sort_keys=True) for x in val["keys"]]))
            checkpoint_path = run / "best.pt"
            torch.save({"model": best_state, "arm": arm, "station": station,
                        "seed": args.seed, "pack_manifest_sha256": pack_sha,
                        "oracle_only": True}, checkpoint_path)
            metrics = {"station": station, "seed": args.seed, "arm": arm,
                "n_train": n, "n_val": len(truth), "best_val_rmse_wm2": val_rmse,
                "mae_wm2": float(np.mean(np.abs(pred - truth))),
                "bias_wm2": float(np.mean(pred - truth)), "updates": updates,
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint_sha256": sha(checkpoint_path),
                "predictions_sha256": sha(prediction_path),
                "test_payload_opened": False, "oracle_only": True}
            (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            (run / "training_curve.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            summaries.append(metrics)
            print("ORACLE_ARM_COMPLETE " + json.dumps(metrics, separators=(",", ":")), flush=True)
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
        del base
    report["state"] = "COMPLETE_ORACLE_FUTURE_COT_GHI_SEED"
    report["runs"] = summaries
    (args.output / "seed_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ORACLE_SEED_COMPLETE " + json.dumps({"seed": args.seed, "runs": len(summaries),
          "oracle_only": True, "test_payload_opened": False}), flush=True)


if __name__ == "__main__":
    main()
