#!/usr/bin/env python3
"""Infer both completed attempt13 arms and score validation COT on common CPP support."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str]) -> None:
    print("RUN " + json.dumps(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    a = p.parse_args()

    pair = a.attempt / "PAIR_COMPLETE_attempt13.json"
    prepared = a.attempt / "prepared_inputs.json"
    r_val = a.base / "R_runs_attempt12" / "full"
    r_source = a.base / "code_pipeline_attempt12"
    r_checkpoint = a.base / "R_runs_attempt12" / "full" / "best.pt"
    r_norm = a.base / "prepared_attempt12" / "R_packs" / "full" / "norm.json"
    index = a.base / "prepared_attempt12" / "indexes" / "full.csv"
    s_norm = a.base / "prepared_attempt12" / "norms" / "full.npz"
    for path in (pair, prepared, r_val / "val_keys.csv",
                 r_val / "val_prediction_cot.npy", r_val / "val_reference_cpp_cot.npy",
                 r_val / "val_valid_mask.npy", r_source / "train_cot_repaired.py",
                 r_checkpoint, r_norm, index, s_norm,
                 a.repo / "tools" / "train.py", a.stage_root / "stage_manifest.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)

    pair_meta = json.loads(pair.read_text(encoding="utf-8"))
    data_meta = json.loads(prepared.read_text(encoding="utf-8"))
    if pair_meta.get("state") != "ATTEMPT13_PAIR_COMPLETE":
        raise RuntimeError("The paired attempt13 training has not completed successfully")
    if pair_meta.get("test_payloads_read") != 0 or pair_meta.get("buffer_payloads_read") != 0:
        raise RuntimeError("Pair receipt reports test/buffer reads")
    if data_meta.get("test_payloads_read") != 0 or data_meta.get("buffer_payloads_read") != 0:
        raise RuntimeError("Prepared-input receipt reports test/buffer reads")
    for arm in ("mse_control", "cot_aux"):
        if not (a.attempt / "runs" / arm / "checkpoints" / "best.ckpt").is_file():
            raise FileNotFoundError(f"Missing selected checkpoint for {arm}")

    out = a.attempt / "evaluation_retry4"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {out}")
    (out / "forecast_banks").mkdir(parents=True)
    (out / "cot_banks").mkdir()
    infer_s = a.base / "code_pipeline_attempt12" / "infer_S.py"
    infer_r = a.attempt / "infer_forecast_cot_attempt13.py"
    scorer = a.attempt / "score_attempt13_cot_pair_fp64.py"
    for path in (infer_s, infer_r, scorer):
        if not path.is_file():
            raise FileNotFoundError(path)

    for arm in ("mse_control", "cot_aux"):
        s_bank = out / "forecast_banks" / arm
        run([a.python, str(infer_s), "--repo", str(a.repo), "--index", str(index),
             "--norm", str(s_norm), "--checkpoint",
             str(a.attempt / "runs" / arm / "checkpoints" / "best.ckpt"),
             "--output", str(s_bank), "--stage-root", str(a.stage_root),
             "--batch-size", "2", "--workers", "4", "--device", "cuda:0",
             "--splits", "val"])
        cot_bank = out / "cot_banks" / f"{arm}_physical_cot.npy"
        run([a.python, str(infer_r), "--simvp-bank", str(s_bank),
             "--r-source", str(r_source), "--r-checkpoint", str(r_checkpoint),
             "--r-norm", str(r_norm), "--output", str(cot_bank),
             "--device", "cuda:0", "--sequence-batch-size", "8"])

    run([a.python, str(scorer),
         "--mse-cot-bank", str(out / "cot_banks" / "mse_control_physical_cot.npy"),
         "--aux-cot-bank", str(out / "cot_banks" / "cot_aux_physical_cot.npy"),
         "--mse-keys", str(out / "forecast_banks" / "mse_control" / "val" / "keys.csv"),
         "--aux-keys", str(out / "forecast_banks" / "cot_aux" / "val" / "keys.csv"),
         "--r-validation", str(r_val), "--output", str(out / "scored_fp64"),
         "--bootstrap-draws", "5000", "--seed", "20261021"])

    summary = out / "scored_fp64" / "summary_fp64.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    receipt = {
        "state": "ATTEMPT13_RETRY4_VALIDATION_INFERENCE_AND_FP64_SCORING_COMPLETE",
        "pair_receipt_sha256": sha256(pair),
        "prepared_inputs_sha256": sha256(prepared),
        "summary_fp64_sha256": sha256(summary),
        "summary_fp64": str(summary),
        "simvp_checkpoints": {
            arm: sha256(a.attempt / "runs" / arm / "checkpoints" / "best.ckpt")
            for arm in ("mse_control", "cot_aux")
        },
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "claim_scope": "validation-only; CPP retrieval reference; not an independent test or physical-truth claim",
    }
    receipt_path = out / "RUN_COMPLETE_attempt13_retry4.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
