#!/usr/bin/env python3
"""Run validation-only full-scene AGRI guard after the paired COT scorer.

This is post-hoc assessment only: it neither trains nor selects checkpoints,
uses no CPP payloads, and reads only the frozen validation split.
"""
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
    cot_run = a.attempt / "evaluation_retry4" / "RUN_COMPLETE_attempt13_retry4.json"
    index = a.base / "prepared_attempt12" / "indexes" / "full.csv"
    norm = a.base / "prepared_attempt12" / "norms" / "full.npz"
    stage_manifest = a.stage_root / "stage_manifest.csv"
    infer = a.attempt / "infer_S_physical_agri_guard_attempt13_v2.py"
    audit = a.attempt / "audit_attempt13_agri_guard_fp64_v2.py"
    for path in (pair, prepared, cot_run, index, norm, stage_manifest, infer, audit,
                 a.repo / "openstl", a.repo / "openstl" / "datasets" / "dataloader_hbagri.py"):
        if not path.exists():
            raise FileNotFoundError(path)

    pair_meta = json.loads(pair.read_text(encoding="utf-8"))
    prepared_meta = json.loads(prepared.read_text(encoding="utf-8"))
    cot_meta = json.loads(cot_run.read_text(encoding="utf-8"))
    if pair_meta.get("state") != "ATTEMPT13_PAIR_COMPLETE":
        raise RuntimeError("Paired training receipt is not complete")
    for name, meta in (("pair", pair_meta), ("prepared", prepared_meta), ("COT scorer", cot_meta)):
        if meta.get("test_payloads_read") != 0 or meta.get("buffer_payloads_read") != 0:
            raise RuntimeError(f"{name} receipt reports test/buffer reads")
    if cot_meta.get("state") != "ATTEMPT13_RETRY4_VALIDATION_INFERENCE_AND_FP64_SCORING_COMPLETE":
        raise RuntimeError("COT inference/scoring receipt is not complete")

    for arm in ("mse_control", "cot_aux"):
        checkpoint = a.attempt / "runs" / arm / "checkpoints" / "best.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")

    out = a.attempt / "agri_guard_retry4"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite AGRI guard output: {out}")
    banks = out / "banks"
    banks.mkdir(parents=True)

    for arm in ("mse_control", "cot_aux"):
        bank = banks / arm
        run([
            a.python, str(infer), "--repo", str(a.repo), "--index", str(index),
            "--norm", str(norm), "--checkpoint",
            str(a.attempt / "runs" / arm / "checkpoints" / "best.ckpt"),
            "--output", str(bank), "--stage-root", str(a.stage_root),
            "--batch-size", "2", "--workers", "4", "--device", "cuda:0",
            "--splits", "val",
        ])

    audit_out = out / "fp64_audit"
    run([
        a.python, str(audit), "--mse-bank", str(banks / "mse_control"),
        "--cot-aux-bank", str(banks / "cot_aux"), "--output", str(audit_out),
    ])
    result = audit_out / "agri_guard_fp64.json"
    if not result.is_file():
        raise FileNotFoundError(result)
    result_meta = json.loads(result.read_text(encoding="utf-8"))
    receipt = {
        "state": "ATTEMPT13_VALIDATION_AGRI_PAIRED_GUARD_COMPLETE",
        "attempt": "retry4",
        "paired_training_receipt_sha256": sha256(pair),
        "prepared_inputs_receipt_sha256": sha256(prepared),
        "cot_scoring_receipt_sha256": sha256(cot_run),
        "validation_index_sha256": sha256(index),
        "train_only_normalization_sha256": sha256(norm),
        "guard_inference_code_sha256": sha256(infer),
        "guard_audit_code_sha256": sha256(audit),
        "selected_checkpoints_sha256": {
            arm: sha256(a.attempt / "runs" / arm / "checkpoints" / "best.ckpt")
            for arm in ("mse_control", "cot_aux")
        },
        "agri_guard_result_sha256": sha256(result),
        "agri_guard_result": str(result),
        "common_validation_sequences": result_meta.get("common_validation_sequences"),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "claim_scope": "validation-only, same-index paired AGRI safeguard; not an independent test or deployment result",
    }
    receipt_path = out / "AGRI_GUARD_COMPLETE_attempt13_retry4.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
