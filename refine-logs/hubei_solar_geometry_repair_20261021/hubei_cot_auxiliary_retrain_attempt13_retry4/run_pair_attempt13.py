#!/usr/bin/env python3
"""Profile the joint frozen-R step, then train the two matched attempt13 arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_config(path: Path, cfg: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{k} = {v!r}" for k, v in cfg.items()) + "\n", encoding="utf-8")


def run_profile(args, prepared: dict[str, object], rpack: Path) -> int:
    code = args.base / "code_pipeline_attempt13"
    profile = args.attempt / "profile_joint_step_attempt13.py"
    # Preserve the earlier dependency-failure log under profile/; this run is
    # isolated so retry diagnostics never overwrite a failed attempt.
    report_dir = args.attempt / "profile_retry2"
    report_dir.mkdir(exist_ok=True)
    for batch in (16, 12, 8, 6, 4):
        report = report_dir / f"batch_{batch}.json"
        log = report_dir / f"batch_{batch}.log"
        command = [args.python, str(profile), "--repo", str(args.simvp_repo),
                   "--code-dir", str(args.attempt), "--data-root", str(prepared["S_cache"]),
                   "--index", str(prepared["index"]), "--s-norm", str(prepared["S_norm"]),
                   "--r-pack", str(rpack), "--r-checkpoint", str(prepared["r_checkpoint"]),
                   "--sidecar", str(prepared["sidecar"]), "--batch", str(batch), "--report", str(report)]
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode == 20:
            print(f"PROFILE_OOM batch={batch}; trying smaller batch", flush=True)
            continue
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        metrics = json.loads(report.read_text(encoding="utf-8"))
        peak = float(metrics["peak_allocated_gib"])
        print(f"PROFILE_PASS batch={batch} peak_gib={peak:.3f} seconds={metrics['elapsed_seconds']:.2f}", flush=True)
        if peak <= 72.0:
            return batch
        print(f"PROFILE_MEMORY_MARGIN_INSUFFICIENT batch={batch} peak_gib={peak:.3f}; trying smaller batch", flush=True)
    raise RuntimeError("No batch size passed the real joint SimVP+frozen-R profile with <=72 GiB peak")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--simvp-repo", type=Path, required=True)
    a = p.parse_args()
    prepared_path = a.attempt / "prepared_inputs.json"
    if not prepared_path.is_file():
        raise FileNotFoundError("Attempt13 data/CPP sidecar preflight must pass before training")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("state") != "PASS_ATTEMPT13_TRAIN_VAL_INPUTS_AND_CPP_PHYSICAL_TARGETS":
        raise ValueError("Attempt13 train/val input preflight is not PASS")
    if prepared.get("test_payloads_read") != 0 or prepared.get("buffer_payloads_read") != 0:
        raise ValueError("Attempt13 preflight reports test/buffer reads")
    repo = a.simvp_repo
    if not (repo / "tools" / "train.py").is_file():
        raise FileNotFoundError(f"Attempt12 SimVP code snapshot is unavailable: {repo}")
    rpack = Path(prepared["prepared_dir"]) / "R_packs" / "full"
    r_ckpt = Path(prepared["r_checkpoint"])
    r_norm = Path(prepared["r_norm"])
    sidecar = Path(prepared["sidecar"])
    for f in (r_ckpt, r_norm, sidecar / "manifest.json"):
        if not f.is_file():
            raise FileNotFoundError(f)
    if (a.attempt / "PAIR_COMPLETE_attempt13.json").exists():
        raise FileExistsError("Attempt13 pair completion already exists; refuse to overwrite")

    chosen_batch = run_profile(a, prepared, rpack)
    n_train = 4086
    steps = math.ceil(n_train / chosen_batch)
    epoch = 50
    run_root = a.attempt / "runs"
    run_root.mkdir(exist_ok=True)
    lambda_receipt = a.attempt / "lambda" / "train_only_lambda.json"
    lambda_receipt.parent.mkdir(exist_ok=True)
    arms = ("mse_control", "cot_aux")
    core_config = {
        "method": "SimVP", "model_type": "gSTA", "dataname": "hbagri",
        "pre_seq_length": 8, "aft_seq_length": 16, "total_length": 24,
        "temporal_stride": 1, "in_shape": [8, 16, 256, 256], "out_channels": 13,
        "metrics": ["mse", "mae"], "hid_S": 32, "hid_T": 256, "N_T": 8,
        "N_S": 4, "spatio_kernel_enc": 3, "spatio_kernel_dec": 3, "drop_path": 0.1,
        "lr": 1e-3, "opt": "adamw", "weight_decay": 1e-4, "sched": "cosine",
        "min_lr": 1e-5, "warmup_epoch": 5, "warmup_lr": 1e-5,
        "filter_bias_and_bn": False, "early_stop_patience": 100,
        "metric_for_bestckpt": "val_cot_rmse", "no_display_method_info": True,
        "batch_size": chosen_batch, "val_batch_size": chosen_batch, "num_workers": 4,
        "drop_last": False, "gpus": [0], "seed": 42, "epoch": epoch,
        "data_root": str(prepared["S_cache"]), "index_csv": str(prepared["index"]),
        "test": False, "eval_train": False, "dist": False, "fp16": False,
        "use_augment": False, "use_prefetcher": False, "deterministic": False,
    }
    common_json = json.dumps(core_config, sort_keys=True, separators=(",", ":")).encode()
    common_sha = hashlib.sha256(common_json).hexdigest()
    expected_updates = steps * epoch
    pair_started = {"state": "ATTEMPT13_PAIR_RUNNING", "batch": chosen_batch,
                    "steps_per_epoch": steps, "epochs": epoch, "expected_updates_per_arm": expected_updates,
                    "core_config_sha256": common_sha, "data_ready_sha256": prepared["prepared_data"]["DATA_READY_sha256"],
                    "s_norm_sha256": prepared["prepared_data"]["S_norm_sha256"],
                    "r_checkpoint_sha256": sha(r_ckpt), "r_norm_sha256": sha(r_norm),
                    "cpp_sidecar_sha256": sha(sidecar / "manifest.json"),
                    "selection": "equal-lead validation physical COT RMSE on exact CPP-common support",
                    "test_payloads_read": 0, "buffer_payloads_read": 0}
    (a.attempt / "PAIR_STARTED_attempt13.json").write_text(json.dumps(pair_started, indent=2), encoding="utf-8")
    arm_receipts = {}
    for arm in arms:
        arm_dir = run_root / arm
        arm_dir.mkdir(parents=True, exist_ok=False)
        cwd = arm_dir / "cwd"
        cwd.mkdir()
        config = dict(core_config)
        config_path = arm_dir / "config.py"
        write_config(config_path, config)
        env = os.environ.copy()
        env.update({
            "SIMVP_REPO": str(repo),
            "HUBEI_SIMVP_CONFIG": str(config_path),
            "HUBEI_SIMVP_NORM": str(prepared["S_norm"]),
            "HUBEI_SIMVP_RUN_CWD": str(cwd),
            "HUBEI_SOLAR_RUN_DIR": str(arm_dir),
            "HUBEI_COT_RUN_DIR": str(arm_dir),
            "HUBEI_SOLAR_ROLE": "full",
            "HUBEI_SOLAR_DATA_READY": str(Path(prepared["prepared_dir"]) / "DATA_READY.json"),
            "HUBEI_COT_AUX_MODE": arm,
            "HUBEI_COT_LAMBDA_RECEIPT": str(lambda_receipt),
            "HUBEI_COT_R_CHECKPOINT": str(r_ckpt),
            "HUBEI_COT_R_NORM": str(r_norm),
            "HUBEI_COT_TARGET_SIDECAR": str(sidecar),
            "HUBEI_COT_RUN_DIR": str(arm_dir),
        })
        command = [a.python, str(a.attempt / "run_cot_aux_simvp_attempt13.py"),
                   "--gpus", "0", "--res_dir", str(run_root), "--ex_name", arm]
        log_path = arm_dir / "training.log"
        print(f"START_ARM {arm} output={arm_dir}", flush=True)
        with log_path.open("w", encoding="utf-8") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            (a.attempt / "PAIR_FAILURE_attempt13.json").write_text(json.dumps(
                {"state": "ARM_FAILED", "arm": arm, "returncode": result.returncode,
                 "log": str(log_path), "test_payloads_read": 0, "buffer_payloads_read": 0}, indent=2), encoding="utf-8")
            raise subprocess.CalledProcessError(result.returncode, command)
        metadata_path = arm_dir / "run_metadata.json"
        init_path = arm_dir / "attempt13_init_receipt.json"
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        init = json.loads(init_path.read_text(encoding="utf-8"))
        if meta.get("state") != "ATTEMPT13_SIMVP_COT_AUX_ARM_COMPLETE" or meta.get("arm") != arm:
            raise ValueError(f"Arm {arm} did not produce the required completion receipt")
        if meta.get("test_payloads_read") != 0 or meta.get("buffer_payloads_read") != 0:
            raise ValueError(f"Arm {arm} reports test/buffer reads")
        if meta.get("completed_epochs") != epoch or meta.get("configured_max_epochs") != epoch:
            raise ValueError(f"Arm {arm} did not complete the same fixed 50 epochs")
        if meta.get("global_updates") != expected_updates or init.get("lambda_receipt_sha256") is None:
            raise ValueError(f"Arm {arm} update count or frozen lambda receipt mismatch")
        arm_receipts[arm] = {"run_metadata": meta, "init_receipt": init,
                             "run_metadata_sha256": sha(metadata_path),
                             "init_receipt_sha256": sha(init_path),
                             "training_log": str(log_path), "config_sha256": sha(config_path)}
        (a.attempt / "PAIR_STARTED_attempt13.json").write_text(json.dumps(
            {**pair_started, "completed_arms": list(arm_receipts), "test_payloads_read": 0,
             "buffer_payloads_read": 0}, indent=2), encoding="utf-8")

    control = arm_receipts["mse_control"]
    cot = arm_receipts["cot_aux"]
    m0, c0 = control["init_receipt"], cot["init_receipt"]
    invariant_fields = ("initial_simvp_state_sha256", "r_checkpoint_sha256", "simvp_norm_sha256",
                        "r_norm_sha256", "sidecar_manifest_sha256", "lambda_receipt_sha256")
    mismatches = {k: (m0.get(k), c0.get(k)) for k in invariant_fields if m0.get(k) != c0.get(k)}
    if mismatches:
        raise ValueError(f"Paired initialization/provenance mismatch: {mismatches}")
    if control["run_metadata"]["global_updates"] != cot["run_metadata"]["global_updates"]:
        raise ValueError("Paired arms used different optimizer update counts")
    final = {**pair_started, "state": "ATTEMPT13_PAIR_COMPLETE",
             "completed_arms": arm_receipts,
             "pair_invariants": {k: m0[k] for k in invariant_fields},
             "same_initial_state": True, "same_data_norm_R_CPP_sidecar_lambda": True,
             "same_optimizer_updates": True,
             "interpretation": "train/validation screening only; CPP is a retrieval reference, not independent truth",
             "test_payloads_read": 0, "buffer_payloads_read": 0}
    out = a.attempt / "PAIR_COMPLETE_attempt13.json"
    out.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"ATTEMPT13_PAIR_COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
