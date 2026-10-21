#!/usr/bin/env python3
"""Profile 1/2/4 independent GPU workers, then run paired station-seed trials."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def launch_worker(args: list[str], gpu: str, log_path: Path, extra_env: dict[str, str] | None = None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--history-bank", type=Path, required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cuda-devices", required=True, help="Comma-separated scheduler-assigned physical IDs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1536)
    p.add_argument("--profile-updates", type=int, default=64)
    a = p.parse_args()
    gpu_ids = [x.strip() for x in a.cuda_devices.split(",") if x.strip()]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError(f"Expected exactly four distinct allocated GPUs, got {gpu_ids}")
    if (a.history_bank / "bank_audit.json").is_file():
        audit = json.loads((a.history_bank / "bank_audit.json").read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("History-bank independent audit is required before profile/training")
    if audit.get("state") != "PASS_HISTORY_COT_BANK_INDEPENDENT_AUDIT" or audit.get("test_payloads_read") != 0:
        raise ValueError("History-bank audit did not pass train/val-only gate")
    if a.output.exists():
        raise FileExistsError(f"Refusing existing orchestrator output: {a.output}")
    a.output.mkdir(parents=True)
    profile_root = a.output / "throughput_profile"
    profile_root.mkdir()
    profile_results = []
    profile_station_cycle = list(STATIONS)

    for nworker in (1, 2, 4):
        run_dir = profile_root / f"workers_{nworker}"
        run_dir.mkdir()
        active = []
        start_wall = time.perf_counter()
        for slot in range(nworker):
            station = profile_station_cycle[slot % len(profile_station_cycle)]
            out_json = run_dir / f"gpu_slot{slot}_{station}.json"
            log_path = run_dir / f"gpu_slot{slot}_{station}.log"
            cmd = [sys.executable, str(a.trainer), "--pack", str(a.pack),
                   "--history-bank", str(a.history_bank), "--station", station,
                   "--seed", str(42 + slot), "--batch-size", str(a.batch_size),
                   "--profile-only", "--profile-updates", str(a.profile_updates),
                   "--profile-output", str(out_json)]
            proc, log = launch_worker(cmd, gpu_ids[slot], log_path)
            active.append((proc, log, out_json, station, gpu_ids[slot], log_path))
        failures = []
        for proc, log, out_json, station, gpu, log_path in active:
            code = proc.wait()
            log.close()
            if code != 0 or not out_json.is_file():
                failures.append({"station": station, "gpu": gpu, "exit_code": code, "log": str(log_path)})
        elapsed = time.perf_counter() - start_wall
        if failures:
            raise RuntimeError(f"{nworker}-worker throughput profile failed: {failures}")
        worker_receipts = [json.loads(item[2].read_text(encoding="utf-8")) for item in active]
        aggregate = sum(float(np.mean([r["examples_per_second"] for r in w["results"]]))
                        for w in worker_receipts)
        profile_results.append({"workers": nworker, "assigned_gpu_ids": gpu_ids[:nworker],
                                "aggregate_examples_per_second": aggregate,
                                "wall_seconds_including_startup": elapsed,
                                "worker_receipts": [str(item[2]) for item in active]})
        print(json.dumps({"event": "PROFILE_COMPLETE", **profile_results[-1]}, ensure_ascii=False), flush=True)

    max_rate = max(x["aggregate_examples_per_second"] for x in profile_results)
    # Prefer the highest card count that stays within 5% of the measured peak.
    eligible = [x for x in profile_results if x["aggregate_examples_per_second"] >= 0.95 * max_rate]
    chosen = max(eligible, key=lambda x: x["workers"])
    receipt = {"state": "HISTORY_COT_GHI_1_2_4_GPU_PROFILE_COMPLETE",
               "profile_results": profile_results, "selected_workers": chosen["workers"],
               "selection": "highest worker count within 5 percent of peak aggregate measured examples/s",
               "test_payloads_read": 0, "buffer_payloads_read": 0}
    atomic_json(a.output / "throughput_profile.json", receipt)
    print(json.dumps({"event": "PROFILE_SELECTION", "selected_workers": chosen["workers"],
                      "profile": profile_results}, ensure_ascii=False), flush=True)

    trials_root = a.output / "pairs"
    trials_root.mkdir()
    tasks = [(st, seed) for seed in SEEDS for st in STATIONS]
    task_queue = iter(tasks)
    running = {}
    complete = []
    failed = []
    while True:
        while len(running) < chosen["workers"]:
            try:
                station, seed = next(task_queue)
            except StopIteration:
                break
            task_name = f"{station}_seed{seed}"
            out = trials_root / task_name
            log_path = trials_root / f"{task_name}.log"
            cmd = [sys.executable, str(a.trainer), "--pack", str(a.pack),
                   "--history-bank", str(a.history_bank), "--station", station,
                   "--seed", str(seed), "--epochs", str(a.epochs),
                   "--batch-size", str(a.batch_size), "--output", str(out)]
            busy_gpus = {x["gpu"] for x in running.values()}
            gpu = next(g for g in gpu_ids[:chosen["workers"]] if g not in busy_gpus)
            proc, log = launch_worker(cmd, gpu, log_path)
            running[proc.pid] = {"process": proc, "log": log, "station": station, "seed": seed,
                                 "gpu": gpu, "output": out, "log_path": log_path,
                                 "started": time.time()}
            print(json.dumps({"event": "PAIR_WORKER_STARTED", "station": station,
                              "seed": seed, "gpu": gpu, "output": str(out)}, ensure_ascii=False), flush=True)
        if not running:
            break
        finished = []
        while not finished:
            for pid, item in list(running.items()):
                code = item["process"].poll()
                if code is not None:
                    finished.append((pid, code, item))
            if not finished:
                time.sleep(1.0)
        for pid, code, item in finished:
            item["log"].close()
            del running[pid]
            record = {"station": item["station"], "seed": item["seed"], "gpu": item["gpu"],
                      "exit_code": code, "output": str(item["output"]), "log": str(item["log_path"]),
                      "wall_seconds": time.time() - item["started"]}
            if code == 0:
                meta = json.loads((item["output"] / "run_metadata.json").read_text(encoding="utf-8"))
                if meta.get("state") != "HISTORY_COT_PAIRED_TRAINING_COMPLETE":
                    code = 99
                    record["exit_code"] = code
                else:
                    record["state"] = "PAIR_COMPLETE"
            if code == 0:
                complete.append(record)
            else:
                record["state"] = "PAIR_FAILED"
                failed.append(record)
            print(json.dumps({"event": record["state"], **record}, ensure_ascii=False), flush=True)
            atomic_json(a.output / "orchestrator_progress.json", {
                "state": "RUNNING" if not failed else "RUNNING_WITH_FAILURES",
                "selected_workers": chosen["workers"], "completed_pairs": complete,
                "failed_pairs": failed, "pairs_expected": len(tasks),
                "test_payloads_read": 0, "buffer_payloads_read": 0})

    final = {"state": "HISTORY_COT_PAIRED_TRAINING_COMPLETE" if len(complete) == len(tasks) and not failed else "HISTORY_COT_PAIRED_TRAINING_PARTIAL_FAILURE",
             "profile": receipt, "selected_workers": chosen["workers"],
             "pairs_expected": len(tasks), "completed_pairs": complete, "failed_pairs": failed,
             "test_payloads_read": 0, "buffer_payloads_read": 0}
    atomic_json(a.output / "orchestrator_receipt.json", final)
    print(json.dumps({"event": final["state"], "completed": len(complete),
                      "failed": len(failed), "selected_workers": chosen["workers"]}, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    import numpy as np
    main()
