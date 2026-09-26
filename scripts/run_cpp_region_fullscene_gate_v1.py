"""Run the unmodified portable full-scene CLI, then audit frozen regional outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import netCDF4
import numpy as np
from produce_hunan_cpp_region_v1 import atomic_json, grid_mapping, sha


def main():
    p = argparse.ArgumentParser()
    for name in ("base", "grid", "times", "region", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    logs = a.output / "logs"
    logs.mkdir(exist_ok=True)
    rows, cols, *_ = grid_mapping(a.grid)
    times = json.loads(a.times.read_text())
    identity = {"script_sha256": sha(__file__), "times_sha256": sha(a.times),
                "grid_sha256": sha(a.grid), "region_identity_sha256": sha(a.region / "identity.json"),
                "portable_entry_sha256": sha(a.base / "Code/Run_Pipeline.py"), "test_used": False}
    report = {"identity": identity, "frames": []}
    if (a.output / "report.json").exists():
        report = json.loads((a.output / "report.json").read_text())
        if report["identity"] != identity:
            raise RuntimeError("existing gate identity changed")
        if any(not f["passed"] for f in report["frames"]):
            raise RuntimeError("previous numerical failure requires a separate diagnosed run")
    env = os.environ.copy()
    env.update(GLOBAL_CPP_TEST_ROOT=str(a.output), GLOBAL_CPP_RESULT_ROOT=str(a.output / "Result"),
               GLOBAL_CPP_LOG_DIR=str(logs), GLOBAL_CPP_WRITE_EMPTY_ON_ERROR="0",
               GLOBAL_CPP_BATCH_SIZE="64", GLOBAL_CPP_DATALOADER_WORKERS="0",
               GLOBAL_CPP_TORCH_THREADS="4")
    for stamp in times:
        tag = stamp.replace("-", "").replace(":", "").replace("T", "")
        reference = a.output / "Result/FY/FY4B/Result/CPP" / tag[:4] / tag[:8] / f"FY4B_AGRI_{tag}.nc"
        region = a.region / f"{tag}.npz"
        previous = next((f for f in report["frames"] if f["utc"] == stamp), None)
        if previous:
            if sha(reference) != previous["reference_sha256"] or sha(region) != previous["region_sha256"]:
                raise RuntimeError("completed gate artifacts changed")
            continue
        started = time.perf_counter()
        run_id = f"fullscene_gate_{tag}"
        if reference.exists():
            raise RuntimeError(f"unregistered full-scene output: {reference}")
        with (logs / f"{tag}.log").open("w") as log:
            subprocess.run([sys.executable, "Code/Run_Pipeline.py", "satellite", "--satellite", "FY4B_105E",
                            "--time", stamp, "--device", "cuda:0", "--time-workers", "1",
                            "--satellite-workers", "1", "--retrieval-policy", "skip", "--run-id", run_id,
                            "--log-dir", str(logs), "--summary", str(a.output / f"{run_id}_summary.json")],
                           cwd=a.base, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with netCDF4.Dataset(reference) as ds:
            sl = (slice(rows[0], rows[-1]+1), slice(cols[0], cols[-1]+1))
            cot = np.asarray(np.ma.filled(ds["COT"][sl], np.nan))
            clp = np.asarray(np.ma.filled(ds["CLP"][sl], np.nan))
            qa = np.asarray(ds["QA"][sl])
        with np.load(region, allow_pickle=False) as z:
            rcot, rclp, valid, observed = z["cot"], z["clp"], z["valid"], z["observation_valid"]
        same_mask = bool(np.array_equal(np.isfinite(cot), np.isfinite(rcot)))
        common = valid & np.isfinite(cot) & (qa == 0)
        delta = np.abs(rcot[common] - cot[common])
        same_clp = bool(np.array_equal(rclp[np.isfinite(clp)], clp[np.isfinite(clp)]))
        close = bool(common.any() and np.all(delta <= 1e-4 + 1e-5*np.abs(cot[common])))
        frame = {"utc": stamp, "passed": same_mask and same_clp and close,
                 "mask_same": same_mask, "clp_same": same_clp,
                 "common_pixels": int(common.sum()), "observed_fraction": float(observed.mean()),
                 "reference_valid_fraction": float(np.isfinite(cot).mean()),
                 "max_abs": float(delta.max()) if delta.size else None,
                 "mae": float(delta.mean()) if delta.size else None,
                 "reference_sha256": sha(reference), "region_sha256": sha(region),
                 "seconds": time.perf_counter()-started}
        report["frames"].append(frame)
        atomic_json(a.output / "report.json", report)
        print(json.dumps(frame), flush=True)
        if not frame["passed"]:
            raise RuntimeError("strict full-scene gate failed; do not expand production")
    atomic_json(a.output / "complete.json", {"frames": len(times), "passed": True, "test_used": False})


if __name__ == "__main__":
    main()
