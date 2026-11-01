#!/usr/bin/env python3
"""Capture a successful PBS accounting record from the tc6000 login node."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--ssh-target", default="slfu@10.65.129.39")
    ap.add_argument("--identity-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"\d+\.tc6000", args.job_id):
        raise SystemExit("job id must be an exact tc6000 PBS id")
    cmd = [
        "ssh", "-i", str(args.identity_file), "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", args.ssh_target,
        f"tracejob {args.job_id}",
    ]
    run = subprocess.run(cmd, capture_output=True, check=False)
    combined = run.stdout + run.stderr
    decoded = combined.decode("utf-8", errors="replace")
    lines = [line.strip() for line in decoded.splitlines()
             if "Exit_status=" in line and "resources_used." in line]
    successful = [line for line in lines if re.search(r"Exit_status=0(?:\s|$)", line)]
    if run.returncode != 0 or not successful:
        raise SystemExit("No successful Exit_status=0 accounting record; receipt not written")
    receipt = {
        "status": "PBS_TRAINING_EXIT_CONFIRMED",
        "job_id": args.job_id,
        "exit_status": 0,
        "evidence_source": "tracejob queried over SSH from tc6000 login node",
        "exit_record_line": successful[-1],
        "tracejob_combined_output_sha256": hashlib.sha256(combined).hexdigest(),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing receipt: {args.output}")
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "job_id": args.job_id,
                      "exit_record_line": receipt["exit_record_line"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
