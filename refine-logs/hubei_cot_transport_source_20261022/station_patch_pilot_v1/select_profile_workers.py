#!/usr/bin/env python3
"""Select a stable worker count from fixed-global-batch GPU profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--max-memory-gib", type=float, default=68.0)
    args = p.parse_args()
    target = args.output
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("state") != "PROFILE_SELECTION_FROZEN":
            raise ValueError(f"unexpected existing profile selection: {existing}")
        print(json.dumps(existing, indent=2))
        return

    profiles = []
    for workers in (1, 2, 4):
        path = args.profiles / f"profile_{workers}gpu.json"
        x = json.loads(path.read_text(encoding="utf-8"))
        if (x.get("state") != "PROFILE_COMPLETE"
                or x.get("world_size") != workers
                or x.get("global_batch") != args.global_batch
                or x.get("local_batch") != args.global_batch // workers
                or not x.get("global_batch_constant")
                or x.get("profile_steps", 0) < 30
                or x.get("max_rank_peak_allocated_gib", float("inf")) >= args.max_memory_gib):
            raise ValueError(f"profile failed contract for {workers} GPU(s): {x}")
        profiles.append(x)
    best = max(x["global_samples_per_second"] for x in profiles)
    eligible = [x for x in profiles
                if x["global_samples_per_second"] >= (1.0 - args.tolerance) * best]
    # Within the 5% throughput band, prefer more GPUs as the user requested.
    chosen = max(eligible, key=lambda x: (x["world_size"], x["global_samples_per_second"]))
    result = {
        "state": "PROFILE_SELECTION_FROZEN",
        "selected_workers": chosen["world_size"],
        "selected_global_samples_per_second": chosen["global_samples_per_second"],
        "fastest_global_samples_per_second": best,
        "within_fastest_tolerance": args.tolerance,
        "tie_rule": "choose the largest worker count within 5% of fastest stable throughput",
        "global_batch": args.global_batch,
        "profiles": [{"world_size": x["world_size"],
                      "global_samples_per_second": x["global_samples_per_second"],
                      "max_rank_peak_allocated_gib": x["max_rank_peak_allocated_gib"],
                      "seconds_per_update": x["seconds_per_update"]}
                     for x in profiles],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
