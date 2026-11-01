#!/usr/bin/env python3
"""CPU-only schema/key smoke for the causal inference dataset."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    module_path = Path(__file__).with_name("infer_frozen_monthly_validation.py")
    spec = importlib.util.spec_from_file_location("attempt16_infer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load inference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.CausalValidationDataset(args.cache)
    if len(dataset) != 786:
        raise AssertionError(f"unexpected validation windows: {len(dataset)}")
    first, last = dataset[0], dataset[len(dataset) - 1]
    for row in (first, last):
        if tuple(row["history"].shape) != (8, 1, 16, 16):
            raise AssertionError(tuple(row["history"].shape))
        if tuple(row["history_valid"].shape) != (8, 1, 16, 16):
            raise AssertionError(tuple(row["history_valid"].shape))
        if tuple(row["geometry"].shape) != (24, 3, 16, 16):
            raise AssertionError(tuple(row["geometry"].shape))
        if "target" in row or "target_valid" in row:
            raise AssertionError("future target was present in causal inference sample")
    print({"state": "PASS_CAUSAL_VALIDATION_DATASET_SCHEMA", "windows": len(dataset),
           "first_init_bjt": first["init_time"], "last_init_bjt": last["init_time"],
           "cpp_arrays_opened": False})


if __name__ == "__main__":
    main()
