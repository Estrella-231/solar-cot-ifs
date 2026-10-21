#!/usr/bin/env python3
"""Audit physical bounds of saved attempt13 validation forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.predictions, allow_pickle=False) as z:
        scaled = np.asarray(z["pred_log_scaled"], dtype=np.float64)
        keys = [str(x) for x in z["init_time_bjt"].tolist()]
    if not np.isfinite(scaled).all():
        raise ValueError("nonfinite normalized predictions")
    if scaled.min() < 0.0 or scaled.max() > 1.0:
        raise ValueError("normalized predictions violate the model's [0,1] bound")
    physical = np.expm1(scaled * np.log(101.0))
    excess = physical[physical > 100.0] - 100.0
    score = json.loads(a.score.read_text(encoding="utf-8"))
    scored_excess_count = int(score["out_of_training_range_forecast_pixel_count_cot_lt0_or_gt100"])
    max_excess = float(excess.max()) if excess.size else 0.0
    # FP64 expm1 at exactly log(101) is 100 + 2.84e-14; tolerate 1e-10.
    if max_excess > 1e-10:
        raise ValueError(f"physical COT exceeds 100 beyond round-off: {max_excess}")
    result = {
        "state": "PASS_NORMALIZED_AND_PHYSICAL_COT_BOUNDS",
        "prediction_shape": list(scaled.shape),
        "init_keys": len(keys),
        "scaled_log_cot_min": float(scaled.min()),
        "scaled_log_cot_max": float(scaled.max()),
        "physical_cot_min": float(physical.min()),
        "physical_cot_max_fp64": float(physical.max()),
        "full_saved_array_count_fp64_gt100": int(excess.size),
        "max_fp64_excess_over_100": max_excess,
        "scorer_flagged_values_outside_0_100_on_validation_pairs": scored_excess_count,
        "interpretation": "All normalized values obey [0,1]. The tiny expm1 excess at the exact upper endpoint is floating-point round-off, not a physical forecast above 100.",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
