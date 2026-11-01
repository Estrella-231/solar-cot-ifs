#!/usr/bin/env python3
"""CPU witness for the fixed physical-amplitude auxiliary COT loss."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from scripts.retrieved_cot_forecaster_20261013 import cot_supervision_masked_mse

trainer_path = Path(__file__).with_name("train_retrieved_cot_forecaster_20261017.py")
spec = importlib.util.spec_from_file_location("retrieved_cot_trainer", trainer_path)
trainer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trainer)

# A partly thick-cloud target plus an invalid NaN pixel exercises scale and mask handling.
pred = torch.full((2, 16, 1, 4, 4), math.log1p(8.0) / math.log(101.0), requires_grad=True)
target = torch.full_like(pred, 8.0)
valid = torch.ones_like(pred, dtype=torch.bool)
target[:, :, :, 0, 0] = 45.0
target[:, :, :, 3, 3] = float("nan")
valid[:, :, :, 3, 3] = False
base = cot_supervision_masked_mse(pred, target, valid)
aux = trainer.cot_amplitude_auxiliary_mse(pred, target, valid)
combined = base + 0.1 * aux
combined.backward()

zero_pred = torch.zeros((1, 16, 1, 2, 2), requires_grad=True)
zero_target = torch.zeros_like(zero_pred)
zero_mask = torch.ones_like(zero_pred, dtype=torch.bool)
zero_aux = trainer.cot_amplitude_auxiliary_mse(zero_pred, zero_target, zero_mask)

checks = {
    "base_finite": bool(torch.isfinite(base)),
    "amplitude_finite": bool(torch.isfinite(aux)),
    "combined_finite": bool(torch.isfinite(combined)),
    "gradients_finite": bool(torch.isfinite(pred.grad).all()),
    "invalid_nan_masked": bool(torch.isfinite(combined)),
    "zero_error_auxiliary_is_zero": float(zero_aux) == 0.0,
    "base_loss": float(base.detach()),
    "amplitude_auxiliary_loss": float(aux.detach()),
    "weighted_loss": float(combined.detach()),
    "test_payload_opened": False,
}
if not all(v for k, v in checks.items() if k.endswith("finite") or k in ("invalid_nan_masked", "zero_error_auxiliary_is_zero")):
    raise SystemExit(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
