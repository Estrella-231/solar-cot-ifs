#!/usr/bin/env python3
"""Run the frozen attempt13 trainer with one loss-only override."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


if len(sys.argv) < 2:
    raise SystemExit("usage: run_train_hybrid_loss.py <trainer.py> [trainer args...]")
trainer_path = Path(sys.argv[1]).resolve()
if not trainer_path.is_file():
    raise FileNotFoundError(trainer_path)
smoke_only = "--smoke-only" in sys.argv[2:]
trainer_args = [arg for arg in sys.argv[2:] if arg != "--smoke-only"]
new_code = Path(__file__).resolve().parent
sys.path.insert(0, str(new_code))
sys.path.insert(0, str(trainer_path.parent))
scripts = trainer_path.parents[3] / "scripts"
sys.path.insert(0, str(scripts))

import retrieved_cot_forecaster_20261013 as base_model_module  # noqa: E402
from persistence_residual_cot_forecaster_20261022 import (  # noqa: E402
    PersistenceResidualCOTForecaster,
)
from hybrid_physical_loss_20261022 import (  # noqa: E402
    equal_lead_hybrid_loss,
)

base_model_module.RetrievedCOTForecaster = PersistenceResidualCOTForecaster

spec = importlib.util.spec_from_file_location("attempt14_frozen_trainer", trainer_path)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot import frozen trainer: {trainer_path}")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.distributed_equal_lead_loss = (
    lambda pred, target, valid, initialized: equal_lead_hybrid_loss(
        pred, target, valid, initialized, trainer.all_reduce_sum
    )
)
sys.argv = [str(trainer_path), *trainer_args]

# Verify the exact persistence initialization before any training output exists.
import torch  # noqa: E402

torch.manual_seed(42)
model = PersistenceResidualCOTForecaster(width=16).eval()
history = torch.rand(1, 8, 1, 16, 16)
valid_history = torch.ones_like(history, dtype=torch.bool)
geometry = torch.rand(1, 24, 3, 16, 16)
with torch.no_grad():
    prediction = model(history, valid_history, geometry)
    anchor = model.persistence_anchor(history, valid_history)
assert prediction.shape == (1, 16, 1, 16, 16)
assert torch.all((prediction >= 0) & (prediction <= 1))
assert torch.allclose(prediction, anchor[:, None].expand_as(prediction))
print("ATTEMPT14_MODEL_INIT_PASS; LOSS=LOG_MSE+0.10_PHYSICAL_MSE", flush=True)

if smoke_only:
    raise SystemExit(0)

trainer.main()
