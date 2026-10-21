#!/usr/bin/env python3
"""Run the existing frozen-contract trainer with the residual COT model class."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if len(sys.argv) < 2:
    raise SystemExit("usage: run_train_candidate.py <path-to-existing-trainer> [trainer args...]")
trainer = Path(sys.argv[1]).resolve()
if not trainer.is_file():
    raise FileNotFoundError(trainer)
sys.argv = [str(trainer), *sys.argv[2:]]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(trainer.parent))
scripts = trainer.parents[3] / "scripts"
sys.path.insert(0, str(scripts))

import retrieved_cot_forecaster_20261013 as base_module  # noqa: E402
from persistence_residual_cot_forecaster_20261022 import (  # noqa: E402
    PersistenceResidualCOTForecaster,
)

base_module.RetrievedCOTForecaster = PersistenceResidualCOTForecaster

# Run an executable architecture/initialization check in the training
# environment before the existing trainer can create checkpoints.
import torch  # noqa: E402

torch.manual_seed(42)
smoke_model = PersistenceResidualCOTForecaster(width=16).eval()
smoke_history = torch.rand(2, 8, 1, 16, 16)
smoke_valid = torch.ones_like(smoke_history, dtype=torch.bool)
smoke_valid[0, -1, 0, 2, 3] = False
smoke_valid[0, :-1, 0, 2, 3] = False
smoke_valid[0, -3, 0, 2, 3] = True
smoke_valid[1, :, 0, 4, 5] = False
smoke_geometry = torch.rand(2, 24, 3, 16, 16)
with torch.no_grad():
    smoke_prediction = smoke_model(smoke_history, smoke_valid, smoke_geometry)
    smoke_anchor = smoke_model.persistence_anchor(smoke_history, smoke_valid)
assert smoke_prediction.shape == (2, 16, 1, 16, 16)
assert torch.all((smoke_prediction >= 0) & (smoke_prediction <= 1))
assert torch.allclose(smoke_prediction, smoke_anchor[:, None].expand_as(smoke_prediction))
assert smoke_anchor[0, 0, 2, 3] == smoke_history[0, -3, 0, 2, 3]
assert smoke_anchor[1, 0, 4, 5] == 0
smoke_model.decoder[-1].bias.data.fill_(0.5)
with torch.no_grad():
    changed = smoke_model(smoke_history, smoke_valid, smoke_geometry)
assert torch.all(changed <= 1)
assert torch.all(changed[0, :, 0, 2, 3] > smoke_anchor[0, 0, 2, 3])
print("PERSISTENCE_RESIDUAL_SMOKE_PASS shape/bounds/invalid-mask/learnable-residual")

runpy.run_path(str(trainer), run_name="__main__")
