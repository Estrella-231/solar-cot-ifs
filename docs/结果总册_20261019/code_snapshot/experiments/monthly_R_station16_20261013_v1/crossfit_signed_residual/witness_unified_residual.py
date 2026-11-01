#!/usr/bin/env python3
"""Small deterministic CPU witness for the unified signed residual head."""
import json
import torch

from signed_residual_head_unified import SignedResidualHeadUnified, trainable_parameter_count

torch.manual_seed(20261014)
agri = torch.randn(4, 13, 3, 3)
geometry = torch.randn(4, 2, 3, 3)
cot = torch.randn(4, 1, 3, 3)
clear = torch.full((4,), 600.0)
base = torch.tensor([100.0, 250.0, 400.0, 550.0])
counts, max_identity = {}, {}
for arm in ("AGRI", "AGRI_COT"):
    model = SignedResidualHeadUnified(arm)
    counts[arm] = trainable_parameter_count(model)
    model.eval()
    with torch.no_grad():
        pred, _, delta = model(agri, geometry, cot, clear, base)
    max_identity[arm] = float((pred - base).abs().max())
    assert torch.equal(pred, base) and torch.count_nonzero(delta) == 0
    model.train()
    pred, _, _ = model(agri, geometry, cot, clear, base)
    loss = ((pred - torch.tensor([120.0, 240.0, 430.0, 520.0])) / 1000).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and model.residual_readout[-1].weight.grad is not None
assert counts["AGRI"] == counts["AGRI_COT"]
print(json.dumps({"witness": "UNIFIED_RESIDUAL_CPU_PASS", "parameter_counts": counts,
                  "zero_identity_max_abs": max_identity}, indent=2))
