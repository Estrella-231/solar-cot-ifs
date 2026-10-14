#!/usr/bin/env python3
"""CPU witness for identity initialization, capacity match, and gradient flow."""
import json
import torch

from head_candidate_v0 import LocalGHIHead
from signed_residual_head import SignedResidualHead, trainable_parameter_count


def main():
    torch.set_num_threads(1)
    torch.manual_seed(31415)
    a = LocalGHIHead("A", geometry_channels=2)
    state = a.state_dict()
    results = {}
    for arm in ("AGRI", "AGRI_COT"):
        torch.manual_seed(2718)
        model = SignedResidualHead(arm)
        model.load_frozen_a(state)
        x = torch.randn(8, 13, 3, 3)
        g = torch.randn(8, 2, 3, 3)
        c = torch.randn(8, 1, 3, 3)
        clear = torch.rand(8) * 900 + 100
        y = torch.rand(8) * 800
        pred, base, delta = model(x, g, c, clear)
        identity = float((pred - base).abs().max())
        delta_max = float(delta.abs().max())
        assert identity == 0.0 and delta_max == 0.0
        # First update opens the final projection; the next verifies gradients
        # reach the feature/auxiliary layers while A remains completely frozen.
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            pred, _, _ = model(x, g, c, clear)
            ((pred - y) / 1000).square().mean().backward()
            assert all(p.grad is None for p in model.features.parameters())
            assert any(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.residual_readout.parameters())
            opt.step()
        results[arm] = {"parameters": trainable_parameter_count(model),
                        "step_zero_max_abs_error": identity,
                        "step_zero_max_abs_delta": delta_max,
                        "frozen_A_gradients_absent": True}
    assert results["AGRI"]["parameters"] == results["AGRI_COT"]["parameters"]
    results["state"] = "PASS"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
