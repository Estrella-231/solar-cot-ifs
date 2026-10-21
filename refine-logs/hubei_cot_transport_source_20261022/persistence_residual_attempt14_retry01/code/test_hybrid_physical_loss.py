#!/usr/bin/env python3
"""Small CPU contract test for attempt14 masked/equal-lead loss."""
from __future__ import annotations

import torch

from hybrid_physical_loss_20261022 import (
    LOG101,
    PHYSICAL_WEIGHT,
    equal_lead_hybrid_loss,
)


def reduce_sum(value: torch.Tensor, initialized: bool) -> torch.Tensor:
    assert not initialized
    return value


def main() -> None:
    pred = torch.tensor(
        [[[[[0.1, 0.3]]], [[[0.55, 0.8]]]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        [[[[[0.0, 2.0]]], [[[40.0, float("nan")]]]]], dtype=torch.float64
    )
    valid = torch.tensor(
        [[[[[True, True]]], [[[True, False]]]]], dtype=torch.bool
    )
    loss = equal_lead_hybrid_loss(pred, target, valid, False, reduce_sum)

    safe = torch.tensor([0.0, 2.0, 40.0], dtype=torch.float64)
    pred_valid = torch.stack((pred[0, 0, 0, 0, 0], pred[0, 0, 0, 0, 1],
                              pred[0, 1, 0, 0, 0]))
    target_log = torch.log1p(safe) / LOG101
    pred_phys = torch.expm1(pred_valid * LOG101)
    expected_log = torch.stack(((pred_valid[:2] - target_log[:2]).square().mean(),
                                (pred_valid[2:] - target_log[2:]).square().mean())).mean()
    expected_physical = torch.stack((((pred_phys[:2] - safe[:2]) / 100).square().mean(),
                                     ((pred_phys[2:] - safe[2:]) / 100).square().mean())).mean()
    expected = expected_log + PHYSICAL_WEIGHT * expected_physical
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 1, 0, 0, 1] == 0
    print("ATTEMPT14_HYBRID_LOSS_TEST_PASS")


if __name__ == "__main__":
    main()
