"""Train-only hybrid log/physical-COT loss for attempt14."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


LOG101 = math.log(101.0)
PHYSICAL_WEIGHT = 0.10


def equal_lead_hybrid_loss(
    pred: torch.Tensor,
    target_cot: torch.Tensor,
    valid: torch.Tensor,
    initialized: bool,
    reduce_sum,
) -> torch.Tensor:
    """Return equal-lead log-MSE + 0.10 normalized physical-COT MSE."""
    mask = valid.bool()
    safe_target = torch.where(mask, target_cot, torch.zeros_like(target_cot))
    safe_target = safe_target.clamp(0.0, 100.0)
    log_target = torch.log1p(safe_target) / LOG101

    log_error = torch.where(mask, pred - log_target, torch.zeros_like(pred))
    pred_cot = torch.expm1(pred * LOG101)
    physical_error = torch.where(
        mask, (pred_cot - safe_target) / 100.0, torch.zeros_like(pred_cot)
    )

    dims = (0, 2, 3, 4)
    log_sums = log_error.square().sum(dim=dims)
    physical_sums = physical_error.square().sum(dim=dims)
    counts = mask.sum(dim=dims).to(dtype=torch.float64)
    global_counts = reduce_sum(counts.clone(), initialized)
    active = global_counts > 0
    if not bool(active.any()):
        raise ValueError("global batch has no valid COT target pixels")

    world = dist.get_world_size() if initialized else 1
    combined_sums = log_sums + PHYSICAL_WEIGHT * physical_sums
    per_lead = world * combined_sums / global_counts.to(combined_sums.dtype).clamp_min(1)
    return per_lead[active].mean()
