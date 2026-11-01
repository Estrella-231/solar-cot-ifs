"""High-confidence CPP-only targets for the isolated COT-label-source ablation."""
from __future__ import annotations

import torch


def cpp_only_cot_target(cpp_physical, cpp_mask, r_valid_mask):
    """Return log1p CPP targets on common valid support; never fill with R pseudo-labels."""
    if cpp_physical.shape != cpp_mask.shape or cpp_physical.shape != r_valid_mask.shape:
        raise ValueError("CPP values and masks must have identical shapes")
    valid = (
        cpp_mask.bool()
        & r_valid_mask.bool()
        & torch.isfinite(cpp_physical)
        & (cpp_physical >= 0)
    )
    safe_cpp = torch.where(valid, cpp_physical.float(), torch.zeros_like(cpp_physical.float()))
    target = torch.log1p(safe_cpp)
    return target, valid, valid, torch.zeros_like(valid)
