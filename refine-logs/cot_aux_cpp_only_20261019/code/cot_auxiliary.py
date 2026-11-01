"""Finite-safe differentiable COT supervision for a frozen station16 retriever.

Tensors use B,T,C,H,W for SimVP image/geometry, station coordinates per
target frame, and B,T,S,1,16,16 for reference fields/masks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def station_r_inputs(pred_agri_norm, geometry, centers_rc, agri_mean, agri_std,
                     r_mean, r_std, patch_size=16):
    """Build train-normalized FP32 R inputs from predicted normalized AGRI.

    centers_rc is [B,T,S,2] and stores exact source-grid row/column centers.
    Returns [B,T,S,16,patch,patch]. No detach/no_grad is used.
    """
    if pred_agri_norm.ndim != 5 or geometry.ndim not in (5, 6):
        raise ValueError("expected B,T,C,H,W or station-cropped B,T,S,C,P,P geometry")
    b, t, c, h, w = pred_agri_norm.shape
    if c != 13 or geometry.shape[:2] != (b, t):
        raise ValueError("expected 13 AGRI and geometry on matching B,T")
    if centers_rc.ndim != 4 or centers_rc.shape[:2] != (b, t) or centers_rc.shape[-1] != 2:
        raise ValueError("centers_rc must have shape B,T,S,2")
    s = centers_rc.shape[2]
    p = patch_size
    if geometry.ndim == 5 and geometry.shape[2] != 3:
        raise ValueError("full-grid geometry must have three channels")
    if geometry.ndim == 6 and geometry.shape != (b, t, s, 3, p, p):
        raise ValueError("cropped geometry must have shape B,T,S,3,P,P")
    physical = pred_agri_norm.float() * agri_std.float().view(1, 1, c, 1, 1) \
        + agri_mean.float().view(1, 1, c, 1, 1)
    rows = centers_rc[..., 0].long() - p // 2
    cols = centers_rc[..., 1].long() - p // 2
    if (rows < 0).any() or (cols < 0).any() or (rows + p > h).any() or (cols + p > w).any():
        raise ValueError("station patch lies outside the AGRI grid")
    # Convert all crop indices once; repeated CUDA scalar-to-int conversion
    # inside the B×T×S loop would synchronize the device for every patch.
    rows_cpu, cols_cpu = rows.detach().cpu(), cols.detach().cpu()
    out = []
    # Explicit B/T/S indexing avoids materializing all sliding windows of the
    # 256x256 prediction and also supports frame-varying station grid indices.
    for bi in range(b):
        ts = []
        for ti in range(t):
            stations = []
            for si in range(s):
                r, col = int(rows_cpu[bi, ti, si]), int(cols_cpu[bi, ti, si])
                a = physical[bi, ti, :, r:r+p, col:col+p]
                if geometry.ndim == 6:
                    g = geometry[bi, ti, si].float()
                else:
                    g = geometry[bi, ti, :, r:r+p, col:col+p].float()
                x = torch.cat((a, g), dim=0)
                x = (x - r_mean.float().view(16, 1, 1)) / r_std.float().view(16, 1, 1)
                stations.append(x)
            ts.append(torch.stack(stations))
        out.append(torch.stack(ts))
    return torch.stack(out)


def mixed_cot_target(cpp_physical, cpp_mask, observed_r_log1p, r_valid_mask):
    """Use CPP where valid; otherwise use observed-AGRI→R as train-only target."""
    if cpp_physical.shape != cpp_mask.shape or cpp_physical.shape != observed_r_log1p.shape \
            or cpp_physical.shape != r_valid_mask.shape:
        raise ValueError("CPP/R target and masks must have identical shape")
    rm = r_valid_mask.bool() & torch.isfinite(observed_r_log1p)
    cm = cpp_mask.bool() & rm & torch.isfinite(cpp_physical) & (cpp_physical >= 0)
    # R-derived fallback is used only outside the CPP-valid support.
    use_r = rm & ~cm
    valid = cm | use_r
    cpp_log = torch.log1p(torch.where(cm, cpp_physical.float(), torch.zeros_like(cpp_physical.float())))
    target = torch.where(cm, cpp_log, torch.where(use_r, observed_r_log1p.float(), 0.0))
    return target, valid, cm, use_r


def cpp_only_cot_target(cpp_physical, cpp_mask, r_valid_mask):
    """Use CPP supervision only; retain the frozen-R common valid support.

    This is the denoising ablation against mixed_cot_target: no AGRI->R
    fallback values enter the auxiliary target. The mask remains CPP-valid
    AND R-input-valid so that target coverage is explicit and comparable.
    """
    if cpp_physical.shape != cpp_mask.shape or cpp_physical.shape != r_valid_mask.shape:
        raise ValueError("CPP target and masks must have identical shape")
    valid = (cpp_mask.bool() & r_valid_mask.bool() & torch.isfinite(cpp_physical)
             & (cpp_physical >= 0))
    safe = torch.where(valid, cpp_physical.float(), torch.zeros_like(cpp_physical.float()))
    target = torch.log1p(safe)
    return target, valid, valid, torch.zeros_like(valid)


def frozen_r_cot_loss(r_model, r_input, target_log1p, valid_mask, delta=0.1):
    """Return masked Huber loss and R output while retaining gradients to input."""
    if any(p.requires_grad for p in r_model.parameters()):
        raise ValueError("retriever parameters must be frozen")
    if r_input.shape[:-3] != target_log1p.shape[:-3] or \
            r_input.shape[-2:] != target_log1p.shape[-2:] or \
            target_log1p.shape != valid_mask.shape:
        raise ValueError("R input, target, and mask dimensions do not align")
    lead_shape = target_log1p.shape[:-3]
    x = r_input.float().reshape(-1, 16, *r_input.shape[-2:])
    target = target_log1p.float().reshape(-1, 1, *target_log1p.shape[-2:])
    mask = valid_mask.bool().reshape_as(target)
    # Keep the retriever inference in the same FP32 mode used to create the
    # frozen R COT bank, even when the caller autocasts SimVP to BF16.
    with torch.autocast(device_type=x.device.type, enabled=False):
        pred = r_model(x).float().reshape_as(target)
    safe_pred = torch.where(mask, pred, torch.zeros_like(pred))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    values = F.huber_loss(safe_pred, safe_target, reduction="none", delta=delta)
    loss = values.sum() / mask.sum().clamp_min(1)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite masked COT auxiliary loss")
    return loss, pred.reshape(*lead_shape, 1, *pred.shape[-2:])
