"""Masked auxiliary losses for the isolated SimVP supervision experiment."""
from __future__ import annotations

import torch


def _valid(valid: torch.Tensor) -> torch.Tensor:
    if valid.ndim != 5:
        raise ValueError(f"expected [B,T,C,H,W] validity, got {tuple(valid.shape)}")
    return valid.bool()


def base_mse(pred: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if pred.shape != truth.shape or pred.shape != valid.shape:
        raise ValueError("prediction, truth, and validity shapes must match")
    m = _valid(valid)
    n = m.sum()
    if not n:
        raise ValueError("batch has no valid target pixels")
    err = pred.float() - truth.float()
    if not torch.isfinite(err[m]).all():
        raise FloatingPointError("non-finite prediction on valid target support")
    return err[m].square().mean()


def temporal_delta_l1(pred: torch.Tensor, truth: torch.Tensor,
                      valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match future adjacent-frame signed changes; returns loss and pair count."""
    if pred.shape != truth.shape or pred.shape != valid.shape:
        raise ValueError("prediction, truth, and validity shapes must match")
    m = _valid(valid)
    if pred.shape[1] < 2:
        return pred.float().sum() * 0.0, torch.zeros((), device=pred.device, dtype=torch.long)
    pair = m[:, 1:] & m[:, :-1]
    n = pair.sum()
    if not n:
        return pred.float().sum() * 0.0, n
    p = pred.float()
    y = truth.float()
    dp = p[:, 1:] - p[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    err = (dp - dy).abs()
    if not torch.isfinite(err[pair]).all():
        raise FloatingPointError("non-finite temporal error on valid pair support")
    return err[pair].mean(), n


def spatial_gradient_l1(pred: torch.Tensor, truth: torch.Tensor,
                        valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match signed row/column forward differences without wraparound."""
    if pred.shape != truth.shape or pred.shape != valid.shape:
        raise ValueError("prediction, truth, and validity shapes must match")
    m = _valid(valid)
    p = pred.float()
    y = truth.float()
    losses = []
    counts = []
    for dim in (-2, -1):
        if p.shape[dim] < 2:
            continue
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[dim] = slice(None, -1)
        right[dim] = slice(1, None)
        left, right = tuple(left), tuple(right)
        pair = m[left] & m[right]
        n = pair.sum()
        if n:
            err = ((p[right] - p[left]) - (y[right] - y[left])).abs()
            if not torch.isfinite(err[pair]).all():
                raise FloatingPointError("non-finite spatial error on valid pair support")
            losses.append(err[pair].sum())
            counts.append(n)
    if not losses:
        return p.sum() * 0.0, torch.zeros((), device=pred.device, dtype=torch.long)
    count = torch.stack(counts).sum()
    return torch.stack(losses).sum() / count, count


def loss_witness(device: str = "cpu") -> dict[str, float | int]:
    """Small deterministic forward/backward witness, independent of project data."""
    y = torch.zeros((2, 16, 3, 6, 7), device=device)
    m = torch.ones_like(y, dtype=torch.bool)
    p = y.clone().requires_grad_(True)
    l0 = base_mse(p, y, m)
    lt, nt = temporal_delta_l1(p, y, m)
    ls, ns = spatial_gradient_l1(p, y, m)
    assert l0.item() == lt.item() == ls.item() == 0.0
    # Constant offset is an amplitude error but has no temporal/spatial gradient error.
    q = torch.ones_like(y, requires_grad=True)
    assert base_mse(q, y, m).item() == 1.0
    assert temporal_delta_l1(q, y, m)[0].item() == 0.0
    assert spatial_gradient_l1(q, y, m)[0].item() == 0.0
    # Invalid sentinels cannot affect the loss or its finite-support denominator.
    m2 = m.clone()
    m2[..., 0, 0] = False
    a = torch.zeros_like(y, requires_grad=True)
    b = a.detach().clone()
    b[..., 0, 0] = 1e6
    b.requires_grad_(True)
    assert base_mse(a, y, m2).item() == base_mse(b, y, m2).item()
    # A real frame-to-frame change has a signed-delta error and a gradient path.
    r = torch.zeros_like(y, requires_grad=True)
    with torch.no_grad():
        y[:, 8:] = 1.0
    lt2, _ = temporal_delta_l1(r, y, m)
    assert lt2.item() > 0
    lt2.backward()
    assert r.grad is not None and torch.isfinite(r.grad).all() and r.grad.abs().sum().item() > 0
    # Spatial forward differences never wrap row/column boundaries.
    k = torch.zeros_like(y)
    k[..., 0, -1] = 1
    grad, ngrad = spatial_gradient_l1(k, torch.zeros_like(k), m)
    assert grad.item() > 0 and ngrad.item() == 2 * 16 * 3 * (5 * 7 + 6 * 6)
    return {"zero_losses": 1, "constant_offset_base": 1.0,
            "temporal_pairs": int(nt), "spatial_pairs": int(ns),
            "change_loss": float(lt2), "gradient_l1": float(grad)}
