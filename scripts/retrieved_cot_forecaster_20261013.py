"""Forecast a COT trajectory from retrieved historical COT maps.

This is a separate candidate from DirectCOTForecaster: the latter consumes
historical AGRI, while this model consumes historical log1p-COT and its
validity mask. It never receives future AGRI, CPP, COT labels, or their masks.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3,
                 stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = kernel // 2
        groups = 8 if cout % 8 == 0 else 1
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
        )


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class RetrievedCOTForecaster(nn.Module):
    """Map 8 retrieved COT frames to 16 future log1p(COT) maps.

    Args:
        history_cot: [B, 8, 1, H, W], log1p(COT)/log(101), invalid values 0.
        history_valid: [B, 8, 1, H, W], 1 only where historical retrieval is valid.
        geometry: [B, 24, 3, H, W], [cosSOZ, cosRAA, day_mask] for 8 history
            frames followed by 16 known target times. Geometry is deterministic
            and may be available at inference; target-label masks are not.

    Returns:
        [B, 16, 1, H, W] nonnegative normalized log1p(COT).
    """

    def __init__(self, width: int = 96):
        super().__init__()
        self.width = int(width)
        self.history_encoder = nn.Sequential(
            ConvNormAct(5, 32, 5, 2, 2),
            ConvNormAct(32, 64, 4, 2, 1),
            ConvNormAct(64, width, 4, 2, 1),
        )
        self.history_spatial = ResidualSpatialBlock(width, 1)
        self.history_temporal = nn.Conv3d(
            width, width, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False
        )
        self.temporal_norm = nn.GroupNorm(8, width)
        self.temporal_act = nn.SiLU()
        self.time_projection = nn.Conv2d(8 * width, 16 * width, 1)
        self.future_geometry_encoder = nn.Sequential(
            ConvNormAct(3, 24, 5, 2, 2),
            ConvNormAct(24, 48, 4, 2, 1),
            ConvNormAct(48, width, 4, 2, 1),
        )
        self.future_spatial = nn.Sequential(
            ResidualSpatialBlock(width, 1),
            ResidualSpatialBlock(width, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, 64, 4, 2, 1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, history_cot: torch.Tensor, history_valid: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        if history_cot.ndim != 5 or history_cot.shape[1:3] != (8, 1):
            raise ValueError(
                f"history_cot must be [B,8,1,H,W], got {tuple(history_cot.shape)}"
            )
        if history_valid.shape != history_cot.shape:
            raise ValueError("history_valid must match history_cot shape")
        if geometry.ndim != 5 or geometry.shape[1:3] != (24, 3):
            raise ValueError(
                f"geometry must be [B,24,3,H,W], got {tuple(geometry.shape)}"
            )
        b, _, _, h, w = history_cot.shape
        if geometry.shape[0] != b or geometry.shape[-2:] != (h, w):
            raise ValueError("COT and geometry batch/grid dimensions must match")
        if h % 8 or w % 8:
            raise ValueError("H and W must be divisible by 8")

        # Keep invalid history distinct from valid COT=0, which is a real value.
        history_valid = history_valid.bool()
        safe_history_cot = torch.where(
            history_valid, history_cot, torch.zeros_like(history_cot)
        )
        hist = torch.cat((safe_history_cot, history_valid.to(history_cot.dtype),
                          geometry[:, :8]), dim=2)
        latent = self.history_encoder(hist.reshape(b * 8, 5, h, w))
        _, c, lh, lw = latent.shape
        latent = self.history_spatial(latent).reshape(b, 8, c, lh, lw)

        # Temporal convolution acts only on the eight observed frames.
        temporal = latent.permute(0, 2, 1, 3, 4).contiguous()
        temporal = temporal + self.temporal_act(
            self.temporal_norm(self.history_temporal(temporal))
        )
        temporal = temporal.reshape(b, 8 * c, lh, lw)
        future = self.time_projection(temporal).reshape(b, 16, c, lh, lw)

        future_geom = geometry[:, 8:].reshape(b * 16, 3, h, w)
        future_geom = self.future_geometry_encoder(future_geom)
        future_geom = future_geom.reshape(b, 16, c, lh, lw)
        future = self.future_spatial(
            (future + future_geom).reshape(b * 16, c, lh, lw)
        )
        pred = self.decoder(future)
        return F.softplus(pred).reshape(b, 16, 1, h, w)


def cot_supervision_masked_mse(pred_log_scaled: torch.Tensor,
                               target_cot: torch.Tensor,
                               valid: torch.Tensor) -> torch.Tensor:
    """Equal-weight per-lead MSE in log1p(COT)/log(101) space.

    target_cot is physical, dimensionless COT. Masked NaNs are removed with
    where() before squaring to prevent 0*NaN contamination.
    """
    if pred_log_scaled.shape != target_cot.shape or valid.shape != target_cot.shape:
        raise ValueError("prediction, physical COT, and mask shapes must match")
    valid = valid.bool()
    safe_target = torch.where(valid, target_cot, torch.zeros_like(target_cot))
    target = torch.log1p(safe_target.clamp_min(0.0)) / torch.log(
        safe_target.new_tensor(101.0)
    )
    error = torch.where(valid, pred_log_scaled - target,
                        torch.zeros_like(pred_log_scaled))
    dims = (0, 2, 3, 4)
    counts = valid.sum(dim=dims)
    per_lead = error.square().sum(dim=dims) / counts.clamp_min(1)
    active = counts > 0
    if not bool(active.any()):
        raise ValueError("batch has no valid COT supervision")
    return per_lead[active].mean()
