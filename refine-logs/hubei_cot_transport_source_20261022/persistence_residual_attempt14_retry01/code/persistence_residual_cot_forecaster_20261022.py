"""Persistence-anchored residual head for the Hubei 16x16 COT pilot.

The base encoder/temporal/decoder modules are identical to the frozen pilot.
Only the output mapping changes: forecast log-COT is the most recent valid
observed log-COT plus a bounded learned residual. The final layer is zeroed so
the initial function is exactly persistence wherever a history value exists.
"""
from __future__ import annotations

import torch
from torch import nn

from retrieved_cot_forecaster_20261013 import RetrievedCOTForecaster


class PersistenceResidualCOTForecaster(RetrievedCOTForecaster):
    def __init__(self, width: int = 96, residual_bound: float = 1.0):
        super().__init__(width=width)
        self.residual_bound = float(residual_bound)
        if self.residual_bound <= 0:
            raise ValueError("residual_bound must be positive")
        # Start at the persistence function; training learns only deviations.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def persistence_anchor(history_cot: torch.Tensor,
                           history_valid: torch.Tensor) -> torch.Tensor:
        if history_cot.ndim != 5 or history_cot.shape[1:3] != (8, 1):
            raise ValueError("history_cot must have shape [B,8,1,H,W]")
        if history_valid.shape != history_cot.shape:
            raise ValueError("history_valid must match history_cot shape")
        valid = history_valid.bool()
        indices = torch.arange(8, device=history_cot.device).view(1, 8, 1, 1, 1)
        last = torch.where(valid, indices, torch.full_like(indices, -1)).amax(dim=1)
        gather_index = last.clamp_min(0).unsqueeze(1)
        anchor = history_cot.gather(1, gather_index).squeeze(1)
        return torch.where(last >= 0, anchor, torch.zeros_like(anchor))

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

        valid = history_valid.bool()
        safe_history = torch.where(valid, history_cot, torch.zeros_like(history_cot))
        hist = torch.cat((safe_history, valid.to(history_cot.dtype), geometry[:, :8]), dim=2)
        latent = self.history_encoder(hist.reshape(b * 8, 5, h, w))
        _, c, lh, lw = latent.shape
        latent = self.history_spatial(latent).reshape(b, 8, c, lh, lw)
        temporal = latent.permute(0, 2, 1, 3, 4).contiguous()
        temporal = temporal + self.temporal_act(
            self.temporal_norm(self.history_temporal(temporal))
        )
        temporal = temporal.reshape(b, 8 * c, lh, lw)
        future = self.time_projection(temporal).reshape(b, 16, c, lh, lw)
        future_geom = self.future_geometry_encoder(
            geometry[:, 8:].reshape(b * 16, 3, h, w)
        ).reshape(b, 16, c, lh, lw)
        future = self.future_spatial((future + future_geom).reshape(b * 16, c, lh, lw))
        residual_logits = self.decoder(future).reshape(b, 16, 1, h, w)

        anchor = self.persistence_anchor(history_cot, valid)
        residual = self.residual_bound * torch.tanh(residual_logits)
        return (anchor.unsqueeze(1) + residual).clamp_(0.0, 1.0)
