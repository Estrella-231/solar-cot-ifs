"""Signed residual head with one shared, input-derived AGRI/geometry encoder.

The fold-specific A models contribute only their physical GHI predictions.
No fold-specific latent feature is consumed by this model.
"""
from __future__ import annotations

import torch
from torch import nn


class SignedResidualHeadUnified(nn.Module):
    def __init__(self, variant: str, agri_channels: int = 13,
                 geometry_channels: int = 2, cot_channels: int = 1,
                 trunk_width: int = 32, auxiliary_width: int = 4):
        super().__init__()
        if variant not in {"AGRI", "AGRI_COT"}:
            raise ValueError(f"unsupported variant: {variant}")
        self.variant = variant
        self.input_encoder = nn.Sequential(
            nn.Conv2d(agri_channels + geometry_channels, trunk_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(trunk_width, trunk_width, 3, padding=1),
            nn.SiLU(),
        )
        # Identical capacity in both arms: AGRI control uses its local encoded
        # map as auxiliary input; the COT arm uses the normalized COT image.
        self.aux_encoder = nn.Sequential(
            nn.Conv2d(trunk_width, auxiliary_width, 1), nn.SiLU())
        self.residual_readout = nn.Sequential(
            nn.Linear((trunk_width + auxiliary_width) * 3 * 3 + 2, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.residual_readout[-1].weight)
        nn.init.zeros_(self.residual_readout[-1].bias)

    def forward(self, agri, geometry, cot, clear_sky, base_override):
        if agri.ndim != 4 or tuple(agri.shape[1:]) != (13, 3, 3):
            raise ValueError("agri must have shape [B,13,3,3]")
        if geometry.shape != (agri.shape[0], 2, 3, 3):
            raise ValueError("geometry must have shape [B,2,3,3]")
        if cot.shape != (agri.shape[0], 1, 3, 3):
            raise ValueError("cot must have shape [B,1,3,3]")
        if clear_sky.shape != (agri.shape[0],) or base_override.shape != clear_sky.shape:
            raise ValueError("base_override and clear_sky must have shape [B]")
        encoded = self.input_encoder(torch.cat((agri, geometry), dim=1))
        auxiliary = (cot.expand(-1, encoded.shape[1], -1, -1)
                     if self.variant == "AGRI_COT" else encoded)
        fused = torch.cat((encoded, self.aux_encoder(auxiliary)), dim=1)
        residual = self.residual_readout(torch.cat(
            (fused.flatten(start_dim=1), base_override[:, None] / 1000.0,
             clear_sky[:, None] / 1000.0), dim=1)).squeeze(-1) * 1000.0
        prediction = torch.clamp_min(base_override + residual, 0.0)
        return prediction, base_override, residual


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
