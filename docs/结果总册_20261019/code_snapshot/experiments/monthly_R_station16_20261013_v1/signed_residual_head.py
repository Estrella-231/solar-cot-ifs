"""Capacity-matched signed GHI residual heads on a frozen A baseline."""
from __future__ import annotations

import torch
from torch import nn

from head_candidate_v0 import LocalGHIHead


class SignedResidualHead(nn.Module):
    """Predict a signed GHI adjustment while keeping the A head frozen.

    Both variants have the same trainable parameter count. The auxiliary
    encoder always maps 32 channels to four channels. The no-COT arm feeds it
    the AGRI feature map; the COT arm feeds a channel-wise expansion of the
    normalized COT patch. This keeps the learnable capacity identical.
    The last layer is zero initialized, so the initial forecast is exactly A.
    """

    def __init__(self, variant: str, geometry_channels: int = 2,
                 agri_channels: int = 13, trunk_width: int = 32,
                 auxiliary_width: int = 4):
        super().__init__()
        if variant not in {"AGRI", "AGRI_COT"}:
            raise ValueError(f"unsupported variant: {variant}")
        self.variant = variant
        self.features = LocalGHIHead("A", geometry_channels,
                                     agri_channels=agri_channels,
                                     trunk_width=trunk_width)
        # The frozen A readout is used only to form the shared base forecast.
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.aux_encoder = nn.Sequential(
            nn.Conv2d(trunk_width, auxiliary_width, 1), nn.SiLU())
        fused = trunk_width + auxiliary_width
        self.residual_readout = nn.Sequential(
            nn.Linear(fused * 3 * 3 + 2, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.residual_readout[-1].weight)
        nn.init.zeros_(self.residual_readout[-1].bias)

    def load_frozen_a(self, state_dict):
        self.features.load_state_dict(state_dict, strict=True)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

    def forward(self, agri, geometry, cot, clear_sky):
        if agri.ndim != 4 or tuple(agri.shape[1:]) != (13, 3, 3):
            raise ValueError("agri must have shape [B,13,3,3]")
        if geometry.shape != (agri.shape[0], 2, 3, 3):
            raise ValueError("geometry must have shape [B,2,3,3]")
        if cot.shape != (agri.shape[0], 1, 3, 3):
            raise ValueError("cot must have shape [B,1,3,3]")
        if clear_sky.ndim != 1 or clear_sky.shape[0] != agri.shape[0]:
            raise ValueError("clear_sky must have shape [B]")
        with torch.no_grad():
            base_kt = self.features(agri, geometry)
            base_ghi = base_kt * clear_sky
            encoded = self.features.agri_encoder(
                torch.cat((agri, geometry), dim=1))
        auxiliary_input = (cot.expand(-1, encoded.shape[1], -1, -1)
                           if self.variant == "AGRI_COT" else encoded)
        fused = torch.cat((encoded, self.aux_encoder(auxiliary_input)), dim=1)
        residual = self.residual_readout(torch.cat(
            (fused.flatten(start_dim=1),
             base_ghi[:, None] / 1000.0,
             clear_sky[:, None] / 1000.0), dim=1)).squeeze(-1) * 1000.0
        return torch.clamp_min(base_ghi + residual, 0.0), base_ghi, residual


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
