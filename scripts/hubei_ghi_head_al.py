"""Capacity-controlled Hubei GHI head using frozen-R decoder features."""
from __future__ import annotations

import torch
from torch import nn


class HubeiLocalGHIHeadAL(nn.Module):
    """Mirror the Hubei A/AP readout; replace AP's learned aux with frozen R dec1."""
    def __init__(self, trunk_width: int = 32, auxiliary_width: int = 4,
                 r_feature_channels: int = 16):
        super().__init__()
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(15, trunk_width, 1), nn.SiLU(),
            nn.Conv2d(trunk_width, trunk_width, 1), nn.SiLU(),
        )
        self.aux_encoder = nn.Sequential(
            nn.Conv2d(r_feature_channels, auxiliary_width, 1), nn.SiLU(),
        )
        fused = trunk_width + auxiliary_width
        self.readout = nn.Sequential(
            nn.Linear(fused * 3 * 3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, agri: torch.Tensor, r_feature: torch.Tensor) -> torch.Tensor:
        if agri.ndim != 4 or tuple(agri.shape[1:]) != (15, 3, 3):
            raise ValueError("agri must be [B,15,3,3] = 13 AGRI + cosSOZ + cosRAA")
        if r_feature.ndim != 4 or tuple(r_feature.shape[1:]) != (16, 3, 3):
            raise ValueError("r_feature must be frozen Hubei R dec1 [B,16,3,3]")
        main = self.agri_encoder(agri)
        aux = self.aux_encoder(r_feature.detach())
        return self.readout(torch.cat((main, aux), dim=1).flatten(1)).squeeze(-1)


def parameter_count() -> int:
    return sum(p.numel() for p in HubeiLocalGHIHeadAL().parameters())
