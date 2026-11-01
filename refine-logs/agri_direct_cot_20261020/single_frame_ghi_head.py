"""Frozen single-frame 3x3 station head for the direct-COT experiment.

All four COT-source groups use this same COT-capable architecture. The no-COT
group supplies a zero-valued, normalized COT tensor, keeping parameterization
and initialization identical across the paired comparison.
"""
import torch
from torch import nn


class SingleFrameGHIHead(nn.Module):
    def __init__(self, agri_channels: int = 13, geometry_channels: int = 2,
                 trunk_width: int = 32, cot_width: int = 4):
        super().__init__()
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(agri_channels + geometry_channels, trunk_width, 1),
            nn.SiLU(),
            nn.Conv2d(trunk_width, trunk_width, 1),
            nn.SiLU(),
        )
        self.cot_encoder = nn.Sequential(nn.Conv2d(1, cot_width, 1), nn.SiLU())
        self.readout = nn.Sequential(
            nn.Linear((trunk_width + cot_width) * 3 * 3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, agri, geometry, cot):
        if agri.ndim != 4 or tuple(agri.shape[1:]) != (13, 3, 3):
            raise ValueError("agri must have shape [B,13,3,3]")
        if geometry.shape != (agri.shape[0], 2, 3, 3):
            raise ValueError("geometry must have shape [B,2,3,3]")
        if cot.shape != (agri.shape[0], 1, 3, 3):
            raise ValueError("cot must have shape [B,1,3,3]")
        ag = self.agri_encoder(torch.cat((agri, geometry), dim=1))
        aux = self.cot_encoder(cot)
        return self.readout(torch.cat((ag, aux), dim=1).flatten(1)).squeeze(-1)
