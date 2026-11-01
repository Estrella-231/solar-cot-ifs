#!/usr/bin/env python3
"""Isolated capacity-matched AL station head for the T18 R-feature control.

The GHI trunk/readout mirror the existing LocalGHIHead(AP); only the auxiliary
input is supplied by the frozen monthly-R dec1 feature bank.
"""
import torch
from torch import nn


class LocalGHIHeadAL(nn.Module):
    def __init__(self, geometry_channels=2, agri_channels=13,
                 trunk_width=32, auxiliary_width=4):
        super().__init__()
        if geometry_channels < 1:
            raise ValueError("geometry_channels must be positive")
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(agri_channels + geometry_channels, trunk_width, 1),
            nn.SiLU(),
            nn.Conv2d(trunk_width, trunk_width, 1),
            nn.SiLU(),
        )
        self.aux_encoder = nn.Sequential(
            nn.Conv2d(32, auxiliary_width, 1), nn.SiLU())
        fused = trunk_width + auxiliary_width
        self.readout = nn.Sequential(
            nn.Linear(fused * 3 * 3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, agri, geometry, r_feature):
        if agri.ndim != 4 or tuple(agri.shape[1:]) != (13, 3, 3):
            raise ValueError("agri must be [B,13,3,3]")
        if geometry.ndim != 4 or tuple(geometry.shape[1:]) != (2, 3, 3):
            raise ValueError("geometry must be [B,2,3,3]")
        if r_feature.ndim != 4 or tuple(r_feature.shape[1:]) != (32, 3, 3):
            raise ValueError("R dec1 feature must be [B,32,3,3]")
        main = self.agri_encoder(torch.cat((agri, geometry), dim=1))
        aux = self.aux_encoder(r_feature.detach())
        return self.readout(torch.cat((main, aux), dim=1).flatten(1)).squeeze(-1)


def parameter_count(geometry_channels=2):
    return sum(p.numel() for p in LocalGHIHeadAL(geometry_channels).parameters())
