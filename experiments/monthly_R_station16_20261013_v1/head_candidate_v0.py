"""Provisional local A/AC/AP GHI heads for T07 CPU witness.

This is a shape/gradient prototype only. Geometry channel count and all
normalization/data semantics remain subject to T03; do not use for training.
Inputs preserve the native 3x3 pixel layout through pointwise encoders and are
flattened only at the final readout.
"""
import torch
from torch import nn


class LocalGHIHead(nn.Module):
    def __init__(self, variant: str, geometry_channels: int,
                 agri_channels: int = 13, trunk_width: int = 32,
                 auxiliary_width: int = 4):
        super().__init__()
        if variant not in {"A", "AC", "AP"}:
            raise ValueError(f"unsupported variant: {variant}")
        if geometry_channels < 1:
            raise ValueError("geometry_channels must be positive")
        self.variant = variant
        self.agri_channels = agri_channels
        self.geometry_channels = geometry_channels
        self.trunk_width = trunk_width
        self.auxiliary_width = auxiliary_width

        # The common A branch is identical and initialized first for all arms.
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(agri_channels + geometry_channels, trunk_width, 1),
            nn.SiLU(),
            nn.Conv2d(trunk_width, trunk_width, 1),
            nn.SiLU(),
        )
        if variant == "AC":
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(1, auxiliary_width, 1), nn.SiLU())
        elif variant == "AP":
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(trunk_width, auxiliary_width, 1), nn.SiLU())
        else:
            self.aux_encoder = None

        fused_channels = trunk_width + (auxiliary_width if variant != "A" else 0)
        self.readout = nn.Sequential(
            nn.Linear(fused_channels * 3 * 3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, agri, geometry, cot=None):
        expected_agri = (self.agri_channels, 3, 3)
        expected_geometry = (self.geometry_channels, 3, 3)
        if agri.ndim != 4 or tuple(agri.shape[1:]) != expected_agri:
            raise ValueError(f"agri must be [B,{expected_agri[0]},3,3]")
        if geometry.ndim != 4 or tuple(geometry.shape[1:]) != expected_geometry:
            raise ValueError(f"geometry must be [B,{expected_geometry[0]},3,3]")
        features = self.agri_encoder(torch.cat((agri, geometry), dim=1))
        if self.variant == "AC":
            if cot is None or cot.shape != (agri.shape[0], 1, 3, 3):
                raise ValueError("AC requires a co-registered [B,1,3,3] COT field")
            features = torch.cat((features, self.aux_encoder(cot)), dim=1)
        elif self.variant == "AP":
            features = torch.cat((features, self.aux_encoder(features)), dim=1)
        # No spatial pooling: flatten in fixed row-major position order.
        return self.readout(features.flatten(start_dim=1)).squeeze(-1)


def parameter_counts(geometry_channels: int = 2):
    return {v: sum(p.numel() for p in LocalGHIHead(v, geometry_channels).parameters())
            for v in ("A", "AC", "AP")}
