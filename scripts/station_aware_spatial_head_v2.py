"""Station-aware spatial GHI head for the fixed Hunan SimVP/COT contract.

The station nearest pixel is independently verified as patch [8, 8] for both
Sili and Zhujia. AGRI and COT are fused position-by-position at the first
14-channel convolution. The output keeps the exact station pixel, a 5x5
station neighbourhood, and a learned 4x4 positional encoding of the whole
patch. It intentionally has no COT mean, percentile, or other hand-crafted
global COT statistic.
"""
import torch
from torch import nn


STATION_ROW_COL = (8, 8)


class StationAwareSpatialHead(nn.Module):
    """Same spatial encoder for A/AC; only the COT input plane changes."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(14, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
        )
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.lead = nn.Embedding(16, 8)
        self.station = nn.Embedding(2, 4)
        # 32 exact station + 32 local neighbourhood + 32*4*4 spatial map,
        # then only causal geometry, lead, and station metadata.
        self.mlp = nn.Sequential(
            nn.Linear(32 + 32 + 32 * 4 * 4 + 3 + 8 + 4, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus(),
        )

    def forward(self, x, c_map, geometry, station, lead):
        if x.shape[1:] != (13, 16, 16) or c_map.shape[1:] != (1, 16, 16):
            raise ValueError("requires aligned 13-AGRI and 1-COT 16x16 patches")
        if geometry.shape[1] != 3:
            raise ValueError("requires three causal geometry channels")
        fused = self.encoder(torch.cat((x, c_map), dim=1))
        row, col = STATION_ROW_COL
        station_pixel = fused[:, :, row, col]
        station_neighbourhood = fused[:, :, row - 2:row + 3, col - 2:col + 3].mean((-1, -2))
        full_patch_positions = self.patch_encoder(fused).flatten(1)
        return self.mlp(torch.cat((station_pixel, station_neighbourhood,
                                   full_patch_positions, geometry, self.lead(lead),
                                   self.station(station)), dim=1)).squeeze(1)
