"""Per-pixel COT-conditioned modulation of aligned AGRI features."""
import torch
from torch import nn

STATION_ROW_COL = (8, 8)


class COTGatedStationHead(nn.Module):
    """COT at (r,c) generates an affine gate for AGRI features at (r,c)."""
    def __init__(self):
        super().__init__()
        self.agri_encoder = nn.Sequential(
            nn.Conv2d(13, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
        )
        # 1x1 condition is deliberately pointwise: no COT from another pixel
        # may gate the AGRI feature at this pixel.
        self.cot_gate = nn.Conv2d(1, 64, 1)
        self.lead = nn.Embedding(16, 8)
        self.station = nn.Embedding(2, 4)
        self.mlp = nn.Sequential(
            nn.Linear(116, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus(),
        )

    def forward(self, x, c_map, c_stat, geometry, station, lead):
        if x.shape[1:] != (13, 16, 16) or c_map.shape[1:] != (1, 16, 16):
            raise ValueError("requires aligned 13-AGRI and 1-COT 16x16 patches")
        agri = self.agri_encoder(x)
        gamma, beta = self.cot_gate(c_map).chunk(2, dim=1)
        fused = torch.nn.functional.silu(agri * (1.0 + 0.1 * torch.tanh(gamma)) + beta)
        row, col = STATION_ROW_COL
        station_pixel = fused[:, :, row, col]
        station_neighbourhood = fused[:, :, row - 2:row + 3, col - 2:col + 3].mean((-1, -2))
        patch_background = fused.mean((-1, -2))
        zero = torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)
        return self.mlp(torch.cat((patch_background, station_neighbourhood, station_pixel,
                                   c_stat, zero, geometry, self.lead(lead),
                                   self.station(station)), dim=1)).squeeze(1)
