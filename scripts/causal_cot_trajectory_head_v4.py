"""Station-centred, causal COT spatiotemporal trajectory head.

This candidate deliberately has no global spatial average.  At each target
lead it selects only the station-position feature after causal dilated 3-D
convolutions have incorporated the available COT trajectory and its spatial
neighbourhood.  COT maps after the target lead are masked before any network
operation.
"""
import torch
from torch import nn
from torch.nn import functional as F


STATION_ROW_COL = (8, 8)
HISTORY_STEPS = 8
FORECAST_STEPS = 16


class CausalDilatedConv3d(nn.Module):
    """3-D convolution with left-only temporal padding and symmetric space."""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=0,
                              dilation=(dilation, dilation, dilation))

    def forward(self, value):
        # F.pad order for 5D is (Wleft,Wright,Hleft,Hright,Tleft,Tright).
        d = self.dilation
        return self.conv(F.pad(value, (d, d, d, d, 2 * d, 0)))


class CausalCOTTrajectoryHead(nn.Module):
    """Use historic COT and forecast COT up to, never after, target lead."""
    def __init__(self):
        super().__init__()
        self.agri = nn.Sequential(
            nn.Conv2d(13, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, padding=4, dilation=4), nn.SiLU(),
        )
        # Input channels are the COT value and an explicit causal-validity map.
        layers = []
        channels = 2
        for dilation in (1, 2, 4, 8):
            layers.extend((CausalDilatedConv3d(channels, 32, dilation), nn.SiLU()))
            channels = 32
        self.trajectory = nn.Sequential(*layers)
        self.lead = nn.Embedding(FORECAST_STEPS, 8)
        self.station = nn.Embedding(2, 4)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 32 + 3 + 8 + 4, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus(),
        )

    @staticmethod
    def _check(x, cot_history, cot_forecast, geometry, station, lead):
        if x.shape[1:] != (13, 16, 16):
            raise ValueError("x must be [B,13,16,16]")
        if cot_history.shape[1:] != (HISTORY_STEPS, 1, 16, 16):
            raise ValueError("cot_history must be [B,8,1,16,16]")
        if cot_forecast.shape[1:] != (FORECAST_STEPS, 1, 16, 16):
            raise ValueError("cot_forecast must be [B,16,1,16,16]")
        if geometry.shape[1:] != (3,) or station.ndim != 1 or lead.ndim != 1:
            raise ValueError("geometry/station/lead shapes drifted")
        if not ((lead >= 0).all() and (lead < FORECAST_STEPS).all()):
            raise ValueError("lead must be 0..15")

    def forward(self, x, cot_history, cot_forecast, geometry, station, lead):
        self._check(x, cot_history, cot_forecast, geometry, station, lead)
        batch = len(x)
        # A sample at lead k may only use forecast maps 0..k.  The mask is
        # applied before encoding, so even an accidental noncausal layer cannot
        # see post-target COT values.
        future_index = torch.arange(FORECAST_STEPS, device=x.device)[None]
        future_valid = future_index <= lead[:, None]
        masked_future = cot_forecast * future_valid[:, :, None, None, None]
        cot = torch.cat((cot_history, masked_future), dim=1)
        valid = torch.cat((torch.ones((batch, HISTORY_STEPS), device=x.device,
                                      dtype=x.dtype), future_valid.to(x.dtype)), dim=1)
        valid = valid[:, :, None, None, None].expand(-1, -1, 1, 16, 16)
        # Conv3d expects [B,C,T,H,W].
        trajectory = self.trajectory(torch.cat((cot, valid), dim=2).transpose(1, 2))
        target_time = HISTORY_STEPS + lead
        target_trajectory = trajectory[torch.arange(batch, device=x.device), :, target_time]
        row, col = STATION_ROW_COL
        cot_station = target_trajectory[:, :, row, col]
        agri_station = self.agri(x)[:, :, row, col]
        return self.mlp(torch.cat((agri_station, cot_station, geometry,
                                   self.lead(lead), self.station(station)), dim=1)).squeeze(1)
