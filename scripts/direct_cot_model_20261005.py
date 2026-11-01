"""Small SimVP/gSTA-style direct COT forecaster for the new monthly contract."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, padding=None):
        if padding is None:
            padding = kernel // 2
        groups = 8 if cout % 8 == 0 else 1
        super().__init__(nn.Conv2d(cin, cout, kernel, stride, padding, bias=False),
                         nn.GroupNorm(groups, cout), nn.SiLU())


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels=96, dilation=1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DirectCOTForecaster(nn.Module):
    """Map eight causal AGRI/geometry frames to sixteen future log1p(COT) maps.

    Args:
        history: normalized physical AGRI channels [B,8,13,H,W].
        geometry: raw [cosSOZ, cosRAA, day_mask] for eight history then sixteen
                  known future target times [B,24,3,H,W].

    Future AGRI and future COT are not arguments to this forward pass.
    """
    def __init__(self, width=96):
        super().__init__()
        self.width = int(width)
        self.history_encoder = nn.Sequential(
            ConvNormAct(16, 32, 5, 2, 2),
            ConvNormAct(32, 64, 4, 2, 1),
            ConvNormAct(64, width, 4, 2, 1),
        )
        self.history_spatial = ResidualSpatialBlock(width, 1)
        self.temporal_spatial = nn.Conv3d(width, width, 3, padding=1, bias=False)
        self.temporal_norm = nn.GroupNorm(8, width)
        self.temporal_act = nn.SiLU()
        self.time_projection = nn.Conv2d(8 * width, 16 * width, 1)
        self.future_geometry_encoder = nn.Sequential(
            ConvNormAct(3, 24, 5, 2, 2),
            ConvNormAct(24, 48, 4, 2, 1),
            ConvNormAct(48, width, 4, 2, 1),
        )
        self.future_spatial = nn.Sequential(ResidualSpatialBlock(width, 1),
                                            ResidualSpatialBlock(width, 2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, history, geometry):
        if history.ndim != 5 or history.shape[1:3] != (8, 13):
            raise ValueError(f"history must be [B,8,13,H,W], got {tuple(history.shape)}")
        if geometry.ndim != 5 or geometry.shape[1:3] != (24, 3):
            raise ValueError(f"geometry must be [B,24,3,H,W], got {tuple(geometry.shape)}")
        b, _, _, h, w = history.shape
        if geometry.shape[-2:] != (h, w) or h % 8 or w % 8:
            raise ValueError("history/geometry grids must match and be divisible by 8")
        hist_input = torch.cat((history, geometry[:, :8]), dim=2)
        latent = self.history_encoder(hist_input.reshape(b * 8, 16, h, w))
        _, c, lh, lw = latent.shape
        latent = self.history_spatial(latent).reshape(b, 8, c, lh, lw)
        # gSTA-style shared spatial encoding followed by a learned 8->16 temporal projection.
        temporal = latent.permute(0, 2, 1, 3, 4)
        temporal = temporal + self.temporal_act(self.temporal_norm(self.temporal_spatial(temporal)))
        temporal = temporal.reshape(b, 8 * c, lh, lw)
        future = self.time_projection(temporal).reshape(b, 16, c, lh, lw)
        future_geom = geometry[:, 8:].reshape(b * 16, 3, h, w)
        future_geom = self.future_geometry_encoder(future_geom).reshape(b, 16, c, lh, lw)
        future = self.future_spatial((future + future_geom).reshape(b * 16, c, lh, lw))
        z = self.decoder(future)
        # Nonnegative log1p(COT), in agreement with the D_CPP target contract.
        return F.softplus(z).reshape(b, 16, 1, h, w)


def masked_log_cot_mse(pred_log1p, cot, valid):
    """Equal-weight mean of per-lead valid-pixel MSE in log1p(COT) space."""
    if pred_log1p.shape != cot.shape or valid.shape != cot.shape:
        raise ValueError("prediction, COT and mask must have identical shapes")
    target = torch.log1p(torch.clamp(cot, min=0.0))
    mask = valid.to(dtype=pred_log1p.dtype)
    # where() prevents masked NaN/Inf labels from leaking through as 0 * NaN.
    error = torch.where(valid, pred_log1p - target, torch.zeros_like(pred_log1p))
    error2 = error.square()
    count = mask.sum(dim=(0, 2, 3, 4))
    per_lead = error2.sum(dim=(0, 2, 3, 4)) / count.clamp_min(1.0)
    active = count > 0
    if not torch.any(active):
        raise ValueError("batch has no supervised target pixels")
    return per_lead[active].mean()
