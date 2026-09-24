"""Fresh Hunan station GHI head with a causal COT-trajectory feature branch.

The image trunk uses residual blocks, CBAM, a centre-pixel branch and a
five-dimensional solar/time MLP.  Its parameters are initialized and trained
from scratch for each Hunan station; no external Direct-GHI checkpoint is
loaded.  The 16-channel image contract and causal COT fusion are specific to
this experiment.
"""
import torch
from torch import nn
from torch.nn import functional as F

STATION_ROW_COL, HISTORY_STEPS, FORECAST_STEPS = (8, 8), 8, 16


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        value = F.silu(self.bn1(self.conv1(x)))
        return F.silu(self.bn2(self.conv2(value)) + x)


class CBAM(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // reduction), nn.SiLU(),
                                nn.Linear(ch // reduction, ch), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        channel = self.fc(x.mean(dim=(2, 3))).view(len(x), -1, 1, 1)
        value = x * channel
        spatial = self.spatial(torch.cat((value.mean(1, keepdim=True),
                                          value.max(1, keepdim=True)[0]), 1))
        return value * spatial


class CenterPixelBranch(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        center = x[:, :, h // 2, w // 2]
        def moments(radius):
            y, z = h // 2 - radius, w // 2 - radius
            value = x[:, :, y:y + 2 * radius + 1, z:z + 2 * radius + 1].reshape(b, c, -1)
            return value.mean(-1), value.std(-1)
        mean3, std3 = moments(1)
        mean5, std5 = moments(2)
        flat = x.reshape(b, c, -1)
        return torch.cat((center, mean3, std3, mean5, std5, flat.mean(-1), flat.std(-1)), 1)


class CausalDilatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv3d(in_channels, out_channels, 3, dilation=(dilation,) * 3)

    def forward(self, x):
        d = self.dilation
        return self.conv(F.pad(x, (d, d, d, d, 2 * d, 0)))


class CausalCOTTrajectorySolarResNetV5(nn.Module):
    """Same-sized paired head; groups differ only by zeroed COT trajectories."""
    def __init__(self, image_channels=16, metadata_dim=5, base_ch=32, dropout=.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(image_channels, base_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base_ch), nn.SiLU())
        self.stage1 = nn.Sequential(ResBlock(base_ch), ResBlock(base_ch))
        self.down1 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(base_ch * 2), nn.SiLU())
        self.stage2 = nn.Sequential(ResBlock(base_ch * 2), ResBlock(base_ch * 2))
        self.down2 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(base_ch * 4), nn.SiLU())
        self.stage3 = nn.Sequential(ResBlock(base_ch * 4), ResBlock(base_ch * 4))
        self.attention = CBAM(base_ch * 4)
        self.gap, self.center = nn.AdaptiveAvgPool2d(1), CenterPixelBranch()
        self.center_proj = nn.Sequential(nn.Linear(7 * image_channels, 64), nn.SiLU(), nn.Dropout(dropout / 2))
        self.meta_mlp = nn.Sequential(nn.Linear(metadata_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        layers, channels = [], 2
        for dilation in (1, 2, 4, 8):
            layers += [CausalDilatedConv3d(channels, 32, dilation), nn.SiLU()]
            channels = 32
        self.trajectory = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(base_ch * 4 + 64 + 64 + 32, 256), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1), nn.Softplus())

    def forward(self, image, metadata, history, forecast, lead):
        if image.shape[1:] != (16, 16, 16) or metadata.shape[1:] != (5,):
            raise ValueError('SolarResNet v5 image/metadata contract drift')
        if history.shape[1:] != (8, 1, 16, 16) or forecast.shape[1:] != (16, 1, 16, 16):
            raise ValueError('trajectory contract drift')
        if not bool(((lead >= 0) & (lead < 16)).all()):
            raise ValueError('lead must be 0..15')
        x = self.stem(image); x = self.stage1(x); x = self.down1(x); x = self.stage2(x)
        x = self.attention(self.stage3(self.down2(x)))
        image_feature = torch.cat((self.gap(x).flatten(1), self.center_proj(self.center(image))), 1)
        valid_future = torch.arange(16, device=image.device)[None] <= lead[:, None]
        cot = torch.cat((history, forecast * valid_future[:, :, None, None, None]), 1)
        validity = torch.cat((torch.ones((len(image), 8), device=image.device, dtype=image.dtype),
                              valid_future.to(image.dtype)), 1)[:, :, None, None, None].expand(-1, -1, 1, 16, 16)
        value = self.trajectory(torch.cat((cot, validity), 2).transpose(1, 2))
        cot_feature = value[torch.arange(len(image), device=image.device), :, 8 + lead, 8, 8]
        return self.head(torch.cat((image_feature, self.meta_mlp(metadata), cot_feature), 1)).squeeze(1)
