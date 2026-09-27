"""Backbones for the Hunan-only SimVP/SwinLSTM/Motion+Source comparison."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F


HISTORY_FRAMES = 8
FORECAST_FRAMES = 16
TOTAL_GEOMETRY_FRAMES = HISTORY_FRAMES + FORECAST_FRAMES


def validate_forecast_inputs(history, geometry):
    assert history.shape[1:3] == (HISTORY_FRAMES, 13)
    assert geometry.shape[1:3] == (TOTAL_GEOMETRY_FRAMES, 3), (
        "geometry must be chronological [8 history, 16 forecast-target] maps"
    )


class SpectralMix2d(nn.Module):
    """Low-mode AFNO mixer with a depth-wise local residual path."""
    def __init__(self, dim, modes=16):
        super().__init__()
        self.dim, self.modes = dim, modes
        scale = dim ** -0.5
        self.wr = nn.Parameter(scale * torch.randn(dim, dim, modes, modes))
        self.wi = nn.Parameter(scale * torch.randn(dim, dim, modes, modes))
        self.norm = nn.GroupNorm(8, dim)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        skip = x
        z = torch.fft.rfft2(self.norm(x).float(), norm='ortho')
        mh, mw = min(self.modes, z.shape[-2]), min(self.modes, z.shape[-1])
        weight = torch.complex(self.wr[:, :, :mh, :mw], self.wi[:, :, :mh, :mw])
        mixed = torch.einsum('bihw,iohw->bohw', z[:, :, :mh, :mw], weight)
        out = torch.zeros_like(z)
        out[:, :, :mh, :mw] = mixed
        return skip + torch.fft.irfft2(out, s=x.shape[-2:], norm='ortho').to(x.dtype) + self.local(skip)


class AFNOTransformer(nn.Module):
    """Hunan 8-to-16 AFNO-temporal Transformer with per-lead solar geometry.

    The encoder sees 13 AGRI plus the eight historical geometry frames.  Each
    future query is conditioned on its own three geometry maps, so solar time
    is not silently discarded while the predicted AGRI is still 13-channel.
    """
    def __init__(self, dim=128, heads=8, layers=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(16, dim // 2, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(dim // 2, dim, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(dim, dim, 4, 2, 1), nn.SiLU(), SpectralMix2d(dim),
        )
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, 0.0,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerDecoder(layer, layers)
        self.history_pos = nn.Parameter(torch.randn(1, 8, dim) * .02)
        self.future_query = nn.Parameter(torch.randn(1, 16, dim) * .02)
        self.geometry_query = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.SiLU(),
                                            nn.AdaptiveAvgPool2d(1))
        self.spectral = nn.Sequential(SpectralMix2d(dim), nn.GroupNorm(8, dim), nn.SiLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(dim // 4, 13, 3, padding=1),
        )

    def forward(self, history, geometry):
        b, t, c, h, w = history.shape
        validate_forecast_inputs(history, geometry)
        observed = torch.cat((history, geometry[:, :8]), 2)
        latent = self.encoder(observed.reshape(b * 8, 16, h, w))
        _, d, lh, lw = latent.shape
        memory = latent.reshape(b, 8, d, lh, lw).permute(0, 3, 4, 1, 2).reshape(b * lh * lw, 8, d)
        memory = memory + self.history_pos
        future_geometry = geometry[:, HISTORY_FRAMES:]
        gq = self.geometry_query(future_geometry.reshape(b * 16, 3, h, w)).reshape(b, 16, d)
        query = (self.future_query + gq).unsqueeze(1).expand(b, lh * lw, 16, d).reshape(b * lh * lw, 16, d)
        future = self.temporal(query, memory).reshape(b, lh, lw, 16, d).permute(0, 3, 4, 1, 2)
        future = self.spectral(future.reshape(b * 16, d, lh, lw))
        return self.decoder(future).reshape(b, 16, 13, h, w)


class ContinuousSwinLSTM(nn.Module):
    """Official SwinLSTM-D cells with a continuous, 16-channel readout."""
    def __init__(self, vendor_root):
        super().__init__()
        vendor_root = str(Path(vendor_root))
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)
        from SwinLSTM_D import SwinLSTM
        # Two encoder/decoder scales plus the vendor's final 2x ConvTranspose
        # must recover the 256px AGRI grid: 256 / 2 / 2 / 2 * 2 * 2 * 2 = 256.
        self.net = SwinLSTM(img_size=256, patch_size=2, in_chans=16, embed_dim=48,
                            depths_downsample=[2, 2], depths_upsample=[2, 2],
                            num_heads=[3, 6], window_size=8)

    def step(self, frame, states_down, states_up):
        # The upstream SwinLSTM-D cells index their state lists directly.  At
        # the first observed frame there is no recurrent state yet, so seed a
        # list of ``None`` entries rather than passing ``None`` wholesale.
        down_seed = (
            [None] * len(self.net.Downsample.layers)
            if states_down is None
            else states_down
        )
        down, x = self.net.Downsample(frame, down_seed)
        up = []
        for index, layer in enumerate(self.net.Upsample.layers):
            prior = None if states_up is None else states_up[index]
            x, state = layer(x, prior)
            x = self.net.Upsample.upsample[index](x)
            up.append(state)
        return self.net.Upsample.Unembed(x), down, up

    def forward(self, history, geometry):
        validate_forecast_inputs(history, geometry)
        down = up = None
        for k in range(HISTORY_FRAMES):
            _, down, up = self.step(torch.cat((history[:, k], geometry[:, k]), 1), down, up)
        fields = history[:, -1]
        outputs = []
        for k in range(FORECAST_FRAMES):
            raw, down, up = self.step(
                torch.cat((fields, geometry[:, HISTORY_FRAMES + k]), 1), down, up
            )
            fields = raw[:, :13]
            outputs.append(fields)
        return torch.stack(outputs, 1)


class MotionSource(nn.Module):
    """Direct multi-lead backward warp plus learned source field from causal history."""
    def __init__(self, max_displacement_pixels=24.):
        super().__init__()
        self.max_displacement_pixels = float(max_displacement_pixels)
        in_channels = 8 * 13 + 8 * 3 + 16 * 3
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 96, 5, padding=2), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.flow = nn.Conv2d(96, 16 * 2, 3, padding=1)
        self.source = nn.Conv2d(96, 16 * 13, 3, padding=1)

    @staticmethod
    def warp(image, flow):
        batch, _, height, width = image.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype),
                                torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype), indexing='ij')
        grid = torch.stack((xx, yy), -1)[None].expand(batch, -1, -1, -1).clone()
        grid[..., 0] -= 2 * flow[:, 0] / max(width - 1, 1)
        grid[..., 1] -= 2 * flow[:, 1] / max(height - 1, 1)
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, history, geometry):
        validate_forecast_inputs(history, geometry)
        batch, _, _, height, width = history.shape
        history_geometry = geometry[:, :HISTORY_FRAMES]
        future_geometry = geometry[:, HISTORY_FRAMES:]
        features = torch.cat((history.reshape(batch, -1, height, width),
                              history_geometry.reshape(batch, -1, height, width),
                              future_geometry.reshape(batch, -1, height, width)), 1)
        latent = self.encoder(features)
        flows = torch.tanh(self.flow(latent).reshape(batch, 16, 2, height, width)) * self.max_displacement_pixels
        source = self.source(latent).reshape(batch, 16, 13, height, width)
        last = history[:, -1]
        return torch.stack([self.warp(last, flows[:, k]) + source[:, k] for k in range(16)], 1)


class SimVPBackbone(nn.Module):
    """Same two-stage gSTA SimVP recurrence as the matched B0 reference."""
    def __init__(self, hunan_source):
        super().__init__()
        source = str(Path(hunan_source))
        if source not in sys.path:
            sys.path.insert(0, source)
        from simvp import SimVP
        self.net = SimVP(input_channels=16, output_channels=13, input_frames=8,
                         hid_S=32, hid_T=256, N_S=4, N_T=8, mlp_ratio=8., drop=0.,
                         drop_path_prob=.1, spatial_kernel=3, gsta_kernel=21)
        self.future_geometry = nn.Conv2d(3, 13, kernel_size=1)
        nn.init.zeros_(self.future_geometry.weight)
        nn.init.zeros_(self.future_geometry.bias)

    def forward(self, history, geometry):
        validate_forecast_inputs(history, geometry)
        # Two 8-frame stages. Each stage's query-aligned solar geometry is
        # projected onto its outputs; the second stage also sees the first
        # forecast block paired with that block's own geometry as context.
        future = geometry[:, HISTORY_FRAMES:]
        first_base = self.net(torch.cat((history, geometry[:, :HISTORY_FRAMES]), 2))
        first = first_base + self.future_geometry(future[:, :8].flatten(0, 1)).reshape_as(first_base)
        second_input = torch.cat((first, future[:, :8]), 2)
        second_base = self.net(second_input)
        second = second_base + self.future_geometry(future[:, 8:].flatten(0, 1)).reshape_as(second_base)
        return torch.cat((first, second), 1)


def create_model(name, vendor_root, hunan_source=None):
    if name == 'afno_transformer':
        return AFNOTransformer()
    if name == 'simvp':
        assert hunan_source, 'hunan_source is required for SimVP'
        return SimVPBackbone(hunan_source)
    if name == 'swinlstm':
        return ContinuousSwinLSTM(vendor_root)
    if name == 'motion_source':
        return MotionSource()
    raise ValueError(name)
