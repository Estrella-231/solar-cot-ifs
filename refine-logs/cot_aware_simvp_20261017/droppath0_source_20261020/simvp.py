from __future__ import annotations

import math

import torch
from torch import nn


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        upsampling: bool = False,
        act_norm: bool = False,
    ) -> None:
        super().__init__()
        if upsampling:
            self.conv: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        self.act_norm = act_norm
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - stride + 1) // 2
        if stride == 1:
            self.conv = BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                act_norm=act_norm,
            )
        elif transpose:
            self.conv = BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                upsampling=True,
                act_norm=act_norm,
            )
        else:
            self.conv = BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                act_norm=act_norm,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class MixMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class AttentionModule(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 21, dilation: int = 3
    ) -> None:
        super().__init__()
        first_kernel = 2 * dilation - 1
        first_padding = (first_kernel - 1) // 2
        dilated_kernel = kernel_size // dilation + (
            (kernel_size // dilation) % 2 - 1
        )
        dilated_padding = dilation * (dilated_kernel - 1) // 2
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=first_kernel,
            padding=first_padding,
            groups=channels,
        )
        self.conv_spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=dilated_kernel,
            stride=1,
            padding=dilated_padding,
            groups=channels,
            dilation=dilation,
        )
        self.conv1 = nn.Conv2d(channels, 2 * channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv0(x)
        attention = self.conv_spatial(attention)
        features, gate = torch.chunk(self.conv1(attention), 2, dim=1)
        return torch.sigmoid(gate) * features


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 21) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(channels, kernel_size)
        self.proj_2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class GASubBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path_prob: float = 0.1,
        kernel_size: int = 21,
        init_value: float = 1e-2,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = SpatialAttention(channels, kernel_size)
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        )
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = MixMlp(
            channels,
            hidden_features=int(channels * mlp_ratio),
            out_features=channels,
            drop=drop,
        )
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(channels))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_1 = self.layer_scale_1[:, None, None]
        scale_2 = self.layer_scale_2[:, None, None]
        x = x + self.drop_path(scale_1 * self.attn(self.norm1(x)))
        return x + self.drop_path(scale_2 * self.mlp(self.norm2(x)))


def _sampling_generator(num_layers: int, reverse: bool = False) -> list[bool]:
    values = [(index % 2 == 1) for index in range(num_layers)]
    return list(reversed(values)) if reverse else values


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int,
    ):
        super().__init__()
        samplings = _sampling_generator(num_layers)
        layers = [
            ConvSC(
                input_channels if index == 0 else hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=2 if sampling else 1,
            )
            for index, sampling in enumerate(samplings)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.layers[0](x)
        latent = first
        for layer in self.layers[1:]:
            latent = layer(latent)
        return latent, first


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        num_layers: int,
        kernel_size: int,
    ):
        super().__init__()
        samplings = _sampling_generator(num_layers, reverse=True)
        self.layers = nn.ModuleList(
            [
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=2 if sampling else 1,
                    transpose=sampling,
                )
                for sampling in samplings
            ]
        )
        self.readout = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, latent: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        hidden = latent
        for layer in self.layers[:-1]:
            hidden = layer(hidden)
        hidden = self.layers[-1](hidden + skip)
        return self.readout(hidden)


class MetaBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float,
        drop: float,
        drop_path_prob: float,
        kernel_size: int = 21,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = GASubBlock(
            in_channels,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path_prob=drop_path_prob,
            kernel_size=kernel_size,
        )
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.in_channels != self.out_channels:
            x = self.reduction(x)
        return x


class MidMetaNet(nn.Module):
    """The exact gSTA MetaFormer translator used by OpenSTL SimVPv2."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        mlp_ratio: float,
        drop: float,
        drop_path_prob: float,
        kernel_size: int = 21,
    ) -> None:
        super().__init__()
        if num_layers < 2 or mlp_ratio <= 1:
            raise ValueError("gSTA requires num_layers >= 2 and mlp_ratio > 1")
        dpr = torch.linspace(1e-2, drop_path_prob, steps=num_layers).tolist()
        layers: list[nn.Module] = [
            MetaBlock(
                input_channels,
                hidden_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path_prob=dpr[0],
                kernel_size=kernel_size,
            )
        ]
        for index in range(1, num_layers - 1):
            layers.append(
                MetaBlock(
                    hidden_channels,
                    hidden_channels,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path_prob=dpr[index],
                    kernel_size=kernel_size,
                )
            )
        layers.append(
            MetaBlock(
                hidden_channels,
                input_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path_prob=drop_path_prob,
                kernel_size=kernel_size,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        hidden = self.layers(x.reshape(batch, frames * channels, height, width))
        return hidden.reshape(batch, frames, channels, height, width)


class SimVP(nn.Module):
    """One eight-frame SimVP pass."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 13,
        input_frames: int = 8,
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path_prob: float = 0.1,
        spatial_kernel: int = 3,
        gsta_kernel: int = 21,
    ) -> None:
        super().__init__()
        if N_S < 2 or N_T < 2:
            raise ValueError("N_S and N_T must both be at least 2")
        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder = Encoder(input_channels, hid_S, N_S, spatial_kernel)
        self.translator = MidMetaNet(
            input_channels=input_frames * hid_S,
            hidden_channels=hid_T,
            num_layers=N_T,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path_prob=drop_path_prob,
            kernel_size=gsta_kernel,
        )
        self.decoder = Decoder(hid_S, output_channels, N_S, spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], received {tuple(x.shape)}")
        batch, frames, channels, height, width = x.shape
        if frames != self.input_frames or channels != self.input_channels:
            raise ValueError(
                f"Expected T/C={self.input_frames}/{self.input_channels}, "
                f"received {frames}/{channels}"
            )
        latent, skip = self.encoder(x.reshape(batch * frames, channels, height, width))
        _, hidden_channels, hidden_height, hidden_width = latent.shape
        latent = latent.reshape(
            batch, frames, hidden_channels, hidden_height, hidden_width
        )
        translated = self.translator(latent)
        translated = translated.reshape(
            batch * frames, hidden_channels, hidden_height, hidden_width
        )
        output = self.decoder(translated, skip)
        return output.reshape(
            batch, frames, self.output_channels, height, width
        )


class RecurrentSimVP(nn.Module):
    """Two eight-frame SimVP passes that produce sixteen AGRI frames."""

    def __init__(self, variant: str, model_config: dict) -> None:
        super().__init__()
        if variant not in {"simvp13", "simvp15"}:
            raise ValueError(f"Unsupported variant: {variant}")
        self.variant = variant
        self.geometry_channels = 0 if variant == "simvp13" else 2
        self.model = SimVP(
            input_channels=13 + self.geometry_channels,
            output_channels=13,
            input_frames=8,
            hid_S=int(model_config["hid_S"]),
            hid_T=int(model_config["hid_T"]),
            N_S=int(model_config["N_S"]),
            N_T=int(model_config["N_T"]),
            mlp_ratio=float(model_config["mlp_ratio"]),
            drop=float(model_config["drop"]),
            drop_path_prob=float(model_config["drop_path"]),
            spatial_kernel=int(model_config["spatial_kernel"]),
            gsta_kernel=int(model_config["gsta_kernel"]),
        )

    def forward(
        self, history: torch.Tensor, geometry: torch.Tensor | None = None
    ) -> torch.Tensor:
        if history.shape[1:3] != (8, 13):
            raise ValueError(
                f"History must be [B,8,13,H,W], received {tuple(history.shape)}"
            )
        if self.variant == "simvp13":
            first = self.model(history)
            second = self.model(first)
        else:
            if geometry is None or geometry.ndim != 5:
                raise ValueError("simvp15 requires geometry [B,16,2,H,W]")
            if geometry.shape[1:3] != (16, 2):
                raise ValueError(
                    f"Geometry must be [B,16,2,H,W], got {tuple(geometry.shape)}"
                )
            first = self.model(torch.cat((history, geometry[:, :8]), dim=2))
            second = self.model(torch.cat((first, geometry[:, 8:16]), dim=2))
        return torch.cat((first, second), dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
