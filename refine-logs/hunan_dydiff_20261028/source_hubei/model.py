"""AGRI DyDiff adaptation with a matched factorized DiT backbone.

Dynamical diffusion equations follow thuml/dynamical-diffusion, pinned in
PROVENANCE.json. The DiT building blocks reuse our MIT-licensed DFoT adaptation.
This pixel-space 13-channel implementation is not the original latent pipeline.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 256
    patch_size: int = 8
    channels: int = 13
    geometry_channels: int = 3
    history_frames: int = 8
    future_frames: int = 16
    width: int = 384
    depth: int = 12  # Each stage contains a spatial and a temporal block.
    heads: int = 6
    mlp_ratio: int = 4
    gradient_checkpointing: bool = True
    diffusion_steps: int = 1000

    def __post_init__(self):
        if any(v <= 0 for v in (self.image_size, self.patch_size, self.channels,
                               self.geometry_channels, self.history_frames,
                               self.future_frames, self.width, self.depth,
                               self.heads, self.mlp_ratio)):
            raise ValueError("model dimensions must be positive")
        if self.image_size % self.patch_size or self.width % self.heads or self.width % 4:
            raise ValueError("image/patch, width/heads and width/4 must be divisible")

    @property
    def frames(self):
        return self.history_frames + self.future_frames


def sincos(values, dim):
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        dim // 2, device=values.device, dtype=torch.float32) / (dim // 2))
    phase = values.float()[..., None] * frequencies
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(
            2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class AdaLNBlock(nn.Module):
    """Pre-norm residual attention/MLP with per-token noise conditioning."""
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * ratio),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(dim * ratio, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, condition):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class AGRIDyDiff(nn.Module):
    """Joint mixed-sequence denoiser with explicit clean historical conditioning.

    noisy: all 24 AGRI frames at their own noise levels.
    levels: B,24 integers/continuous levels in [0,1000].
    geometry: B,24,3,H,W; known astronomical covariates at matching timestamps.
    history_valid: B,8,13,H,W; NEVER accepts a future observation QA tensor.
    The Transformer directly predicts mixed velocity. Epsilon reconstruction uses
    the complete uncompressed noisy input; all supported frames supervise.
    """
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        c = config
        grid = c.image_size // c.patch_size
        self.grid = grid
        self.stem = nn.Conv2d(3 * c.channels + c.geometry_channels + 2, c.width,
                              kernel_size=c.patch_size, stride=c.patch_size)
        self.noise_embedding = nn.Sequential(nn.Linear(256, c.width), nn.SiLU(),
                                             nn.Linear(c.width, c.width))
        self.role_embedding = nn.Embedding(2, c.width)
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        spatial = torch.cat((sincos(yy.flatten(), c.width // 2),
                             sincos(xx.flatten(), c.width // 2)), -1)
        temporal = sincos(torch.arange(c.frames) - (c.history_frames - 1), c.width)
        self.register_buffer("spatial_position", spatial, persistent=False)
        self.register_buffer("temporal_position", temporal, persistent=False)
        roles = torch.cat((torch.zeros(c.history_frames), torch.ones(c.future_frames))).long()
        self.register_buffer("roles", roles, persistent=False)
        self.spatial_blocks = nn.ModuleList([
            AdaLNBlock(c.width, c.heads, c.mlp_ratio) for _ in range(c.depth)])
        self.temporal_blocks = nn.ModuleList([
            AdaLNBlock(c.width, c.heads, c.mlp_ratio) for _ in range(c.depth)])
        self.final_norm = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c.width, 2 * c.width))
        self.head = nn.Linear(c.width, c.patch_size ** 2 * c.channels)
        # Trainable pixelwise paths bypass the rank-deficient patch stem.
        # Their coefficients depend on diffusion time and forecast lead.
        self.full_resolution_coeff = nn.Sequential(nn.SiLU(),nn.Linear(c.width,2*c.channels))
        from diffusion import DynamicalDiffusion
        schedule = DynamicalDiffusion(steps=c.diffusion_steps)
        self.register_buffer("mix_q", schedule.theta_bar.sqrt(), persistent=False)
        self.register_buffer("mix_r", (1-schedule.theta_bar).sqrt(), persistent=False)
        self.register_buffer("vp_alpha", schedule.alpha_bar.sqrt(), persistent=False)
        self.register_buffer("vp_sigma", (1-schedule.alpha_bar).sqrt(), persistent=False)
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.stem.weight.view(c.width, -1))
        nn.init.zeros_(self.stem.bias)
        for block in (*self.spatial_blocks, *self.temporal_blocks):
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.full_resolution_coeff[-1].weight)
        nn.init.zeros_(self.full_resolution_coeff[-1].bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _block(self, module, x, condition):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, condition, use_reentrant=False)
        return module(x, condition)

    def predict_velocity(self, noisy, levels, geometry, history_valid, history):
        c = self.config
        b = noisy.shape[0]
        expected = (b, c.frames, c.channels, c.image_size, c.image_size)
        if tuple(noisy.shape) != expected or tuple(levels.shape) != (b, c.frames):
            raise ValueError(f"noisy/levels shape mismatch: expected {expected}, B,T")
        if tuple(geometry.shape) != (b, c.frames, c.geometry_channels, c.image_size, c.image_size):
            raise ValueError("geometry must cover all 8 history + 16 matching future times")
        if tuple(history_valid.shape) != (b, c.history_frames, c.channels, c.image_size, c.image_size):
            raise ValueError("only historical validity may be passed to the denoiser")
        observed_qa = torch.cat((history_valid.to(noisy.dtype),
                                torch.zeros_like(noisy[:, c.history_frames:])), 1)
        if tuple(history.shape) != tuple(history_valid.shape):
            raise ValueError("clean historical condition shape mismatch")
        clean_history = torch.where(history_valid.bool(), history, 0.0)
        condition_images = torch.cat((clean_history, torch.zeros_like(noisy[:, c.history_frames:])), 1)
        observed = torch.zeros_like(noisy[:, :, :1])
        observed[:, :c.history_frames] = 1
        first_future = torch.zeros_like(observed)
        first_future[:, c.history_frames] = 1
        stem_input = torch.cat((noisy, condition_images.to(noisy.dtype),
                                geometry.to(noisy.dtype), observed_qa, observed, first_future), 2)
        x = self.stem(stem_input.flatten(0, 1)).flatten(2).transpose(1, 2)
        p = self.grid ** 2
        x = x.reshape(b, c.frames, p, c.width)
        x = x + self.spatial_position[None, None] + self.temporal_position[None, :, None]
        x = x + self.role_embedding(self.roles)[None, :, None]
        condition = self.noise_embedding(sincos(levels, 256))[:, :, None].expand(-1, -1, p, -1)
        for spatial, temporal in zip(self.spatial_blocks, self.temporal_blocks):
            x = self._block(spatial, x.reshape(b * c.frames, p, c.width),
                            condition.reshape(b * c.frames, p, c.width))
            x = x.reshape(b, c.frames, p, c.width).transpose(1, 2).reshape(b * p, c.frames, c.width)
            temporal_condition = condition.transpose(1, 2).reshape(b * p, c.frames, c.width)
            x = self._block(temporal, x, temporal_condition)
            x = x.reshape(b, p, c.frames, c.width).transpose(1, 2)
        shift, scale = self.final_modulation(condition).chunk(2, -1)
        x = self.head(self.final_norm(x) * (1 + scale) + shift)
        q = c.patch_size
        x = x.reshape(b, c.frames, self.grid, self.grid, q, q, c.channels)
        coarse = x.permute(0, 1, 6, 2, 4, 3, 5).reshape(expected).float()
        coefficients = self.full_resolution_coeff(condition[:,:,0,:]+self.temporal_position[None]).float()
        noisy_coeff, historical_coeff = coefficients.chunk(2,-1)
        # A causal last-frame baseline, mixed by the SAME M_t as the state.
        baseline = torch.cat((clean_history,clean_history[:,-1:].expand(-1,c.future_frames,-1,-1,-1)),1)
        q = self.mix_q[levels[:,0].long()][:,None,None,None]
        r = self.mix_r[levels[:,0].long()][:,None,None,None]
        prior, mixed = baseline[:,0], [baseline[:,0]]
        for i in range(1,c.frames):
            prior = r*prior+q*baseline[:,i]
            mixed.append(prior)
        mixed_baseline = torch.stack(mixed,1)
        return coarse + noisy_coeff[...,None,None]*noisy.float() + historical_coeff[...,None,None]*mixed_baseline

    def epsilon_from_velocity(self, noisy, levels, velocity):
        a = self.vp_alpha[levels.long()][..., None, None, None]
        s = self.vp_sigma[levels.long()][..., None, None, None]
        # Full-resolution path around the compressive Transformer stem.
        # Fused addcmul avoids one physical-sized intermediate allocation.
        return torch.addcmul(s * noisy.float(), a, velocity.float())

    def forward(self, noisy, levels, geometry, history_valid, history):
        return self.predict_velocity(noisy, levels, geometry, history_valid, history)

    def metadata(self):
        return {"architecture": "AGRI_DyDiff_factorized_v5_fullres_direct_v", "config": asdict(self.config),
                "parameters": sum(p.numel() for p in self.parameters()),
                "prediction_target": "mixed_velocity_unweighted_mse",
                "network_output": "mixed_velocity", "trainable_full_resolution_noise_path": True,
                "full_resolution_history_path": "M_t(clean history + last-frame persistence)", "full_resolution_noise_skip": True,
                "sampler": "stable velocity identity; fixed terminal start", "future_qa_in_model_input": False}


