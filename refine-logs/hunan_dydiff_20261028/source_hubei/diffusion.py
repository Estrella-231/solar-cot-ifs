"""DyDiff cosine-0.5 temporal process, reconstructed from pinned official code.

Both observations and independent white noise are transformed by M_t:
    y[0] = x[0]; y[i] = sqrt(1-theta_bar[t])*y[i-1] + sqrt(theta_bar[t])*x[i].
The denoiser directly predicts mixed velocity a*M_t epsilon - s*M_t x0.
DDIM reconstructs physical x0 using M_t^-1,
then maps BOTH signal and noise to M_s at the next step. Identity first stage.
See reference/dydiff.py and ddim_dydiff.py; no natural-image VAE or clipping.
"""
import torch
from torch import nn
import torch.distributed as dist


class DynamicalDiffusion(nn.Module):
    def __init__(self, steps=1000, ratio=0.5):
        super().__init__()
        if steps < 2 or not 0 <= ratio < 1:
            raise ValueError("need steps>=2 and 0<=ratio<1 for stable invertibility")
        self.steps, self.ratio = steps, ratio
        t = torch.arange(steps + 1, dtype=torch.float64) / steps + 0.008
        curve = torch.cos(t / 1.008 * torch.pi / 2).square()
        curve = curve / curve[0]
        # Same clipped stepwise schedules as official make_beta/gamma_schedule.
        beta = (1 - curve[1:] / curve[:-1]).clamp(0, 0.999)
        theta_curve = curve * ratio + (1 - ratio)
        gamma = (1 - theta_curve[1:] / theta_curve[:-1]).clamp(0, 0.999)
        self.register_buffer("alpha_bar", torch.cat((torch.ones(1, dtype=torch.float64),
                                                     (1 - beta).cumprod(0))).float())
        self.register_buffer("theta_bar", torch.cat((torch.ones(1, dtype=torch.float64),
                                                     (1 - gamma).cumprod(0))).float())

    def coefficients(self, levels):
        shape = (-1, 1, 1, 1)
        ab, tb = self.alpha_bar[levels], self.theta_bar[levels]
        return (ab.sqrt().reshape(shape), (1-ab).sqrt().reshape(shape),
                tb.sqrt().reshape(shape), (1-tb).sqrt().reshape(shape))

    def mix(self, x, levels):
        _, _, q, r = self.coefficients(levels)
        previous, frames = x[:, 0], [x[:, 0]]
        for i in range(1, x.shape[1]):
            previous = r * previous + q * x[:, i]
            frames.append(previous)
        return torch.stack(frames, 1)

    def unmix(self, y, levels):
        _, _, q, r = self.coefficients(levels)
        rest = (y[:, 1:] - r[:, None] * y[:, :-1]) / q[:, None]
        return torch.cat((y[:, :1], rest), 1)

    def prepare(self, history, future, geometry, history_valid, future_valid,
                levels=None, noise=None):
        valid = torch.cat((history_valid, future_valid), 1).bool()
        values = torch.cat((history, future), 1)
        if values.shape != valid.shape or not torch.isfinite(values[valid]).all():
            raise ValueError("invalid observations/QA")
        if not torch.isfinite(geometry).all():
            raise ValueError("nonfinite known geometry")
        b, t = values.shape[:2]
        if levels is None:
            levels = torch.randint(1, self.steps + 1, (b,), device=values.device)
        if levels.shape != (b,) or (levels < 1).any() or (levels > self.steps).any():
            raise ValueError("one shared, nonzero diffusion level per sequence")
        clean = torch.where(valid, values, 0.0)
        noise = torch.randn_like(clean) if noise is None else noise
        if noise.shape != clean.shape or not torch.isfinite(noise).all():
            raise ValueError("invalid noise")
        mixed_x, mixed_noise = self.mix(clean, levels), self.mix(noise, levels)
        a, s, _, _ = self.coefficients(levels)
        noisy = a[:, None] * mixed_x + s[:, None] * mixed_noise
        # A mixed target depends on every preceding physical observation.
        # Do not supervise mixtures contaminated by missing satellite values.
        support = valid.clone()
        for i in range(1, support.shape[1]):
            support[:, i] &= support[:, i-1]
        velocity_target = a[:, None] * mixed_noise - s[:, None] * mixed_x
        return noisy, velocity_target, support, levels

    @staticmethod
    def masked_loss(prediction, target, support):
        count = support.sum()
        global_count, world = count.detach().clone(), 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_count)
            world = dist.get_world_size()
        if global_count.item() == 0:
            raise ValueError("no fully supported future mixtures in global batch")
        error = torch.where(support, prediction.float() - target, 0.0)
        return error.square().sum() * world / global_count

    def training_loss(self, model, history, future, geometry, history_valid,
                      future_valid, levels=None, noise=None):
        noisy, target, support, levels = self.prepare(history, future, geometry,
                                                     history_valid, future_valid, levels, noise)
        frame_levels = levels[:, None].expand(-1, noisy.shape[1])
        prediction = model(noisy, frame_levels, geometry, history_valid, history)
        loss = self.masked_loss(prediction, target, support)
        return loss, {"valid_mixed_pixels": support.sum().detach(), "levels": levels.detach()}

    def reverse_step(self, noisy, mixed_noise, levels, next_levels):
        a, s, _, _ = self.coefficients(levels)
        signal = self.unmix((noisy - s[:, None] * mixed_noise) / a[:, None], levels)
        white_noise = self.unmix(mixed_noise, levels)
        an, sn, _, _ = self.coefficients(next_levels)
        return an[:, None] * self.mix(signal, next_levels) + sn[:, None] * self.mix(white_noise, next_levels)

    def reverse_step_velocity(self, noisy, velocity, levels, next_levels):
        a, s, _, _ = self.coefficients(levels)
        # Exact VP identities, without subtraction/division by tiny alpha.
        signal = self.unmix(a[:, None] * noisy - s[:, None] * velocity, levels)
        white_noise = self.unmix(s[:, None] * noisy + a[:, None] * velocity, levels)
        an, sn, _, _ = self.coefficients(next_levels)
        return an[:, None] * self.mix(signal, next_levels) + sn[:, None] * self.mix(white_noise, next_levels)

    @torch.no_grad()
    def sample(self, model, history, geometry, history_valid, steps=25, seed=42):
        core = model.module if hasattr(model, "module") else model
        c = core.config
        if steps < 2 or steps > self.steps:
            raise ValueError("sampling steps must be within [2,diffusion steps]")
        if c.diffusion_steps != self.steps:
            raise ValueError("model/diffusion schedules must match")
        history = torch.where(history_valid.bool(), history, 0.0)
        if not torch.isfinite(history).all() or not torch.isfinite(geometry).all():
            raise ValueError("invalid conditioning")
        # Every step-count starts at the SAME terminal level; only numerical
        # integration resolution changes. Terminal cosine SNR is negligible.
        times = torch.linspace(self.steps, 1, steps, device=history.device).round().long()
        times = torch.cat((times, times.new_zeros(1)))
        shape = (history.shape[0], c.frames, c.channels, c.image_size, c.image_size)
        if isinstance(seed, (list, tuple)):
            if len(seed) != shape[0]:
                raise ValueError("one seed per sequence required")
            white = torch.stack([torch.randn(shape[1:], device=history.device,
                generator=torch.Generator(device=history.device).manual_seed(s)) for s in seed])
        else:
            g = torch.Generator(device=history.device).manual_seed(seed)
            white = torch.randn(shape, device=history.device, generator=g)
        x = self.mix(white, times[0].expand(history.shape[0]))
        previous_training = model.training
        model.eval()
        try:
            for now, nxt in zip(times[:-1], times[1:]):
                levels = now.expand(history.shape[0])
                velocity = core.predict_velocity(x, levels[:, None].expand(-1, c.frames),
                                                  geometry, history_valid, history).float()
                x = self.reverse_step_velocity(x, velocity, levels, nxt.expand_as(levels))
            return x[:, c.history_frames:]
        finally:
            model.train(previous_training)
