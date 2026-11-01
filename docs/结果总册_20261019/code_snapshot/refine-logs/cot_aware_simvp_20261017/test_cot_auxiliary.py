import torch
from torch import nn

from cot_auxiliary import frozen_r_cot_loss, mixed_cot_target, station_r_inputs


class TinyRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        return torch.nn.functional.softplus(self.conv(x))


def main():
    torch.manual_seed(7)
    b, t, h, w = 1, 2, 24, 25
    pred = torch.randn(b, t, 13, h, w, requires_grad=True)
    geo = torch.randn(b, t, 2, 3, 16, 16)
    centers = torch.tensor([[[[11, 12], [12, 13]], [[11, 12], [12, 13]]]])
    means = torch.zeros(13); stds = torch.ones(13)
    rmean = torch.zeros(16); rstd = torch.ones(16)
    r_input = station_r_inputs(pred, geo, centers, means, stds, rmean, rstd)
    assert r_input.shape == (b, t, 2, 16, 16, 16)
    cpp = torch.full((b, t, 2, 1, 16, 16), 4.0)
    cpp[..., 0, 0, 0] = float("nan")
    cpp_mask = torch.zeros_like(cpp, dtype=torch.bool); cpp_mask[..., 1:, 1:] = True
    observed = torch.log1p(torch.full_like(cpp, 3.0))
    r_valid = torch.ones_like(cpp, dtype=torch.bool)
    target, valid, from_cpp, from_r = mixed_cot_target(cpp, cpp_mask, observed, r_valid)
    assert valid.all() and from_cpp.sum() > 0 and from_r.sum() > 0
    assert torch.isfinite(target).all()
    model = TinyRetriever().eval().requires_grad_(False)
    loss, out = frozen_r_cot_loss(model, r_input, target, valid)
    assert out.shape == target.shape and torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    # Invalid CPP values are ignored and an invalid R pixel is not supervised.
    r_valid[..., 2, 2] = False
    target2, valid2, cpp2, r2 = mixed_cot_target(cpp, cpp_mask, observed, r_valid)
    assert not valid2[..., 2, 2].any() and not cpp2[..., 2, 2].any() and not r2[..., 2, 2].any()
    print("COT_AUX_CPU_WITNESS_PASS", {
        "loss": float(loss), "valid_pixels": int(valid.sum()),
        "cpp_pixels": int(from_cpp.sum()), "r_fallback_pixels": int(from_r.sum()),
        "simvp_input_grad_l1": float(pred.grad.abs().sum()), "R_parameter_grads": 0,
    })


if __name__ == "__main__":
    main()
