"""CPU smoke checks for the matched full-history input mapping and loss."""
import torch

from train_sili_transport_thickness_pilot_20260927 import TransportThicknessNet, make_input
from train_sili_full_history_pilot_20260928 import (
    FullHistoryNet, full_history_input, matched_full_state, loss_without_clear_penalty,
)


def main():
    torch.manual_seed(42)
    short = TransportThicknessNet()
    short_state = {k: v.detach().clone() for k, v in short.state_dict().items()}
    full = matched_full_state(short_state)
    h = torch.rand(2, 8, 1, 5, 5)
    f = torch.rand(2, 16, 1, 5, 5)
    adv = torch.rand(2, 1, 16, 5, 5)
    motion = torch.rand(2, 2)
    x_short = make_input(h, f, adv, motion)
    x_full = full_history_input(h, f, adv, motion)
    assert x_short.shape == (2, 8, 16, 5, 5), x_short.shape
    assert x_full.shape == (2, 14, 16, 5, 5), x_full.shape
    with torch.no_grad():
        a = short(x_short, adv)
        b = full(x_full, adv)
    delta = float((a - b).abs().max())
    assert delta < 1e-6, delta
    y = torch.rand(2, 16, 5, 5)
    mask = torch.ones_like(y, dtype=torch.bool)
    loss = loss_without_clear_penalty(full(x_full, adv), y, mask)
    loss.backward()
    assert torch.isfinite(loss)
    print({"short_shape": tuple(x_short.shape), "full_shape": tuple(x_full.shape),
           "initial_output_max_abs_diff": delta, "loss": float(loss.detach())})


if __name__ == "__main__":
    main()
