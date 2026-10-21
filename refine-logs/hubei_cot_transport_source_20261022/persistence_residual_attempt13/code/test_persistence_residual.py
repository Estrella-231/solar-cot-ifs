from __future__ import annotations

import torch

from persistence_residual_cot_forecaster_20261022 import PersistenceResidualCOTForecaster


def test_zero_head_is_last_valid_persistence_and_bounded() -> None:
    torch.manual_seed(42)
    model = PersistenceResidualCOTForecaster(width=16)
    x = torch.rand(2, 8, 1, 16, 16)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[0, -1, 0, 2, 3] = False
    valid[0, :-1, 0, 2, 3] = False
    valid[0, -3, 0, 2, 3] = True
    valid[1, :, 0, 4, 5] = False
    geom = torch.rand(2, 24, 3, 16, 16)

    with torch.no_grad():
        y = model(x, valid, geom)
        anchor = model.persistence_anchor(x, valid)

    assert y.shape == (2, 16, 1, 16, 16)
    assert torch.all(y >= 0) and torch.all(y <= 1)
    assert torch.allclose(y, anchor[:, None].expand_as(y))
    assert anchor[0, 0, 2, 3] == x[0, -3, 0, 2, 3]
    assert anchor[1, 0, 4, 5] == 0


def test_residual_changes_forecast_without_changing_contract() -> None:
    model = PersistenceResidualCOTForecaster(width=16)
    x = torch.full((1, 8, 1, 16, 16), 0.3)
    valid = torch.ones_like(x, dtype=torch.bool)
    geom = torch.zeros(1, 24, 3, 16, 16)
    with torch.no_grad():
        model.decoder[-1].bias.fill_(0.5)
        y = model(x, valid, geom)
    assert y.shape == (1, 16, 1, 16, 16)
    assert torch.all(y > 0.3)
    assert torch.all(y <= 1)


if __name__ == "__main__":
    test_zero_head_is_last_valid_persistence_and_bounded()
    test_residual_changes_forecast_without_changing_contract()
    print("PERSISTENCE_RESIDUAL_UNIT_SMOKE_PASS")
