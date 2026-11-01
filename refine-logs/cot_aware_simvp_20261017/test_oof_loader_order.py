"""Witness fold-0-equivalent per-epoch shuffle order for OOF training."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_monthly_simvp_oof_20261019 import make_epoch_loader


def sample_order(loader):
    return torch.cat([batch[0] for batch in loader]).tolist()


def main():
    ds = TensorDataset(torch.arange(257, dtype=torch.int64))
    actual = [sample_order(make_epoch_loader(ds, 19, 0, 420 + epoch))
              for epoch in range(3)]
    expected = []
    for epoch in range(3):
        reference = DataLoader(
            ds, batch_size=19, shuffle=True,
            generator=torch.Generator().manual_seed(420 + epoch),
            num_workers=0, persistent_workers=False,
        )
        expected.append(sample_order(reference))
    assert actual == expected
    assert all(sorted(order) == list(range(len(ds))) for order in actual)
    assert actual[0] != actual[1] != actual[2]
    print("OOF_PER_EPOCH_LOADER_ORDER_WITNESS_PASS")


if __name__ == "__main__":
    main()
