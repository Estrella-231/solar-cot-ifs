"""CPU witness that AC50 shrinks only future normalized COT anomalies."""
from __future__ import annotations

import torch

from train_future_ghi_heads_cot_shrink import cot_for_arm


def main():
    cot = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3, 1, 1, 1)
    mask = torch.tensor([[[[[1.]]], [[[1.]]], [[[0.]]]],
                         [[[[1.]]], [[[0.]]], [[[0.]]]],
                         [[[[0.]]], [[[0.]]], [[[0.]]]],
                         [[[[0.]]], [[[0.]]], [[[0.]]]]])
    data = {"cot": cot, "history_cot_mask": mask}
    assert torch.count_nonzero(cot_for_arm(data, "A")) == 0
    assert torch.equal(cot_for_arm(data, "AH"), cot * mask)
    assert torch.equal(cot_for_arm(data, "AC"), cot)
    half = cot_for_arm(data, "AC50")
    assert torch.equal(half * mask, cot * mask)
    assert torch.equal(half * (1.0 - mask), 0.5 * cot * (1.0 - mask))
    assert torch.equal(cot_for_arm(data, "AC50")[:, :, 0, 0, 0],
                       torch.tensor([[0., 1., 1.], [3., 2., 2.5],
                                     [3., 3.5, 4.], [4.5, 5., 5.5]]))
    assert torch.equal(mask[:, :, 0, 0, 0], torch.tensor([[1, 1, 0], [1, 0, 0],
                                                          [0, 0, 0], [0, 0, 0.]]))
    print("FUTURE_GHI_FUTURE_COT_SHRINKAGE_WITNESS_PASS")


if __name__ == "__main__":
    main()
