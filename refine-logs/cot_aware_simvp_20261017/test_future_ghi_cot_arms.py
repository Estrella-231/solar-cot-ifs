"""CPU witness for no-COT, history-only COT, and full forecast-COT arms."""
from __future__ import annotations

import torch

from train_future_ghi_heads import cot_for_arm


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
    assert torch.equal(mask[:, :, 0, 0, 0], torch.tensor([[1, 1, 0], [1, 0, 0],
                                                          [0, 0, 0], [0, 0, 0.]]))
    print("FUTURE_GHI_COT_SOURCE_ARMS_WITNESS_PASS")


if __name__ == "__main__":
    main()
