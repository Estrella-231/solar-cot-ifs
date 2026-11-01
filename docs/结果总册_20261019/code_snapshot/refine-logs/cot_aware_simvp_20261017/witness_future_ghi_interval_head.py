"""Tiny CPU/GPU-compatible shape and gradient witness for the frozen GHI head."""
import sys
import torch

sys.path.insert(0, sys.argv[1])
from train_loss_pair import IntervalHead

model = IntervalHead()
agri = torch.randn(2, 3, 13, 3, 3)
geometry = torch.randn(2, 3, 2, 3, 3)
cot = torch.randn(2, 3, 1, 3, 3, requires_grad=True)
pred = model(agri, geometry, cot)
pred.sum().backward()
assert pred.shape == (2,) and torch.isfinite(pred).all()
assert model.temporal.weight.grad[:, 15].norm() > 0
assert cot.grad is not None and torch.isfinite(cot.grad).all()
print("INTERVAL_HEAD_3FRAME_COT_GRADIENT_PASS", sum(p.numel() for p in model.parameters()))
