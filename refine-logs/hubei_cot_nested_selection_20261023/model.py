"""Causal input transforms and frozen teacher response; no data loads at import."""
import numpy as np
import torch
from torch import nn

DIRECT_COLUMNS = list(range(135)) + [306, 307]

class Correction(nn.Module):
    def __init__(self, direct=False):
        super().__init__()
        dims = [137, 36, 11, 1] if direct else [308, 16, 16, 9]
        self.net = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.SiLU(),
                                 nn.Linear(dims[1], dims[2]), nn.SiLU(),
                                 nn.Linear(dims[2], dims[3]))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

def response(coefficient, scale, history_valid):
    return 1000 * (coefficient[279:288] / scale[279:288] +
                   history_valid * coefficient[288:297] / scale[288:297])

def losses(log_out, log_reference, sensitivity):
    physical = ((torch.expm1(log_out) - torch.expm1(log_reference)) / 100).square().mean()
    task = (((log_out - log_reference) * sensitivity).sum(-1) / 1000).square().mean()
    return physical, task

def inference(base, available, sensitivity, log_out, log_original):
    result = torch.clamp_min(base + (sensitivity * (log_out-log_original)).sum(-1), 0)
    return torch.where(available, result, base)

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
