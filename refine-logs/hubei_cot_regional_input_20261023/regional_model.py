"""Regional COT retrieval model and safe valid-pixel loss components."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b),
                         nn.SiLU(), nn.Conv2d(b, b, 3, padding=1, bias=False),
                         nn.BatchNorm2d(b), nn.SiLU())


class COTUNet(nn.Module):
    """Original base16 two-downsampling architecture, trained anew on 256²."""
    def __init__(self):
        super().__init__()
        self.enc1 = block(16, 16); self.enc2 = block(16, 32)
        self.middle = block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16); self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.enc1(x); b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat((self.up2(c), b), 1))
        return F.softplus(self.head(self.dec1(torch.cat((self.up1(d), a), 1))))


def loss_parts(pred, target, mask):
    # Mask before nonlinear arithmetic; 0*NaN would still be NaN.
    error = torch.where(mask, pred.float()-target.float(), torch.zeros_like(pred, dtype=torch.float32))
    return F.huber_loss(error, torch.zeros_like(error), reduction='sum', delta=0.1), mask.sum(dtype=torch.float64)


def ddp_loss(local_sum, global_valid_pixels, world):
    # DDP averages gradients over ranks; multiply by world to recover the
    # single global valid-pixel mean even when rank masks have unequal counts.
    return local_sum * (world/global_valid_pixels.clamp_min(1))
