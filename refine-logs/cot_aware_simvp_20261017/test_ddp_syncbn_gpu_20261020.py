#!/usr/bin/env python3
"""Two-GPU witness: SyncBatchNorm + globally pooled loss matches one-device batch."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(5)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(5, 2, 1)

    def forward(self, x):
        return self.conv2(self.act(self.bn(self.conv1(x))))


def main():
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == 2
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(9102)
    init = Tiny().state_dict()
    x_cpu = torch.randn(20, 3, 16, 16, generator=torch.Generator().manual_seed(772))
    y_cpu = torch.randn(20, 2, 16, 16, generator=torch.Generator().manual_seed(773))
    m_cpu = (torch.rand(20, 2, 16, 16, generator=torch.Generator().manual_seed(774)) > .2)

    single = None
    expected_out = expected_grads = expected_buffers = None
    if rank == 0:
        single = Tiny().to(device).train()
        single.load_state_dict(init)
        sx, sy, sm = (t.to(device) for t in (x_cpu, y_cpu, m_cpu))
        sp = single(sx)
        sl = ((sp - sy).square() * sm).sum() / sm.sum()
        sl.backward()
        expected_out = sp.detach()
        expected_grads = {n: p.grad.detach().clone() for n, p in single.named_parameters()}
        expected_buffers = {n: b.detach().clone() for n, b in single.named_buffers()}
    dist.barrier()

    local = Tiny().to(device)
    local.load_state_dict(init)
    local = nn.SyncBatchNorm.convert_sync_batchnorm(local)
    ddp = DDP(local, device_ids=[local_rank], output_device=local_rank)
    ddp.train()
    lo, hi = rank * 10, (rank + 1) * 10
    x, y, m = (t[lo:hi].to(device) for t in (x_cpu, y_cpu, m_cpu))
    p = ddp(x)
    local_num = ((p - y).square() * m).sum()
    global_den = m.sum().to(dtype=local_num.dtype)
    dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
    loss = local_num * world / global_den
    loss.backward()

    gathered = [torch.empty_like(p.detach()) for _ in range(world)]
    dist.all_gather(gathered, p.detach())
    if rank == 0:
        actual_out = torch.cat(gathered, dim=0)
        torch.testing.assert_close(actual_out, expected_out, rtol=2e-5, atol=2e-5)
        for n, q in ddp.module.named_parameters():
            torch.testing.assert_close(q.grad, expected_grads[n], rtol=3e-5, atol=3e-6,
                                       msg=f"gradient mismatch: {n}")
        for n, b in ddp.module.named_buffers():
            torch.testing.assert_close(b, expected_buffers[n], rtol=2e-5, atol=2e-6,
                                       msg=f"buffer mismatch: {n}")
        print("DDP_SYNCBN_GLOBAL_BATCH20_PARITY_PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
