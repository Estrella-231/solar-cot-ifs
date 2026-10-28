"""Check exact globally masked DDP gradient with unequal support per rank."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion import DynamicalDiffusion

rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
m = torch.nn.Linear(1, 1, bias=False).cuda()
with torch.no_grad():
    m.weight.fill_(0.5)
model = DDP(m, device_ids=[int(os.environ["LOCAL_RANK"])])
x = torch.arange(1, 5, device="cuda", dtype=torch.float32).reshape(4, 1) + rank
y = 2 * x
support = torch.arange(4, device="cuda")[:, None] < rank % 4
pred = model(x)
loss = DynamicalDiffusion.masked_loss(pred, y, support)
loss.backward()
num = torch.where(support, 2 * (0.5*x-y) * x, 0.0).sum()
den = support.sum()
dist.all_reduce(num);dist.all_reduce(den)
torch.testing.assert_close(m.weight.grad.squeeze(), num/den)
if rank == 0:
    print("DDP_UNEQUAL_AND_EMPTY_RANK_SUPPORT_GRADIENT_PASS", flush=True)
dist.destroy_process_group()
