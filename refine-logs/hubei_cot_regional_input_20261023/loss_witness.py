"""CPU/Gloo two-rank gradient witness, no scientific payloads or GPUs."""
import json, os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from regional_model import loss_parts, ddp_loss


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.manual_seed(42)
    model = nn.Linear(2, 1, bias=False)
    ref = nn.Linear(2, 1, bias=False); ref.load_state_dict(model.state_dict())
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2)/10
    y = torch.tensor([0.1, float('nan'), float('nan'), float('nan'), .2, .4, .6, float('nan')])[:, None]
    mask = torch.isfinite(y)
    start, end = rank*4, (rank+1)*4
    ddp = DDP(model)
    value, count = loss_parts(ddp(x[start:end]), y[start:end], mask[start:end])
    dist.all_reduce(count)
    ddp_loss(value, count, world).backward()
    value_ref, count_ref = loss_parts(ref(x), y, mask)
    (value_ref/count_ref).backward()
    err = float((model.weight.grad-ref.weight.grad).abs().max())
    assert torch.isfinite(model.weight.grad).all() and err < 1e-7, err
    result = dict(state='PASS_CPU_TWO_RANK_GLOBAL_PIXEL_GRADIENT', rank=rank,
                  local_valid_pixels=int(mask[start:end].sum()), global_valid_pixels=int(count),
                  gradient_max_abs_difference=err, masked_nan_target=True, scientific_payloads_read=0)
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
