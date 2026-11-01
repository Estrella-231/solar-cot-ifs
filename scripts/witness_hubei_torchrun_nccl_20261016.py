"""Tiny four-rank NCCL witness for the Hubei SimVP training launch."""
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    value = torch.tensor([local_rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    ok = torch.isfinite(value).all().item() and value.item() == expected
    print(
        f"TORCHRUN_NCCL_WITNESS rank={dist.get_rank()} "
        f"local_rank={local_rank} world={dist.get_world_size()} "
        f"device={torch.cuda.get_device_name(local_rank)} sum={value.item()} pass={ok}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()
    if not ok:
        raise RuntimeError("NCCL all_reduce witness failed")


if __name__ == "__main__":
    main()
