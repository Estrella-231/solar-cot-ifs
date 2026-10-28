"""Real-data capacity gate, with discarded synthetic-run weights."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from dataset import CachedDataset
from model import AGRIDyDiff, ModelConfig
from diffusion import DynamicalDiffusion


p = argparse.ArgumentParser()
p.add_argument("--calibration")
p.add_argument("--output", required=True)
a = p.parse_args()
reference = {"batch_per_gpu":10, "purpose":"Hunan A100 bounded real-batch profile"}
torch.cuda.set_device(0)
torch.manual_seed(42)
dataset = CachedDataset("train")
total = torch.cuda.get_device_properties(0).total_memory
records = []
selected = None
for batch_size in (reference["batch_per_gpu"], max(1, reference["batch_per_gpu"] - 4)):
    model, ema = AGRIDyDiff(ModelConfig()).cuda(), AGRIDyDiff(ModelConfig()).cuda()
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    diffusion = DynamicalDiffusion().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    started = time.monotonic()
    host = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=0)))
    batch = {k: v.cuda() for k, v in host.items()}
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = diffusion.training_loss(model, **batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for e, v in zip(ema.parameters(), model.parameters()):
                    e.lerp_(v, 0.001)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        record = {"batch_per_gpu": batch_size, "peak_reserved_gib": peak / 2**30,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "loss": loss.item(), "elapsed_seconds": time.monotonic() - started,
                  "stable_with_3gib_ddp_margin": peak + 3*2**30 < total * 0.96}
        records.append(record)
        if record["stable_with_3gib_ddp_margin"]:
            selected = batch_size
        print(json.dumps(record), flush=True)
        del loss
    except torch.cuda.OutOfMemoryError:
        records.append({"batch_per_gpu": batch_size, "status": "OOM"})
        print(json.dumps(records[-1]), flush=True)
    del model, ema, diffusion, optimizer, host, batch
    gc.collect()
    torch.cuda.empty_cache()
    if selected:
        break
if selected is None:
    raise RuntimeError("no real-data batch meets capacity gate")
result = {"status": "PASS", "world_size": 8, "batch_per_gpu": selected,
          "global_batch": selected * 8, "gpu_name": torch.cuda.get_device_name(0),
          "total_gib": total / 2**30, "target_fraction": 0.96,
          "purpose": "real-data capacity gate; optimizer weights discarded",
          "train_sequences_read": sum(r.get("batch_per_gpu",0) for r in records), "test_payloads_read": 0,
          "records": records, "source_synthetic_calibration": reference}
with Path(a.output).open("x") as f:
    json.dump(result, f, indent=2)
print("REAL_BATCH_CAPACITY_PASS", json.dumps(result), flush=True)
