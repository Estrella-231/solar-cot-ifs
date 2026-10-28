"""Bounded DDP training, EMA, complete validation, and resumable checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from dataset import CachedDataset, IndexedDataset
from model import AGRIDyDiff, ModelConfig
from diffusion import DynamicalDiffusion
from samplers import BlockDistributedBatchSampler, StridedValidationSampler

_CHECKPOINT_CPU_GROUP = None


def ema_decay_for_update(update, maximum=0.999):
    if update < 1 or not 0 <= maximum < 1:
        raise ValueError("invalid EMA update/decay")
    return min(maximum, (1 + update) / (10 + update))


def log(event, **values):
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"event": event, **values}), flush=True)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def loader(dataset, *, workers, batch_sampler=None, batch_size=None, sampler=None, seed=42):
    kwargs = dict(num_workers=workers, pin_memory=True,
                  generator=torch.Generator().manual_seed(seed))
    if workers:
        kwargs.update(prefetch_factor=1, persistent_workers=True, multiprocessing_context="spawn")
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **kwargs)


class SyntheticDataset(Dataset):
    """Small reproducible tensors for testing trainer save/resume, never science."""
    def __init__(self, size, config):
        self.size, self.config = size, config

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        c = self.config
        g = torch.Generator().manual_seed(10000 + index)
        def rand(t, channels):
            return torch.randn(t, channels, c.image_size, c.image_size, generator=g)
        h, y = rand(c.history_frames, c.channels), rand(c.future_frames, c.channels)
        hm, fm = torch.ones_like(h, dtype=torch.bool), torch.ones_like(y, dtype=torch.bool)
        fm[..., :index % 4, :] = False
        return {"history": h, "future": y, "geometry": rand(c.frames, 3),
                "history_valid": hm, "future_valid": fm}


@torch.no_grad()
def validate(model, diffusion, val_loader, device, world, seed, sampling_steps, forecast, expected_sequences=None):
    model.eval()
    c = model.config
    # denoising SSE/support, physical forecast SSE/AE/support,
    # persistence SSE and model SSE on matching latest-frame support.
    sums = torch.zeros(10, c.future_frames, c.channels, dtype=torch.float64, device=device)
    sequences = torch.zeros((), dtype=torch.long, device=device)
    for batch_number, data in enumerate(val_loader):
        indices = data.pop("sample_index")
        batch = {k: v.to(device, non_blocking=True) for k, v in data.items()}
        h, y = batch["history"], batch["future"]
        noises, level_list = [], []
        for index in indices.tolist():
            g = torch.Generator(device=device).manual_seed(seed + index)
            level_list.append(torch.randint(1, diffusion.steps + 1, (), device=device, generator=g))
            noises.append(torch.randn(c.frames, c.channels, c.image_size, c.image_size,
                                      device=device, generator=g))
        levels, noise = torch.stack(level_list), torch.stack(noises)
        noisy, target, support, levels = diffusion.prepare(**batch, levels=levels, noise=noise)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(noisy, levels[:, None].expand(-1, c.frames), batch["geometry"],
                         batch["history_valid"], h)
        e = torch.where(support, pred.float() - target, 0.0)[:, c.history_frames:]
        sums[0] += e.square().sum((0, 3, 4), dtype=torch.float64)
        sums[1] += support[:, c.history_frames:].sum((0, 3, 4))
        del noisy, target, support, pred, e, noise, noises
        if forecast:
            # Index-specific sampling is independent of batch size/world-size.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = diffusion.sample(model, h, batch["geometry"], batch["history_valid"],
                    steps=sampling_steps, seed=[seed + 1000000 + i for i in indices.tolist()])
            mask = batch["future_valid"].bool()
            e = torch.where(mask, predicted - y, 0.0)
            if not torch.isfinite(e).all():
                raise RuntimeError("non-finite validation forecast")
            sums[2] += e.square().sum((0, 3, 4), dtype=torch.float64)
            sums[3] += e.abs().sum((0, 3, 4), dtype=torch.float64)
            sums[4] += mask.sum((0, 3, 4))
            common = mask & batch["history_valid"][:, -1:]
            persistence_error = torch.where(common, h[:, -1:] - y, 0.0)
            matched_error = torch.where(common, predicted - y, 0.0)
            sums[5] += persistence_error.square().sum((0, 3, 4), dtype=torch.float64)
            sums[6] += matched_error.square().sum((0, 3, 4), dtype=torch.float64)
            sums[7] += common.sum((0, 3, 4))
            daylight = mask & (batch["geometry"][:, c.history_frames:, 2:3] > 0.5)
            sums[8] += torch.where(daylight, predicted-y, 0.0).square().sum((0,3,4), dtype=torch.float64)
            sums[9] += daylight.sum((0,3,4))
        sequences += len(indices)
        if batch_number % 20 == 0:
            log("validation_progress", local_batch=batch_number + 1, local_sequences=sequences.item(), forecast=forecast)
    if world > 1:
        dist.all_reduce(sums)
        dist.all_reduce(sequences)
    if sequences.item() != (expected_sequences if expected_sequences is not None else len(val_loader.dataset)):
        raise RuntimeError("validation support has missing/duplicate sequences")
    result = {"sequence_count": sequences.item(), "test_payloads_read": 0,
              "denoising_mixed_velocity_mse": (sums[0].sum() / sums[1].sum()).item(),
              "denoising_supported_pixels": sums[1].sum().item(), "forecast_evaluated": forecast}
    if forecast:
        def rmse(sse, count):
            return (sse.sum() / count.sum()).sqrt().item()
        result.update({"forecast_normalized_rmse": rmse(sums[2], sums[4]),
                       "forecast_normalized_mae": (sums[3].sum() / sums[4].sum()).item(),
                       "forecast_rmse_lead_channel": (sums[2] / sums[4].clamp_min(1)).sqrt().cpu().tolist(),
                       "valid_pixels_lead_channel": sums[4].cpu().tolist(),
                       "matched_persistence_rmse": rmse(sums[5], sums[7]),
                       "matched_model_rmse": rmse(sums[6], sums[7]),
                       "mse_skill_vs_persistence": (1 - sums[6].sum() / sums[5].sum().clamp_min(1e-12)).item(),
                       "daylight_reflective_rmse": rmse(sums[8, :, :6], sums[9, :, :6]) if sums[9, :, :6].sum() else None,
                       "all_time_ir_rmse": rmse(sums[2, :, 6:], sums[4, :, 6:]) if sums[4, :, 6:].sum() else None,
                       "sampling_steps": sampling_steps, "ensemble_members": 1,
                       "forecast_seed_policy": "val_seed + 1000000 + individual sequence index"})
        for label, lo, hi in (("0_1h", 0, 4), ("1_2h", 4, 8), ("2_4h", 8, 16)):
            if lo < c.future_frames:
                result["rmse_" + label] = rmse(sums[2, lo:hi], sums[4, lo:hi])
    return result


def save_checkpoint(path, raw, ema, optimizer, epoch, next_batch, step, best, signature, world, rank):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state().cpu()}
    if world > 1:
        rng = [None] * world if rank == 0 else None
        # RNG states are small Python/CPU objects. Do not route them through
        # NCCL's CUDA object-gather staging on a memory-heavy eight-GPU run.
        dist.gather_object(state, rng, dst=0, group=_CHECKPOINT_CPU_GROUP)
    else:
        rng = [state]
    if rank == 0:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": raw.state_dict(), "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "next_batch": next_batch, "global_step": step,
                    "best_forecast_rmse": best, "signature": signature,
                    "rng_by_rank": rng}, tmp)
        tmp.replace(path)
    if world > 1:
        dist.barrier()


def main():
    global _CHECKPOINT_CPU_GROUP
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--calibration")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-batch", type=int, default=4)
    p.add_argument("--forecast-every", type=int, default=5)
    p.add_argument("--sampling-steps", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--expected-world-size", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stop-after-steps", type=int)
    p.add_argument("--stop-file", help="save a resumable checkpoint and stop when this file exists")
    a = p.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
        _CHECKPOINT_CPU_GROUP = dist.new_group(backend="gloo")
    if not a.smoke and world != a.expected_world_size:
        raise RuntimeError(f"formal run requires {a.expected_world_size} allocated GPUs")
    calibration = json.loads(Path(a.calibration).read_text()) if a.calibration else None
    batch_size = a.batch_size or (calibration["batch_per_gpu"] if calibration else 2)
    if not a.smoke and (calibration is None or calibration["status"] != "PASS"):
        raise RuntimeError("calibration must pass before formal training")
    torch.manual_seed(a.seed)
    random.seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    cfg = (ModelConfig(image_size=16, patch_size=4, history_frames=2, future_frames=3,
                       width=32, depth=2, heads=4) if a.smoke else ModelConfig())
    raw, ema = AGRIDyDiff(cfg).to(device), AGRIDyDiff(cfg).to(device)
    ema.load_state_dict(raw.state_dict())
    ema.requires_grad_(False)
    ema.eval()
    net = DDP(raw, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else raw
    # Same initial weights on every rank, independent subsequent diffusion draws.
    torch.manual_seed(a.seed + rank)
    diffusion = DynamicalDiffusion().to(device)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.smoke:
        train_data, val_data = SyntheticDataset(16, cfg), SyntheticDataset(5, cfg)
    else:
        train_data, val_data = CachedDataset("train"), CachedDataset("val", cache_frames=32)
    sampler = BlockDistributedBatchSampler(len(train_data), batch_size, rank, world, a.seed)
    data_loader = loader(train_data, workers=a.workers, batch_sampler=sampler, seed=a.seed + rank)
    val_loader = loader(IndexedDataset(val_data), workers=0, batch_size=a.val_batch,
                        sampler=StridedValidationSampler(len(val_data), rank, world), seed=a.seed)
    monitor_indices = np.linspace(0, len(val_data)-1, min(64,len(val_data)), dtype=int).tolist()
    monitor_loader = loader(IndexedDataset(val_data), workers=0, batch_size=a.val_batch,
                            sampler=monitor_indices[rank::world], seed=a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in ("model.py", "diffusion.py", "data.py", "dataset.py", "samplers.py", "train.py")}
    protocol = {"architecture": raw.metadata(), "epochs": a.epochs, "batch_per_gpu": batch_size,
                "world_size": world, "global_batch": batch_size * world,
                "train_samples": len(train_data), "val_samples": len(val_data),
                "steps_per_epoch": len(sampler), "padded_train_presentations": sampler.padded_presentations,
                "seed": a.seed, "lr": a.lr, "min_lr": 1e-6, "warmup_epochs": 2,
                "weight_decay": 1e-4, "ema_decay": a.ema_decay, "ema_warmup": "min(maximum,(1+updates)/(10+updates))", "precision": "bf16",
                "sampling_steps": a.sampling_steps, "sampling_start": cfg.diffusion_steps, "forecast_every": a.forecast_every,
                "val_batch_per_rank": a.val_batch, "val_seed": 20261022,
                "workers_per_rank": a.workers, "noise_policy": "DyDiff shared sequence level, cosine-0.5, mixed signal and noise; direct unweighted v MSE",
                "early_stopping": False, "paired_forecast_monitor": "raw+EMA, fixed64 validation sequences every epoch", "first_stage": "identity, standardized physical channels",
                "objective": "MSE(v, alpha*M_t(epsilon)-sigma*M_t(x0)); no implicit alpha^2 weighting",
                "training_mask": "all-frame valid prefix intersection for temporal mixtures",
                "selection_metric": "EMA fixed-seed single-member full-val normalized forecast RMSE",
                "training_order": "epoch-shuffled chronological blocks; all train samples plus minimal DDP padding",
                "source_sha256": source_hashes, "synthetic_smoke": a.smoke,
                "test_payloads_read": 0}
    if not a.smoke:
        protocol.update({"region": "Hunan_FULL", "data_version": "hunan_monthly_block_2d_buffer1d_seed42_v1",
                         "pixel_support": "original AGRI channel QA bits AND finite; valid night zeros retained; normalization reused from same frozen Hunan train",
                         "manifest_sha256": train_data.manifest_sha256,
                         "normalization_sha256": train_data.normalization_sha256,
                         "source_trainval_frames": 55733, "source_manifest_sha256": train_data.ready["source_manifest_sha256"],
                         "test_metadata_sequences": 3202, "old_subset_weights_reused": False,
                         "date_buffer_days": 1})
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    epoch0, resume_batch, step, best = 0, 0, 0, float("inf")
    last = out / "last.pt"
    # Rank zero must decide before writing metadata. Independent existence
    # checks race with rank zero creating metadata during first DDP startup.
    flags = torch.zeros(2, device=device, dtype=torch.uint8)
    if rank == 0:
        flags[:] = torch.tensor([last.exists(), (out / "run_metadata.json").exists()],
                                device=device, dtype=torch.uint8)
    if world > 1:
        dist.broadcast(flags, src=0)
    can_resume, has_metadata = flags.cpu().tolist()
    if can_resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise RuntimeError("refusing resume with different data/code/training contract")
        raw.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch0, resume_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["global_step"]
        best = checkpoint["best_forecast_rmse"]
        rng = checkpoint["rng_by_rank"][rank]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"]); torch.cuda.set_rng_state(rng["torch_cuda"])
        del checkpoint
        log("resumed", epoch=epoch0 + 1, next_batch=resume_batch, global_step=step)
    elif has_metadata:
        raise RuntimeError("existing output directory without a resumable checkpoint")
    if rank == 0:
        atomic_json(out / "run_metadata.json", {**protocol, "signature": signature,
                    "account": os.environ.get("USER"), "pbs_job_id": os.environ.get("PBS_JOBID"),
                    "gpu_name": torch.cuda.get_device_name(local_rank),
                    "gpu_total_gib": torch.cuda.get_device_properties(local_rank).total_memory / 2**30})
    if world > 1:
        dist.barrier()
    log("training_started", **{k: protocol[k] for k in ("world_size", "batch_per_gpu", "global_batch", "epochs", "steps_per_epoch")})
    total_steps, warmup = len(sampler) * a.epochs, max(1, len(sampler) * min(2, a.epochs))
    for epoch in range(epoch0, a.epochs):
        sampler.set_epoch(epoch)
        sampler.start_step = resume_batch if epoch == epoch0 else 0
        raw.train()
        epoch_started, previous_end = time.monotonic(), time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        for batch_index, host_batch in enumerate(data_loader, start=sampler.start_step):
            loaded = time.monotonic()
            data_wait = loaded - previous_end
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in host_batch.items()}
            if step < warmup:
                lr = a.lr * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                lr = 1e-6 + (a.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = diffusion.training_loss(net, **batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            decay = ema_decay_for_update(step + 1, a.ema_decay)
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), raw.parameters()):
                    ema_parameter.lerp_(parameter, 1 - decay)
            step += 1
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - loaded
            if step <= 5 or step % 10 == 0:
                if world > 1:
                    timing = torch.tensor([seconds, data_wait], device=device)
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                    seconds, data_wait = timing.tolist()
                reported_loss=loss.detach().clone()
                if world > 1:
                    dist.all_reduce(reported_loss)
                    reported_loss/=world
                log("train_step", epoch=epoch + 1, batch=batch_index + 1, global_step=step,
                    loss=reported_loss.item(), lr=lr, grad_norm=norm.item(), ema_decay=decay, compute_seconds=seconds,
                    data_wait_seconds=data_wait,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)
                if rank == 0:
                    atomic_json(out / "progress.json", {"state": "training", "epoch": epoch + 1,
                                "batch": batch_index + 1, "global_step": step,
                                "test_payloads_read": 0})
            del batch, host_batch, loss
            stop_requested = torch.zeros((), device=device, dtype=torch.uint8)
            if rank == 0 and a.stop_file and Path(a.stop_file).exists():
                stop_requested.fill_(1)
            if world > 1:
                dist.broadcast(stop_requested, src=0)
            stop_now = bool(stop_requested.item()) or (a.stop_after_steps and step >= a.stop_after_steps)
            if step % a.checkpoint_every == 0 or stop_now:
                save_checkpoint(last, raw, ema, optimizer, epoch, batch_index + 1, step, best, signature, world, rank)
            if stop_now:
                log("checkpointed_stop", global_step=step)
                if rank == 0:
                    atomic_json(out / "progress.json", {"state": "checkpointed_stop", "epoch": epoch + 1,
                                                        "batch": batch_index + 1, "global_step": step})
                if world > 1:
                    dist.destroy_process_group()
                return
            previous_end = time.monotonic()
        save_checkpoint(last, raw, ema, optimizer, epoch, len(sampler), step, best, signature, world, rank)
        paired = {}
        for label, monitor_model in (("raw", raw), ("ema", ema)):
            paired[label] = validate(monitor_model, diffusion, monitor_loader, device, world,
                20261022, a.sampling_steps, True, expected_sequences=len(monitor_indices))
        if rank == 0:
            atomic_json(out / f"paired_monitor_epoch_{epoch+1:03d}.json", paired)
        log("paired_monitor_complete", epoch=epoch+1, raw_rmse=paired["raw"]["forecast_normalized_rmse"],
            ema_rmse=paired["ema"]["forecast_normalized_rmse"], sequences=len(monitor_indices))
        forecast = epoch == 0 or (epoch + 1) % a.forecast_every == 0 or epoch + 1 == a.epochs
        log("validation_started", epoch=epoch + 1, forecast=forecast)
        metrics = validate(ema, diffusion, val_loader, device, world, 20261022,
                           a.sampling_steps, forecast)
        improved = forecast and metrics["forecast_normalized_rmse"] < best
        if improved:
            best = metrics["forecast_normalized_rmse"]
        if rank == 0:
            atomic_json(out / f"validation_epoch_{epoch + 1:03d}.json", metrics)
        save_checkpoint(last, raw, ema, optimizer, epoch + 1, 0, step, best, signature, world, rank)
        if improved:
            # Retain a separately named best forecast checkpoint, never select on test.
            save_checkpoint(out / "best.pt", raw, ema, optimizer, epoch + 1, 0, step, best, signature, world, rank)
        log("epoch_complete", epoch=epoch + 1, global_step=step, best_forecast_rmse=best,
            elapsed_seconds=time.monotonic() - epoch_started, **metrics)
        resume_batch = 0
    if rank == 0:
        atomic_json(out / "training_complete.json", {"status": "PASS", "epochs": a.epochs,
                    "global_step": step, "best_forecast_rmse": best, "test_payloads_read": 0})
        atomic_json(out / "progress.json", {"state": "complete", "global_step": step})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
