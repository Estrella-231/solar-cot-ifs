#!/usr/bin/env python3
"""Train/profile the bounded Hubei 16x16 future-COT pilot.

Training uses only original train dates. An outer date-blocked holdout carved from those dates is scored once after checkpoint selection; official monthly validation, test, and buffer are not read.
CPP remains a retrieval reference, and the R inputs for train dates are not OOF.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler


LEADS = tuple(range(15, 241, 15))
HISTORY_MINUTES = tuple(range(-105, 1, 15))
STATION_PIXELS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LOG_DENOM = math.log(101.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y%m%d%H%M")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def setup_distributed() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    initialized = world > 1
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        # torchrun sets LOCAL_RANK for single-process runs too. Select that
        # device explicitly before CUDA memory/stat APIs are called.
        torch.cuda.set_device(local_rank)
    if initialized:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP launch requires CUDA on every rank")
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world, initialized


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(initialized: bool) -> None:
    if initialized:
        dist.barrier()


def all_reduce_sum(value: torch.Tensor, initialized: bool) -> torch.Tensor:
    if initialized:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class ForecastWindowDataset(Dataset):
    """Causal 8-frame histories and 16 target frames from an audited cache."""

    def __init__(self, cache: Path, split: str,
                 init_dates: set[str] | None = None,
                 allowed_target_dates: set[str] | None = None):
        self.cache = cache
        self.split = split
        self.arrays = {
            name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("geometry", "history_cot_log", "history_valid", "cpp_cot", "cpp_valid")
        }
        rows = read_csv(cache / "bank_rows.csv")
        self.by_time: dict[dt.datetime, int] = {}
        self.split_by_time: dict[dt.datetime, str] = {}
        for row in rows:
            t = parse_time(row["time_bjt"])
            s = row["split"].lower()
            if t in self.by_time:
                raise ValueError(f"duplicate bank timestamp {t}")
            self.by_time[t] = int(row["index"])
            self.split_by_time[t] = s

        pair_rows = [r for r in read_csv(cache / "forecast_pairs.csv")
                     if r["split"].lower() == split]
        pair_by_init: dict[dt.datetime, dict[int, int]] = {}
        pair_counts: dict[tuple[dt.datetime, int], int] = {}
        for row in pair_rows:
            init_t = parse_time(row["init_time_bjt"])
            target_t = parse_time(row["target_time_bjt"])
            lead = int(row["lead_minutes"])
            if target_t - init_t != dt.timedelta(minutes=lead) or lead not in LEADS:
                raise ValueError(f"invalid pair time/lead: {row}")
            if self.by_time.get(init_t) != int(row["init_index"]):
                raise ValueError(f"init index mismatch: {row}")
            if self.by_time.get(target_t) != int(row["target_index"]):
                raise ValueError(f"target index mismatch: {row}")
            if self.split_by_time.get(init_t) != split or self.split_by_time.get(target_t) != split:
                raise ValueError(f"cross-split pair: {row}")
            slot = pair_by_init.setdefault(init_t, {})
            if lead in slot:
                raise ValueError(f"duplicate init/lead pair: {init_t} {lead}")
            slot[lead] = int(row["target_index"])
            pair_counts[(init_t, lead)] = int(row["target_valid_pixels"])

        self.windows: list[dict] = []
        self.allowed_target_dates = allowed_target_dates
        for init_t in sorted(pair_by_init):
            if init_dates is not None and init_t.strftime("%Y%m%d") not in init_dates:
                continue
            hist_times = [init_t + dt.timedelta(minutes=m) for m in HISTORY_MINUTES]
            if any(t not in self.by_time or self.split_by_time.get(t) != split for t in hist_times):
                raise ValueError(f"incomplete or cross-split input history: {init_t}")
            hist_idx = [self.by_time[t] for t in hist_times]
            future_idx: list[int] = []
            future_dates: list[str] = []
            for lead in LEADS:
                target_t = init_t + dt.timedelta(minutes=lead)
                target_i = self.by_time.get(target_t, -1)
                if target_i >= 0 and self.split_by_time.get(target_t) != split:
                    raise ValueError(f"future target crosses split at {init_t} +{lead}")
                if lead in pair_by_init[init_t]:
                    if target_i != pair_by_init[init_t][lead]:
                        raise ValueError(f"pair target map differs at {init_t} +{lead}")
                    actual = int(np.asarray(self.arrays["cpp_valid"][target_i, 0], dtype=bool).sum())
                    if actual != pair_counts[(init_t, lead)]:
                        raise ValueError(f"pair target mask count mismatch at {init_t} +{lead}")
                future_idx.append(target_i)
                future_dates.append(target_t.strftime("%Y%m%d"))
            self.windows.append({"init": init_t, "history": hist_idx,
                                 "future": future_idx, "future_dates": future_dates})
        if not self.windows:
            raise ValueError(f"no supported forecast windows in split={split}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        row = self.windows[item]
        hidx = row["history"]
        history = np.asarray(self.arrays["history_cot_log"][hidx], dtype=np.float32)
        history_valid = np.asarray(self.arrays["history_valid"][hidx], dtype=np.uint8)
        geom_hist = np.asarray(self.arrays["geometry"][hidx], dtype=np.float32)
        geom_future = np.zeros((16, 3, 16, 16), dtype=np.float32)
        target = np.zeros((16, 1, 16, 16), dtype=np.float32)
        target_valid = np.zeros((16, 1, 16, 16), dtype=np.uint8)
        for k, idx in enumerate(row["future"]):
            if idx < 0:
                continue
            geom_future[k] = np.asarray(self.arrays["geometry"][idx], dtype=np.float32)
            if (self.allowed_target_dates is not None
                    and row["future_dates"][k] not in self.allowed_target_dates):
                continue
            target[k] = np.asarray(self.arrays["cpp_cot"][idx], dtype=np.float32)
            target_valid[k] = np.asarray(self.arrays["cpp_valid"][idx], dtype=np.uint8)
        geometry = np.concatenate((geom_hist, geom_future), axis=0)
        return {
            "history": torch.from_numpy(history),
            "history_valid": torch.from_numpy(history_valid),
            "geometry": torch.from_numpy(geometry),
            "target": torch.from_numpy(target),
            "target_valid": torch.from_numpy(target_valid),
            "init_time": row["init"].strftime("%Y%m%d%H%M"),
        }


class GlobalBatchSampler(Sampler[list[int]]):
    """Slice identical shuffled global batches across ranks without padding."""

    def __init__(self, size: int, global_batch: int, rank: int, world: int,
                 seed: int, shuffle: bool):
        if global_batch % world:
            raise ValueError(f"global batch {global_batch} not divisible by world size {world}")
        self.size, self.global_batch = size, global_batch
        self.rank, self.world, self.seed = rank, world, seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.size // self.global_batch

    def __iter__(self) -> Iterable[list[int]]:
        if self.shuffle:
            gen = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.size, generator=gen).tolist()
        else:
            order = list(range(self.size))
        usable = (len(order) // self.global_batch) * self.global_batch
        local = self.global_batch // self.world
        for start in range(0, usable, self.global_batch):
            a = start + self.rank * local
            yield order[a:a + local]


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, size: int, rank: int, world: int):
        self.size, self.rank, self.world = size, rank, world

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.rank, self.size, self.world))

    def __len__(self) -> int:
        return max(0, (self.size - 1 - self.rank) // self.world + 1) if self.rank < self.size else 0


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(batch[k].to(device, non_blocking=True) for k in
                 ("history", "history_valid", "geometry", "target", "target_valid"))


def loss_local_sums(pred: torch.Tensor, target_cot: torch.Tensor,
                    valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = valid.bool()
    safe_target = torch.where(valid, target_cot, torch.zeros_like(target_cot))
    target = torch.log1p(safe_target.clamp_min(0.0)) / math.log(101.0)
    error = torch.where(valid, pred - target, torch.zeros_like(pred))
    sums = error.square().sum(dim=(0, 2, 3, 4))
    counts = valid.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
    return sums, counts


def distributed_equal_lead_loss(pred: torch.Tensor, target: torch.Tensor,
                                valid: torch.Tensor, initialized: bool) -> torch.Tensor:
    local_sums, local_counts = loss_local_sums(pred, target, valid)
    global_counts = all_reduce_sum(local_counts.clone(), initialized)
    active = global_counts > 0
    if not bool(active.any()):
        raise ValueError("global batch has no valid COT target pixels")
    world = dist.get_world_size() if initialized else 1
    # DDP averages gradients over ranks; compensate so the resulting gradient
    # equals one global numerator / denominator per lead, then average leads.
    per_lead = world * local_sums / global_counts.to(local_sums.dtype).clamp_min(1)
    return per_lead[active].mean()


def reduce_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                   initialized: bool) -> dict[str, list[float]]:
    """Physical-unit per-lead RMSE/MAE/bias, globally aggregated across ranks."""
    with torch.no_grad():
        pred_cot = torch.expm1(pred.double() * math.log(101.0))
        mask = valid.bool()
        err = torch.where(mask, pred_cot - target.double(), torch.zeros_like(pred_cot))
        totals = torch.stack((err.square().sum(dim=(0, 2, 3, 4)),
                              err.abs().sum(dim=(0, 2, 3, 4)),
                              err.sum(dim=(0, 2, 3, 4)),
                              mask.sum(dim=(0, 2, 3, 4)).double()))
        totals = all_reduce_sum(totals, initialized)
        count = totals[3].clamp_min(1)
        rmse = torch.sqrt(totals[0] / count)
        mae = totals[1] / count
        bias = totals[2] / count
        return {"rmse": rmse.cpu().tolist(), "mae": mae.cpu().tolist(),
                "bias": bias.cpu().tolist(), "count": totals[3].cpu().tolist()}


def eval_distributed(model: nn.Module, loader: DataLoader, device: torch.device,
                     initialized: bool) -> dict[str, list[float]]:
    model.eval()
    sums = torch.zeros((4, 16), dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            h, hv, geom, target, valid = move_batch(batch, device)
            pred = model(h, hv, geom)
            pcot = torch.expm1(pred.double() * math.log(101.0))
            mask = valid.bool()
            err = torch.where(mask, pcot - target.double(), torch.zeros_like(pcot))
            sums[0] += err.square().sum(dim=(0, 2, 3, 4))
            sums[1] += err.abs().sum(dim=(0, 2, 3, 4))
            sums[2] += err.sum(dim=(0, 2, 3, 4))
            sums[3] += mask.sum(dim=(0, 2, 3, 4)).double()
    sums = all_reduce_sum(sums, initialized)
    count = sums[3].clamp_min(1)
    return {"rmse": torch.sqrt(sums[0] / count).cpu().tolist(),
            "mae": (sums[1] / count).cpu().tolist(),
            "bias": (sums[2] / count).cpu().tolist(),
            "count": sums[3].cpu().tolist()}


def make_inner_split(cache: Path) -> tuple[set[str], set[str], set[str], dict]:
    all_train = ForecastWindowDataset(cache, "train")
    dates = sorted({w["init"].strftime("%Y%m%d") for w in all_train.windows})
    if len(dates) < 30:
        raise ValueError(f"too few train init dates for date holdout: {len(dates)}")
    n_hold = max(1, int(math.ceil(0.20 * len(dates))))
    start = (len(dates) - n_hold) // 2
    hold = set(dates[start:start + n_hold])
    hold_dt = [dt.datetime.strptime(x, "%Y%m%d").date() for x in hold]
    purge = {d for d in dates if any(
        abs((dt.datetime.strptime(d, "%Y%m%d").date() - h).days) <= 1 for h in hold_dt
    )}
    fit = set(dates) - purge
    report = {"train_init_dates": len(dates), "inner_holdout_dates": sorted(hold),
              "purged_dates_plus_minus_one_day": sorted(purge - hold),
              "fit_dates": sorted(fit), "holdout_fraction_by_dates": n_hold / len(dates),
              "selection_scope": "central contiguous 20% of sorted train init dates; purge +/-1 calendar day"}
    return fit, hold, purge, report


def support_summary(ds: ForecastWindowDataset) -> dict:
    pairs = np.zeros(16, dtype=np.int64)
    pixels = np.zeros(16, dtype=np.int64)
    dates: list[set[str]] = [set() for _ in LEADS]
    for window in ds.windows:
        for k, idx in enumerate(window["future"]):
            if idx < 0:
                continue
            if (ds.allowed_target_dates is not None
                    and window["future_dates"][k] not in ds.allowed_target_dates):
                continue
            n = int(np.asarray(ds.arrays["cpp_valid"][idx, 0], dtype=bool).sum())
            if n:
                pairs[k] += 1
                pixels[k] += n
                dates[k].add(window["future_dates"][k])
    return {"pairs_by_lead": pairs.tolist(), "valid_pixels_by_lead": pixels.tolist(),
            "target_dates_by_lead": [len(x) for x in dates],
            "minimum_pairs_per_lead": int(pairs.min()),
            "minimum_target_dates_per_lead": int(min(map(len, dates)))}


def make_loader(ds: Dataset, args, rank: int, world: int, shuffle: bool,
                eval_sampler: bool = False) -> tuple[DataLoader, GlobalBatchSampler | None]:
    if eval_sampler:
        sampler = DistributedEvalSampler(len(ds), rank, world)
        loader = DataLoader(ds, batch_size=max(1, args.global_batch // world), sampler=sampler,
                            num_workers=args.loader_workers, pin_memory=True)
        return loader, None
    sampler = GlobalBatchSampler(len(ds), args.global_batch, rank, world,
                                 args.seed, shuffle)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.loader_workers,
                        pin_memory=True)
    return loader, sampler


def make_optimizer(model: nn.Module, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.min_lr
    )
    return optimizer, scheduler


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def gather_rng(initialized: bool, rank: int, world: int) -> list[dict] | None:
    local = rng_state()
    if not initialized:
        return [local]
    gathered: list[dict | None] = [None] * world if rank == 0 else []
    dist.gather_object(local, gathered if rank == 0 else None, dst=0)
    return gathered if rank == 0 else None


def restore_rng(states: list[dict], rank: int) -> None:
    state = states[min(rank, len(states) - 1)]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def atomic_torch_save(obj: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def get_model_state(model: nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, DDP) else model.state_dict()


def score_from_metrics(metrics: dict[str, list[float]]) -> float:
    values = [x for x, n in zip(metrics["rmse"], metrics["count"]) if n > 0]
    if not values:
        raise ValueError("no active leads in internal holdout score")
    return float(np.mean(values))


def run_profile(args, cache: Path, model: nn.Module, optimizer,
                loader: DataLoader, sampler: GlobalBatchSampler | None,
                device: torch.device, rank: int, world: int, initialized: bool) -> None:
    if device.type != "cuda":
        raise RuntimeError("throughput profile must run inside an allocated CUDA job")
    if sampler is None:
        raise RuntimeError("profile requires the deterministic global-batch sampler")
    model.train()
    step_times: list[float] = []
    step = 0
    epoch = 0
    while step < args.profile_warmup + args.profile_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.profile_warmup:
                torch.cuda.synchronize(device)
                tic = time.perf_counter()
            h, hv, geom, target, valid = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(h, hv, geom)
            loss = distributed_equal_lead_loss(pred, target, valid, initialized)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            if step >= args.profile_warmup:
                step_times.append(time.perf_counter() - tic)
            step += 1
            if step >= args.profile_warmup + args.profile_steps:
                break
        epoch += 1
    measured = step_times[-args.profile_steps:]
    local_peak = torch.cuda.max_memory_allocated(device)
    peak = torch.tensor([local_peak], device=device, dtype=torch.float64)
    if initialized:
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    total_seconds = float(sum(measured))
    report = {
        "state": "PROFILE_COMPLETE", "world_size": world,
        "global_batch": args.global_batch, "local_batch": args.global_batch // world,
        "profile_steps": len(measured), "seconds_per_update": total_seconds / len(measured),
        "global_samples_per_second": args.global_batch * len(measured) / total_seconds,
        "max_rank_peak_allocated_gib": float(peak.item() / (1024 ** 3)),
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "dataset_windows": len(loader.dataset), "cache": str(cache),
        "global_batch_constant": True, "optimizer_updates": len(measured) + args.profile_warmup,
    }
    if is_main(rank):
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"profile_{world}gpu.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


def train(args, cache: Path, model: nn.Module, optimizer, scheduler,
          rank: int, local_rank: int, world: int, initialized: bool,
          device: torch.device) -> None:
    # Keep cache provenance in this function's scope for the completion receipt.
    # The similarly named variable in main() is not visible here.
    report = json.loads((cache / "cache_report.json").read_text(encoding="utf-8"))
    out = Path(args.output)
    if is_main(rank):
        out.mkdir(parents=True, exist_ok=True)
    barrier(initialized)

    fit_dates, hold_dates, purge_dates, split_report = make_inner_split(cache)
    fit_ds = ForecastWindowDataset(cache, "train", init_dates=fit_dates)
    inner_ds = ForecastWindowDataset(cache, "train", init_dates=hold_dates,
                                     allowed_target_dates=hold_dates)
    train_loader, train_sampler = make_loader(fit_ds, args, rank, world, shuffle=True)
    inner_loader = DataLoader(inner_ds, batch_size=args.eval_batch, shuffle=False,
                              num_workers=args.loader_workers, pin_memory=True)
    if len(train_loader) < 1:
        raise ValueError("fit set has fewer windows than one global batch")
    if len(inner_ds) < 100:
        raise ValueError(f"inner holdout too small after date purge: {len(inner_ds)} windows")
    inner_support = support_summary(inner_ds)
    if (inner_support["minimum_pairs_per_lead"] < 100
            or inner_support["minimum_target_dates_per_lead"] < 12):
        raise ValueError(f"train-only selection holdout support gate failed: {inner_support}")

    start_epoch, global_step, best_score, bad_epochs = 0, 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if int(checkpoint["world_size"]) != world:
            raise ValueError("resume world size differs; exact distributed resume is not certified")
        base_model = model.module if isinstance(model, DDP) else model
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint["best_score"])
        bad_epochs = int(checkpoint["bad_epochs"])
        restore_rng(checkpoint["rng_states"], rank)

    log_path = out / "training.jsonl"
    for epoch in range(start_epoch, args.max_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in train_loader:
            h, hv, geom, target, valid = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(h, hv, geom)
            loss = distributed_equal_lead_loss(pred, target, valid, initialized)
            loss.backward()
            optimizer.step()
            epoch_loss[0] += loss.detach().double()
            epoch_loss[1] += 1
            global_step += 1
        epoch_loss = all_reduce_sum(epoch_loss, initialized)

        # Rank 0 evaluates the train-only blocked holdout; other ranks wait.
        if is_main(rank):
            val_metrics = eval_distributed(model.module if isinstance(model, DDP) else model,
                                           inner_loader, device, False)
            val_score = score_from_metrics(val_metrics)
            improved = val_score < best_score - args.min_delta
            if improved:
                best_score, bad_epochs = val_score, 0
            else:
                bad_epochs += 1
            should_stop = bad_epochs >= args.patience
        else:
            val_metrics, val_score, improved, should_stop = None, None, None, None
        if initialized:
            score_t = torch.tensor([val_score or 0.0, float(bool(improved)), float(bool(should_stop))],
                                   dtype=torch.float64, device=device)
            dist.broadcast(score_t, src=0)
            val_score = float(score_t[0].item())
            improved, should_stop = bool(score_t[1].item()), bool(score_t[2].item())
            bad_t = torch.tensor([bad_epochs, best_score], dtype=torch.float64, device=device)
            dist.broadcast(bad_t, src=0)
            bad_epochs, best_score = int(bad_t[0].item()), float(bad_t[1].item())
        scheduler.step()
        states = gather_rng(initialized, rank, world)
        if is_main(rank):
            checkpoint = {
                "model": get_model_state(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                "global_step": global_step, "best_score": best_score,
                "bad_epochs": bad_epochs, "rng_states": states, "world_size": world,
                "global_batch": args.global_batch, "seed": args.seed,
                "selection_metric": "mean per-lead physical COT RMSE on central train-date holdout",
                "split_report": split_report, "inner_holdout_support": inner_support,
            }
            atomic_torch_save(checkpoint, out / "last.pt")
            if improved:
                atomic_torch_save(checkpoint, out / "best.pt")
            record = {
                "epoch": epoch + 1, "global_step": global_step,
                "train_loss_mean_of_global_updates": float((epoch_loss[0] / epoch_loss[1].clamp_min(1)).item()),
                "inner_holdout_equal_lead_rmse_cot": val_score,
                "best_inner_holdout_rmse": best_score, "improved": improved,
                "bad_epochs": bad_epochs, "lr": optimizer.param_groups[0]["lr"],
                "fit_windows": len(fit_ds), "inner_holdout_windows": len(inner_ds),
                "world_size": world, "global_batch": args.global_batch,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        barrier(initialized)
        if should_stop:
            break

    barrier(initialized)
    # The internal outer train-date holdout is scored once after selection; official monthly validation is never
    # consulted for epoch/architecture/loss-weight choices or opened in this experiment.
    if is_main(rank):
        best = torch.load(out / "best.pt", map_location=device, weights_only=False)
        scored_model = model.module if isinstance(model, DDP) else model
        scored_model.load_state_dict(best["model"])
        scored_model.eval()
        val_ds = ForecastWindowDataset(cache, "val")
        val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False,
                                num_workers=args.loader_workers, pin_memory=True)
        preds, keys = [], []
        with torch.no_grad():
            for batch in val_loader:
                h, hv, geom, _, _ = move_batch(batch, device)
                pred = scored_model(h, hv, geom)
                preds.append(pred.float().cpu().numpy())
                keys.extend(batch["init_time"])
        pred_array = np.concatenate(preds, axis=0)
        np.savez_compressed(out / "outer_train_holdout_predictions.npz",
                            init_time_bjt=np.asarray(keys), pred_log_scaled=pred_array)
        metadata = {
            "state": "COMPLETE_TRAINED_HUBEI_INTERNAL_TRAIN_DATE_HOLDOUT_COT_DIAGNOSTIC",
            "scope": "Hubei shared 16x16 station-neighborhood patch; outer evaluation uses original train dates only, not full region or independent test.",
            "split": "internal train-date holdout from frozen train dates; official monthly validation, test, and buffer not read",
            "selected_checkpoint": "best.pt", "selected_epoch": int(best["epoch"]),
            "global_step": global_step, "world_size": world, "global_batch": args.global_batch,
            "seed": args.seed, "fit_windows": len(fit_ds), "inner_holdout_windows": len(inner_ds),
            "train_config": {
                "width": args.width, "max_epochs": args.max_epochs,
                "patience": args.patience, "min_delta": args.min_delta,
                "learning_rate": args.lr, "minimum_learning_rate": args.min_lr,
                "weight_decay": args.weight_decay, "loader_workers": args.loader_workers,
            },
            "inner_split": split_report,
            "inner_holdout_support": inner_support,
            "training_history_source": report.get("history_source"),
            "outer_holdout_history_source": "final frozen full-R history in both arms",
            "cpp_role": "retrieval reference, not independent cloud truth",
            "test_payloads_read": 0, "buffer_payloads_read": 0,
            "outer_train_holdout_init_windows": len(val_ds),
            "best_checkpoint_sha256": sha256(out / "best.pt"),
            "last_checkpoint_sha256": sha256(out / "last.pt"),
            "outer_holdout_predictions_sha256": sha256(out / "outer_train_holdout_predictions.npz"),
            "cache_report_sha256": sha256(cache / "cache_report.json"),
        }
        (out / "training_complete.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
    barrier(initialized)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("profile", "train"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--profile-warmup", type=int, default=5)
    p.add_argument("--profile-steps", type=int, default=30)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--resume", default="")
    args = p.parse_args()
    cache = args.cache.resolve()
    report = json.loads((cache / "cache_report.json").read_text(encoding="utf-8"))
    audit_path = cache.parent / "cache_audit" / "cache_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("strict cache audit receipt is required before profile/train")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("state") != "PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED":
        raise ValueError("cache audit gate did not pass")
    if report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError("cache report does not keep test/buffer closed")
    rank, local_rank, world, initialized = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    fit_dates, hold_dates, _, split_report = make_inner_split(cache)
    if args.mode == "profile":
        ds = ForecastWindowDataset(cache, "train", init_dates=fit_dates)
    else:
        ds = ForecastWindowDataset(cache, "train", init_dates=fit_dates)
    inner_ds = ForecastWindowDataset(cache, "train", init_dates=hold_dates,
                                     allowed_target_dates=hold_dates)
    inner_support = support_summary(inner_ds)
    if (len(inner_ds) < 100 or inner_support["minimum_pairs_per_lead"] < 100
            or inner_support["minimum_target_dates_per_lead"] < 12):
        raise ValueError(f"train-only selection holdout support gate failed before GPU use: {inner_support}")
    print(json.dumps({"train_only_selection_split": split_report,
                      "train_only_selection_support": inner_support}, indent=2), flush=True)
    loader, sampler = make_loader(ds, args, rank, world, shuffle=True)
    base = __import__("sys").path
    script_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(script_dir) not in base:
        base.insert(0, str(script_dir))
    from retrieved_cot_forecaster_20261013 import RetrievedCOTForecaster

    raw_model = RetrievedCOTForecaster(width=args.width).to(device)
    model: nn.Module = raw_model
    if initialized:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
    optimizer, scheduler = make_optimizer(model, args)
    output_dir = Path(args.output)
    expected = (output_dir / f"profile_{world}gpu.json" if args.mode == "profile"
                else output_dir / "last.pt")
    if not args.resume and expected.exists():
        raise FileExistsError(f"refusing to overwrite existing run artifact: {expected}")
    if args.mode == "train" and not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty training directory: {output_dir}")
    if args.mode == "profile":
        run_profile(args, cache, model, optimizer, loader, sampler, device,
                    rank, world, initialized)
    else:
        train(args, cache, model, optimizer, scheduler, rank, local_rank,
              world, initialized, device)
    barrier(initialized)
    if initialized:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
