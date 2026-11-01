#!/usr/bin/env python3
"""Frozen-R COT auxiliary loss hooks for the matched attempt13 SimVP pair."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PATCH_ROW, PATCH_COL, PATCH_SIZE = 103, 118, 16
LEADS = 16


def sha_state_dict(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _stamp(path: str) -> str:
    return Path(path).name[:12]


def patch_dataset_index(dataset_cls) -> None:
    if getattr(dataset_cls, "_attempt13_index_patched", False):
        return
    original = dataset_cls.__getitem__

    def indexed_getitem(self, index):
        x, y, aux = original(self, index)
        return x, y, aux, int(index)

    dataset_cls.__getitem__ = indexed_getitem
    dataset_cls._attempt13_index_patched = True


def _cot_loss(method, pred_norm: torch.Tensor, future_aux_norm: torch.Tensor,
              indices: torch.Tensor, split: str, return_stats: bool = False):
    st = method.__dict__["_attempt13_cot_state"]
    ids = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    targets_np = np.asarray(st["targets"][split][ids], dtype=np.float32)
    masks_np = np.asarray(st["masks"][split][ids], dtype=np.bool_)
    target = torch.from_numpy(targets_np).to(device=pred_norm.device, dtype=torch.float32)
    mask = torch.from_numpy(masks_np).to(device=pred_norm.device, dtype=torch.bool)

    # Match the released SimVP->R inference chain: raw denormalized AGRI, the
    # paired future solar-geometry channels, then the R model's train-only z-score.
    agri = pred_norm.float() * st["agri_std"] + st["agri_mean"]
    patch = agri[..., PATCH_ROW:PATCH_ROW + PATCH_SIZE, PATCH_COL:PATCH_COL + PATCH_SIZE]
    geom = future_aux_norm[:, 8:24].float() * st["geom_std"] + st["geom_mean"]
    geom = geom[..., PATCH_ROW:PATCH_ROW + PATCH_SIZE, PATCH_COL:PATCH_COL + PATCH_SIZE]
    if patch.shape[1:] != (16, 13, 16, 16) or geom.shape[1:] != (16, 3, 16, 16):
        raise RuntimeError(f"Unexpected attempt13 COT input shapes {tuple(patch.shape)}, {tuple(geom.shape)}")
    r_input = torch.cat((patch, geom), dim=2)
    r_input = (r_input - st["r_mean"]) / st["r_std"]
    b = r_input.shape[0]
    cot_log1p_pred = st["r_model"](r_input.reshape(b * LEADS, 16, 16, 16)).float()
    cot_log1p_pred = cot_log1p_pred.reshape(b, LEADS, 1, 16, 16)
    cot_log1p_target = torch.log1p(target.clamp_min(0.0))
    # torch.where prevents invalid pixels from propagating NaNs through loss.
    diff = torch.where(mask, cot_log1p_pred - cot_log1p_target, torch.zeros_like(cot_log1p_pred))
    reduce_dims = (0, 2, 3, 4)
    lead_sse = diff.square().sum(dim=reduce_dims)
    lead_n = mask.sum(dim=reduce_dims)
    present = lead_n > 0
    if not bool(present.any()):
        zero = pred_norm.sum() * 0.0
        if return_stats:
            # Validation selection is in physical COT, while the auxiliary
            # training objective remains log1p-space MSE.
            cot_physical_pred = torch.expm1(cot_log1p_pred.clamp(0.0, math.log1p(100.0)))
            return zero, torch.zeros_like(lead_sse).detach(), lead_n.detach(), cot_physical_pred
        return zero
    lead_mse = lead_sse[present] / lead_n[present].to(lead_sse.dtype)
    loss = lead_mse.mean()
    if return_stats:
        cot_physical_pred = torch.expm1(cot_log1p_pred.clamp(0.0, math.log1p(100.0)))
        physical_diff = torch.where(mask, cot_physical_pred - target, torch.zeros_like(cot_physical_pred))
        physical_sse = physical_diff.square().sum(dim=reduce_dims)
        return loss, physical_sse.detach(), lead_n.detach(), cot_physical_pred
    return loss


def attach_state(method, *, r_model, s_norm_path: Path, r_norm_path: Path,
                 sidecar_dir: Path, mode: str, lambda_path: Path,
                 r_checkpoint: Path) -> dict[str, object]:
    if mode not in {"mse_control", "cot_aux"}:
        raise ValueError(f"Unknown attempt13 arm: {mode}")
    with np.load(s_norm_path, allow_pickle=False) as z:
        agri_mean = np.asarray(z["agri_mean"], np.float32)
        agri_std = np.asarray(z["agri_std"], np.float32)
        input_mean = np.asarray(z["input_mean"], np.float32)
        input_std = np.asarray(z["input_std"], np.float32)
    rnorm = json.loads(r_norm_path.read_text(encoding="utf-8"))
    r_mean = np.asarray(rnorm["mean"], np.float32)
    r_std = np.asarray(rnorm["std"], np.float32)
    if agri_mean.shape != (13,) or agri_std.shape != (13,) or input_mean.shape != (16,) or input_std.shape != (16,):
        raise ValueError("Unexpected frozen SimVP normalization shape")
    if r_mean.shape != (16,) or r_std.shape != (16,) or np.any(r_std <= 0):
        raise ValueError("Unexpected frozen R normalization")
    arrays = {}
    for split in ("train", "val"):
        arrays[split] = {
            "targets": np.load(sidecar_dir / f"{split}_target_physical.npy", mmap_mode="r", allow_pickle=False),
            "masks": np.load(sidecar_dir / f"{split}_mask.npy", mmap_mode="r", allow_pickle=False),
        }
        if arrays[split]["targets"].shape != arrays[split]["masks"].shape or arrays[split]["targets"].shape[1:] != (16, 1, 16, 16):
            raise ValueError(f"Invalid {split} CPP target sidecar shape")
    device = torch.device("cuda")
    r_model = r_model.to(device).eval()
    for param in r_model.parameters():
        param.requires_grad_(False)
        param.grad = None
    state = {
        "targets": {k: v["targets"] for k, v in arrays.items()},
        "masks": {k: v["masks"] for k, v in arrays.items()},
        "r_model": r_model,
        "agri_mean": torch.from_numpy(agri_mean).to(device).view(1, 1, 13, 1, 1),
        "agri_std": torch.from_numpy(agri_std).to(device).view(1, 1, 13, 1, 1),
        "geom_mean": torch.from_numpy(input_mean[13:]).to(device).view(1, 1, 3, 1, 1),
        "geom_std": torch.from_numpy(input_std[13:]).to(device).view(1, 1, 3, 1, 1),
        "r_mean": torch.from_numpy(r_mean).to(device).view(1, 16, 1, 1),
        "r_std": torch.from_numpy(r_std).to(device).view(1, 16, 1, 1),
        "mode": mode,
        "lambda_path": Path(lambda_path),
        "r_checkpoint": Path(r_checkpoint),
        "s_norm_path": Path(s_norm_path),
        "r_norm_path": Path(r_norm_path),
        "sidecar_dir": Path(sidecar_dir),
    }
    object.__setattr__(method, "_attempt13_cot_state", state)
    object.__setattr__(method, "_attempt13_val_cot_sse", np.zeros(LEADS, dtype=np.float64))
    object.__setattr__(method, "_attempt13_val_cot_n", np.zeros(LEADS, dtype=np.int64))
    object.__setattr__(method, "_attempt13_last_val_lead_rmse", None)
    return {
        "mode": mode,
        "initial_simvp_state_sha256": sha_state_dict(method.model),
        "r_checkpoint_sha256": hashlib.sha256(Path(r_checkpoint).read_bytes()).hexdigest(),
        "simvp_norm_sha256": hashlib.sha256(Path(s_norm_path).read_bytes()).hexdigest(),
        "r_norm_sha256": hashlib.sha256(Path(r_norm_path).read_bytes()).hexdigest(),
        "sidecar_manifest_sha256": hashlib.sha256((Path(sidecar_dir) / "manifest.json").read_bytes()).hexdigest(),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
    }


def _estimate_lambda(method, train_dataset, seed: int = 1301, n_windows: int = 32) -> dict[str, object]:
    state = method.__dict__["_attempt13_cot_state"]
    target_masks = state["masks"]["train"]
    # Stratify by BJT month, and use only windows with actual CPP-supported future pixels.
    month_groups: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(train_dataset.samples):
        aux_paths = sample["aux"]
        month_groups[_stamp(aux_paths[8])[:6]].append(i)
    eligible = {
        month: [i for i in ids if np.asarray(target_masks[i]).any()]
        for month, ids in sorted(month_groups.items())
    }
    eligible = {m: ids for m, ids in eligible.items() if ids}
    if not eligible:
        raise RuntimeError("No train windows with CPP COT support for train-only lambda calibration")
    rng = np.random.default_rng(seed)
    months = sorted(eligible)
    base, rem = divmod(n_windows, len(months))
    chosen: list[int] = []
    per_month = {}
    for j, month in enumerate(months):
        quota = base + int(j < rem)
        pool = eligible[month]
        take = min(quota, len(pool))
        pick = rng.choice(pool, size=take, replace=False).tolist()
        chosen.extend(pick)
        per_month[month] = {"eligible": len(pool), "selected": take}
    if len(chosen) < n_windows:
        chosen_set = set(chosen)
        rest = [i for ids in eligible.values() for i in ids if i not in chosen_set]
        if rest:
            chosen.extend(rng.choice(rest, size=min(n_windows - len(chosen), len(rest)), replace=False).tolist())
    if len(chosen) < 8:
        raise RuntimeError(f"Insufficient CPP-supported train windows for lambda calibration: {len(chosen)}")

    device = next(method.model.parameters()).device
    ratios, measurements = [], []
    was_training = method.training
    method.eval()
    for i in chosen:
        x, y, aux, index = train_dataset[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        aux = aux.unsqueeze(0).to(device)
        ids = torch.tensor([index], dtype=torch.long)
        pred = method(x, y, batch_aux=aux)
        loss_agri = F.mse_loss(pred, y)
        loss_cot = _cot_loss(method, pred, aux, ids, "train")
        if not (torch.isfinite(loss_agri) and torch.isfinite(loss_cot) and loss_cot.detach().item() > 0):
            continue
        g_agri = torch.autograd.grad(loss_agri, pred, retain_graph=True, allow_unused=False)[0]
        g_cot = torch.autograd.grad(loss_cot, pred, allow_unused=False)[0]
        n_agri = float(torch.linalg.vector_norm(g_agri.detach().float()).cpu())
        n_cot = float(torch.linalg.vector_norm(g_cot.detach().float()).cpu())
        if n_agri > 0 and n_cot > 0 and math.isfinite(n_agri / n_cot):
            ratio = n_agri / n_cot
            ratios.append(ratio)
            measurements.append({"sample_index": int(i), "grad_norm_agri": n_agri,
                                 "grad_norm_cot": n_cot, "ratio": ratio,
                                 "valid_cpp_pixels": int(np.asarray(target_masks[i]).sum())})
        del x, y, aux, pred, g_agri, g_cot, loss_agri, loss_cot
        torch.cuda.empty_cache()
    if was_training:
        method.train()
    if len(ratios) < 8:
        raise RuntimeError(f"Only {len(ratios)} usable gradient-ratio windows")
    lam = 0.1 * float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not math.isfinite(lam) or lam <= 0:
        raise RuntimeError(f"Invalid frozen train-only lambda {lam}")
    return {"state": "PASS_FROZEN_TRAIN_ONLY_COT_AUX_WEIGHT",
            "lambda": lam, "formula": "0.1 * median(||grad_pred L_AGRI|| / ||grad_pred L_COT||)",
            "n_windows_selected": len(chosen), "n_windows_used": len(ratios),
            "train_months": per_month, "seed": seed, "measurements": measurements,
            "split_used_for_calibration": "train only", "validation_used": False,
            "test_payloads_read": 0, "buffer_payloads_read": 0}


def patch_simvp_class(simvp_cls) -> None:
    if getattr(simvp_cls, "_attempt13_patched", False):
        return

    def training_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise RuntimeError(f"attempt13 expects indexed train batches, got {len(batch)} fields")
        batch_x, batch_y, batch_aux, indices = batch
        pred_y = self(batch_x, batch_y, batch_aux=batch_aux)
        loss_agri = self.criterion(pred_y, batch_y)
        st = self.__dict__["_attempt13_cot_state"]
        loss_cot = pred_y.new_zeros(())
        if st["mode"] == "cot_aux":
            loss_cot = _cot_loss(self, pred_y, batch_aux, indices, "train")
            loss = loss_agri + st["lambda"] * loss_cot
        else:
            loss = loss_agri
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_agri_mse", loss_agri, on_step=False, on_epoch=True, prog_bar=False)
        if st["mode"] == "cot_aux":
            self.log("train_cot_aux_mse", loss_cot, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_start(self):
        object.__setattr__(self, "_attempt13_val_cot_sse", np.zeros(LEADS, dtype=np.float64))
        object.__setattr__(self, "_attempt13_val_cot_n", np.zeros(LEADS, dtype=np.int64))

    def validation_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise RuntimeError(f"attempt13 expects indexed validation batches, got {len(batch)} fields")
        batch_x, batch_y, batch_aux, indices = batch
        pred_y = self(batch_x, batch_y, batch_aux=batch_aux)
        loss_agri = self.criterion(pred_y, batch_y)
        self.log("val_loss", loss_agri, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_agri_mse", loss_agri, on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            _, sse, count, _ = _cot_loss(self, pred_y.detach(), batch_aux, indices, "val", return_stats=True)
        self.__dict__["_attempt13_val_cot_sse"] += sse.cpu().numpy().astype(np.float64)
        self.__dict__["_attempt13_val_cot_n"] += count.cpu().numpy().astype(np.int64)
        return loss_agri

    def on_validation_epoch_end(self):
        sse = self.__dict__["_attempt13_val_cot_sse"]
        count = self.__dict__["_attempt13_val_cot_n"]
        valid = count > 0
        if not valid.any():
            raise RuntimeError("Validation has zero exact-key CPP support")
        lead_rmse = np.full(LEADS, np.nan, dtype=np.float64)
        lead_rmse[valid] = np.sqrt(sse[valid] / count[valid])
        score = float(np.mean(lead_rmse[valid], dtype=np.float64))
        object.__setattr__(self, "_attempt13_last_val_lead_rmse", lead_rmse.tolist())
        self.log("val_cot_rmse", torch.tensor(score, device=self.device, dtype=torch.float32),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

    simvp_cls.training_step = training_step
    simvp_cls.validation_step = validation_step
    simvp_cls.on_validation_epoch_start = on_validation_epoch_start
    simvp_cls.on_validation_epoch_end = on_validation_epoch_end
    simvp_cls._attempt13_patched = True


def initialize_lambda_or_load(method, train_dataset, lambda_path: Path) -> dict[str, object]:
    state = method.__dict__["_attempt13_cot_state"]
    if lambda_path.exists():
        receipt = json.loads(lambda_path.read_text(encoding="utf-8"))
        if receipt.get("state") != "PASS_FROZEN_TRAIN_ONLY_COT_AUX_WEIGHT" or receipt.get("split_used_for_calibration") != "train only":
            raise ValueError("Existing attempt13 lambda receipt failed provenance check")
        state["lambda"] = float(receipt["lambda"])
        return receipt
    receipt = _estimate_lambda(method, train_dataset)
    lambda_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = lambda_path.with_suffix(lambda_path.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    tmp.replace(lambda_path)
    state["lambda"] = float(receipt["lambda"])
    return receipt

