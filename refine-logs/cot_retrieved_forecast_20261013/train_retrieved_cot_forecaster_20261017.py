#!/usr/bin/env python3
"""Paired P-R / P-Hybrid training on the frozen train/val-only contract."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.retrieved_cot_forecaster_20261013 import RetrievedCOTForecaster, cot_supervision_masked_mse


class SequenceStationDataset(Dataset):
    def __init__(self, split: str, branch: Path, bank: Path, pack: Path):
        self.split = split
        self.ids = np.load(pack / f"sequence_indices_{split}.npy", mmap_mode="r", allow_pickle=False)
        self.keys = list(csv.DictReader((pack / "sequence_keys.csv").open(encoding="utf-8")))
        self.keys = [r for r in self.keys if r["split"] == split]
        if len(self.keys) != len(self.ids):
            raise ValueError(f"key/sequence row mismatch: {len(self.keys)} != {len(self.ids)}")
        self.bank, self.pack, self.branch = bank, pack, branch
        self._arrays = None
        self.future_support = None
        self.station_items = None

    def build_future_support(self, chunk_size: int = 256):
        """Find sequence/station samples with at least one valid future target pixel.

        This uses only the frozen frame-validity sidecar and train/val sequence
        indices. It does not inspect CPP values or choose samples by cloud type.
        """
        valid_path = self.branch / "data/frame_sidecars_20261017_v1/frame_valid_mask.npy"
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        support = np.zeros((len(self.ids), 2), dtype=bool)
        for start in range(0, len(self.ids), chunk_size):
            stop = min(start + chunk_size, len(self.ids))
            future_indices = np.asarray(self.ids[start:stop, 8:], dtype=np.int64)
            # [sequence, lead, station, channel, row, column] -> [sequence, station]
            block = np.asarray(valid[future_indices], dtype=bool)
            support[start:stop] = block.any(axis=(1, 3, 4, 5))
        self.future_support = support
        self.refresh_station_items()
        if len(self.station_items) == 0:
            raise ValueError(f"{self.split} has no future COT supervision")
        return support

    def refresh_station_items(self):
        if self.future_support is None:
            self.station_items = None
        else:
            self.station_items = np.argwhere(self.future_support).astype(np.int64)

    def _open(self):
        if self._arrays is None:
            self._arrays = (
                np.load(self.bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False),
                np.load(self.branch / "data/frame_sidecars_20261017_v1/frame_geometry.npy", mmap_mode="r", allow_pickle=False),
                np.load(self.branch / "data/frame_sidecars_20261017_v1/frame_valid_mask.npy", mmap_mode="r", allow_pickle=False),
                np.load(self.pack / "cpp_cot_physical.npy", mmap_mode="r", allow_pickle=False),
                np.load(self.pack / "cpp_common_mask.npy", mmap_mode="r", allow_pickle=False),
            )
        return self._arrays

    def __len__(self):
        return len(self.station_items) if self.station_items is not None else len(self.ids) * 2

    def __getitem__(self, item):
        cot, geo, valid, cpp, cpp_mask = self._open()
        if self.station_items is None:
            seq, station = divmod(int(item), 2)
        else:
            seq, station = map(int, self.station_items[item])
        ix = np.asarray(self.ids[seq], dtype=np.int64)
        history = np.asarray(cot[ix[:8], station], dtype=np.float32) / np.float32(np.log(101.0))
        hmask = np.asarray(valid[ix[:8], station], dtype=np.float32)
        geometry = np.asarray(geo[ix, station], dtype=np.float32)
        r_target = np.expm1(np.asarray(cot[ix[8:], station], dtype=np.float32)).clip(0.0)
        target_mask = np.asarray(valid[ix[8:], station], dtype=np.float32)
        cpp_target = np.asarray(cpp[ix[8:], station], dtype=np.float32)
        cpp_valid = np.asarray(cpp_mask[ix[8:], station], dtype=np.float32)
        # One immutable base label plus CPP values substituted only where
        # supervised; target_mask stays the R-valid forecast support.
        hybrid = np.where(cpp_valid > 0, cpp_target, r_target).astype(np.float32)
        key = self.keys[seq]
        return (torch.from_numpy(history.copy()), torch.from_numpy(hmask.copy()),
                torch.from_numpy(geometry.copy()), torch.from_numpy(r_target.copy()),
                torch.from_numpy(hybrid), torch.from_numpy(target_mask.copy()),
                torch.from_numpy(cpp_target.copy()), torch.from_numpy(cpp_valid.copy()),
                torch.tensor(station, dtype=torch.int64), key["init_time_utc"])


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def val_metrics(model, loader, device, prediction_path: Path):
    model.eval()
    station_sse = np.zeros(2, np.float64); station_n = np.zeros(2, np.int64)
    lead_sse = np.zeros((2, 16), np.float64); lead_n = np.zeros((2, 16), np.int64)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            x, hm, geo, r, hyb, tm, cpp, cm, sid, _ = batch
            x, hm, geo = x.to(device), hm.to(device), geo.to(device)
            pred = torch.expm1(model(x, hm, geo).float() * math.log(101.0)).clamp_min_(0)
            predictions.append(pred.cpu().numpy().astype(np.float16))
            truth = cpp.to(device); mask = ((cm * tm) > 0).to(device)
            for s in range(2):
                sm = mask[sid.to(device) == s]
                err = torch.where(sm, pred[sid.to(device) == s] - truth[sid.to(device) == s], 0.0)
                station_sse[s] += float(err.double().square().sum().cpu())
                station_n[s] += int(sm.sum().cpu())
                for lead in range(16):
                    ml = sm[:, lead]
                    el = err[:, lead]
                    lead_sse[s, lead] += float(el.double().square().sum().cpu())
                    lead_n[s, lead] += int(ml.sum().cpu())
    if np.any(lead_n == 0):
        raise ValueError(f"validation has unsupported station/lead cells: {np.where(lead_n == 0)}")
    station_rmse = np.sqrt(station_sse / station_n)
    lead_rmse = np.sqrt(lead_sse / lead_n)
    # Equal station and lead weighting; primary selection score.
    score = float(lead_rmse.mean())
    np.save(prediction_path, np.concatenate(predictions, axis=0), allow_pickle=False)
    return {"cpp_common_rmse_equal_station_lead": score,
            "station_rmse": station_rmse.tolist(), "lead_rmse": lead_rmse.tolist(),
            "valid_pixels_by_station": station_n.tolist()}


def cot_amplitude_auxiliary_mse(pred_log_scaled, target_cot, valid):
    """Physical-amplitude MSE on (COT_pred-COT_ref)/100, equal-weight by lead."""
    valid = valid.bool()
    safe_target = torch.where(valid, target_cot, torch.zeros_like(target_cot)).clamp_min(0.0)
    physical_pred = torch.expm1(pred_log_scaled.float() * math.log(101.0)).clamp_min(0.0)
    error = torch.where(valid, (physical_pred - safe_target) / 100.0,
                        torch.zeros_like(physical_pred))
    dims = (0, 2, 3, 4)
    counts = valid.sum(dim=dims)
    per_lead = error.square().sum(dim=dims) / counts.clamp_min(1)
    active = counts > 0
    if not bool(active.any()):
        raise ValueError("batch has no valid amplitude supervision")
    return per_lead[active].mean()


def run_arm(arm, init_state, train_loader, val_loader, device, args, outdir, provenance):
    seed_all(args.seed)
    model = RetrievedCOTForecaster(width=args.width).to(device)
    model.load_state_dict(init_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    history = []; best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total_loss = 0.0; total_base = 0.0; total_amp = 0.0; n_batches = 0
        for batch in train_loader:
            x, hm, geo, r, hyb, tm, _, _, _, _ = batch
            x, hm, geo, tm = x.to(device, non_blocking=True), hm.to(device, non_blocking=True), geo.to(device, non_blocking=True), tm.to(device, non_blocking=True)
            target = (r if arm == "P-R" else hyb).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, hm, geo)
            base_loss = cot_supervision_masked_mse(pred, target, tm)
            amp_aux = pred.new_zeros(())
            if arm == "P-Hybrid" and args.physical_amplitude_weight > 0:
                amp_aux = cot_amplitude_auxiliary_mse(pred, target, tm)
            loss = base_loss + args.physical_amplitude_weight * amp_aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {arm} loss at epoch {epoch+1}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += float(loss.detach()); total_base += float(base_loss.detach())
            total_amp += float(amp_aux.detach()); n_batches += 1
        metrics = val_metrics(model, val_loader, device,
                              outdir / f"{arm.replace('-', '_')}_val_pred_epoch{epoch+1:02d}.npy")
        record = {"epoch": epoch + 1, "train_loss": total_loss / n_batches,
                  "train_base_log_mse": total_base / n_batches,
                  "train_amplitude_aux_mse": total_amp / n_batches,
                  "physical_amplitude_weight": args.physical_amplitude_weight,
                  "lr": optimizer.param_groups[0]["lr"], **metrics}
        history.append(record)
        ckpt = {"model": model.state_dict(), "epoch": epoch + 1, "arm": arm,
                "seed": args.seed, "contract": provenance}
        torch.save(ckpt, outdir / f"{arm.replace('-', '_')}_epoch{epoch+1:02d}.pt")
        if metrics["cpp_common_rmse_equal_station_lead"] < best:
            best = metrics["cpp_common_rmse_equal_station_lead"]
            torch.save(ckpt, outdir / f"{arm.replace('-', '_')}_best.pt")
        print(json.dumps({"arm": arm, **record}, ensure_ascii=False), flush=True)
    (outdir / f"{arm.replace('-', '_')}_metrics.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branch", type=Path, required=True); p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--width", type=int, default=96)
    p.add_argument("--batch-size", type=int, required=True); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--max-train-sequences", type=int, default=2048)
    p.add_argument("--arms", nargs="+", choices=("P-R", "P-Hybrid"), default=("P-R", "P-Hybrid"))
    p.add_argument("--physical-amplitude-weight", type=float, default=0.0)
    args = p.parse_args()
    m = json.loads((args.bank / "bank_manifest.json").read_text())
    pm = json.loads((args.pack / "pack_manifest.json").read_text())
    fm = json.loads((args.branch / "data/frame_sidecars_20261017_v1/pack_manifest.json").read_text())
    audit_path = args.branch / "results/data_gate/retrieved_cot_full_bank_audit_20261017.json"
    if not audit_path.is_file():
        raise RuntimeError("full bank audit is not complete")
    audit = json.loads(audit_path.read_text())
    expected_split = "hunan_monthly_block_2d_buffer1d_seed42_v1"
    if (m.get("state") != "COMPLETE" or pm.get("state") != "PASS_TRAIN_VAL_SEQUENCE_CPP_PACK"
            or fm.get("state") != "COMPLETE"):
        raise RuntimeError("bank/sequence/frame pack is not complete")
    if any(x.get("test_payload_opened") is not False for x in (m, pm, fm, audit)):
        raise RuntimeError("test payload provenance gate failed")
    if any(x.get("split_id") != expected_split for x in (m, pm, fm, audit)):
        raise RuntimeError("unexpected split contract")
    if (audit.get("state") != "PASS_FULL_BANK_FINITE_AND_PIXEL_SUPPORT_AUDIT"
            or audit.get("frames_expected") != m.get("frames_total")
            or audit.get("frames_processed") != m.get("frames_total")
            or audit.get("failure_count") != 0
            or audit.get("candidate_sha256") != m.get("candidate_sha256")):
        raise RuntimeError("full-bank finite/support audit does not match COT bank")
    audit_frame_keys_sha = audit.get("source_sha256", {}).get("frame_keys.csv")
    if not audit_frame_keys_sha or fm.get("bank_frame_keys_sha256") != audit_frame_keys_sha:
        raise RuntimeError("frame-validity sidecar is not bound to audited bank keys")
    if pm.get("source_sha256", {}).get("frame_keys.csv") != audit_frame_keys_sha:
        raise RuntimeError("sequence/CPP pack is not bound to audited bank keys")
    if m.get("R_checkpoint_sha256") != audit.get("R_checkpoint_sha256"):
        raise RuntimeError("bank/audit teacher checkpoint mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = SequenceStationDataset("train", args.branch, args.bank, args.pack)
    val = SequenceStationDataset("val", args.branch, args.bank, args.pack)
    train.build_future_support()
    # Freeze a month-stratified time-hash subsample before looking at errors.
    month_groups = {}
    eligible_sequences = np.flatnonzero(train.future_support.any(axis=1))
    nseq = min(args.max_train_sequences, len(eligible_sequences))
    for i in eligible_sequences:
        row = train.keys[int(i)]
        month_groups.setdefault(row["init_time_utc"][:7], []).append(int(i))
    total = sum(map(len, month_groups.values()))
    quotas = {m: (nseq * len(ids)) // total for m, ids in month_groups.items()}
    left = nseq - sum(quotas.values())
    remainders = sorted(month_groups, key=lambda m: (
        -((nseq * len(month_groups[m])) % total), m))
    for month in remainders[:left]: quotas[month] += 1
    selected = []
    for month, ids in month_groups.items():
        ranked = sorted(ids, key=lambda i: hashlib.sha256(
            f"{args.seed}|{train.keys[i]['init_time_utc']}".encode()).hexdigest())
        selected.extend(ranked[:quotas[month]])
    selected_seq = np.asarray(sorted(selected), dtype=np.int64)
    train.ids = train.ids[selected_seq]
    train.keys = [train.keys[i] for i in selected_seq]
    train.future_support = train.future_support[selected_seq]
    train.refresh_station_items()
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=gen,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                            prefetch_factor=2 if args.workers else None)
    seed_all(args.seed)
    model = RetrievedCOTForecaster(width=args.width).to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    provenance = {"split_id": m["split_id"], "seed": args.seed, "train_sequences": nseq,
                  "eligible_train_sequences_before_subsample": int(len(eligible_sequences)),
                  "train_sequence_station_samples_with_future_support": int(len(train)),
                  "val_station_samples": len(val),
                  "batch_size": args.batch_size, "epochs": args.epochs,
                  "bank_manifest": m, "sequence_cpp_pack_manifest_sha256": sha256(args.pack/"pack_manifest.json"),
                  "frame_sidecar_manifest_sha256": sha256(args.branch/"data/frame_sidecars_20261017_v1/pack_manifest.json"),
                  "objective": {"base": "equal-lead masked MSE(log1p(COT)/log(101))",
                                "physical_amplitude_auxiliary_weight": args.physical_amplitude_weight,
                                "physical_amplitude_auxiliary": "MSE((COT_pred-COT_target)/100), equal-weight over active leads"},
                  "init_state_sha256": hashlib.sha256(b"".join(t.numpy().tobytes() for t in init_state.values())).hexdigest(),
                  "test_payload_opened": False}
    (args.output/"training_contract.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    with (args.output/"validation_prediction_keys.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["val_sample_index", "sequence_index", "station", "init_time_utc"])
        for i, row in enumerate(val.keys):
            for station in ("sili", "zhujia"):
                w.writerow([2*i + (station == "zhujia"), i, station, row["init_time_utc"]])
    # Same immutable initialization and same seeded minibatch order for both arms.
    for arm in args.arms:
        train_loader.generator.manual_seed(args.seed)
        run_arm(arm, init_state, train_loader, val_loader, device, args, args.output, provenance)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
