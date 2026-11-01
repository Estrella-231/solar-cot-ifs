"""Monthly-val SimVP/SHADECast comparison through one frozen COT R.

Only `val` payloads are opened. The R(real AGRI) target is a retrieval proxy,
not independent COT truth. See PLAN.md for exclusions and claim limits.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler


MANIFEST_SHA = "f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2"
NORM_SHA = "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d"
SIMVP_SHA = "f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4"
R_SHA = "74ddd7b2e78c29716260c9a790e0824659f2976d95b4598c3511c9816e77ad25"
SHADECAST_SHA = {
    "vae": "b5a940c943e51c8ccd37f2ab6331bcd21d268d8ed0b23d6538fb9ebec27069d1",
    "nowcaster": "76dd84a81d9d368e6b6fdf509fc731e501713dc33ee872c312407e3bcb3171ce",
    "diffusion": "3a4985d17070ade7c79aaac3b4326b846b4f8b920b3e4a4540f9d25b815a7a95",
}
METHODS = ("simvp", "shadecast_nowcaster", "shadecast_diffusion_mean_agri")
GROUPS = {"short_15_60": (0, 4), "long_120_240": (7, 16)}
COS80 = float(np.cos(np.deg2rad(80.0)))
THICK_LOG_COT = float(np.log1p(10.0))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def stamp(value):
    return re.sub(r"\D", "", str(value))[:12]


class StridedSampler(Sampler):
    def __init__(self, length, rank, world):
        self.ids = range(rank, length, world)
    def __iter__(self): return iter(self.ids)
    def __len__(self): return len(self.ids)


class FixedSampler(Sampler):
    def __init__(self, ids): self.ids = list(map(int, ids))
    def __iter__(self): return iter(self.ids)
    def __len__(self): return len(self.ids)


def load_contract(a, load_models=True):
    solar, shade = Path(a.solar_root), Path(a.shade_root)
    manifest = solar / "experiments/hunan_monthly_block_20261002/manifests_v1"
    simvp_run = solar / "experiments/hunan_monthly_block_20261002/runs/simvp_seed42_v1/training"
    simvp_source = solar / "experiments/hunan_monthly_block_20261002/code"
    rdir = solar / "experiments/regional_station_cot_20260925/r256_full_exploratory_seed42_batch512_node22"
    pair = solar / "experiments/regional_station_cot_20260925/agri_cpp_pair_bank_v1"
    shade_run = Path(a.shade_root) / "runs_node22_ddp4"

    if str(shade) not in sys.path: sys.path.insert(0, str(shade))
    import train as shade_train
    norm, info = shade_train.audit(manifest)
    assert info["manifest_sha256"] == MANIFEST_SHA and info["norm_sha256"] == NORM_SHA
    assert info["counts"] == {"train": 35320, "val": 3140, "test": 3202}
    norm["mean"] = np.asarray(norm["mean"], dtype=np.float32)
    norm["std"] = np.asarray(norm["std"], dtype=np.float32)

    sim_path, r_path = simvp_run / "best.pt", rdir / "best.pt"
    assert sha(sim_path) == SIMVP_SHA and sha(r_path) == R_SHA
    sim_meta = json.loads((simvp_run / "run_metadata.json").read_text())
    sim_done = json.loads((simvp_run / "training_complete.json").read_text())
    assert sim_meta["split_id"] == "hunan_monthly_block_2d_buffer1d_seed42_v1"
    assert sim_meta["manifest_sha256"] == MANIFEST_SHA and sim_meta["normalization_sha256"] == NORM_SHA
    assert sim_meta["initialization"] == "random" and sim_meta["test_payloads_read"] == 0
    assert sim_done["best_checkpoint_sha256"] == SIMVP_SHA and sim_done["test_payloads_read"] == 0
    assert sha(manifest / "hunan_train_norm.json") == NORM_SHA
    assert sha(simvp_source / "triad_models.py") == sim_meta["model_code_sha256"]

    # Require the already-completed full monthly SHADECast validation and all
    # three formal stage receipts before running any inference.
    from eval_val import verify_artifacts
    vae_done = json.loads((shade_run / "vae/complete.json").read_text())
    now_done = json.loads((shade_run / "nowcaster/complete.json").read_text())
    diff_done = json.loads((shade_run / "diffusion/complete.json").read_text())
    assert [vae_done["best_sha256"], now_done["best_sha256"], diff_done["best_sha256"]] == [
        SHADECAST_SHA["vae"], SHADECAST_SHA["nowcaster"], SHADECAST_SHA["diffusion"]]
    eval_done = json.loads((shade_run / "validation_agri_v2/complete.json").read_text())
    assert eval_done["state"] == "COMPLETE_VALIDATION_AGRI_EVALUATION"
    assert eval_done["validation_sequences"] == 3140 and eval_done["test_payloads_read"] == 0
    assert sha(shade_run / "validation_agri_v2/summary.json") == eval_done["summary_sha256"]
    args = argparse.Namespace(output=shade_run)
    dep_hashes = verify_artifacts(args, info)
    assert dep_hashes == SHADECAST_SHA

    r_meta = json.loads((rdir / "run_metadata.json").read_text())
    pair_index = json.loads((pair / "index.json").read_text())
    r_norm_path = pair / "norm.json"
    assert pair_index["state"] == "COMPLETE_EXPLORATORY_REAL_AGRI_CPP_CACHE"
    assert pair_index["test_used"] is False
    assert r_meta["purpose"] == "regional R feasibility and CPP-reference validation only"
    assert "unresolved" in r_meta["source_CPP_physical_QA"].lower()
    assert r_meta["normalization_sha256"] == sha(r_norm_path)
    assert r_meta["r_arch_sha256"] == sha(solar / "scripts/train_cot_repaired.py")
    assert r_meta["test_used"] is False and r_meta["train_frames"] == 6293 and r_meta["validation_frames"] == 3488

    seen = {"train": set(), "validation": set()}
    for shard in pair_index["shards"]:
        records = json.loads((pair / shard["directory"] / "records.json").read_text())
        for rec in records:
            if rec.get("included_for_R"):
                split = rec["split"]
                # The historical field is named `utc`, but the pair-bank
                # builder joins it directly to matched BJT filename stems.
                # Assert that identity rather than silently applying a TZ shift.
                key = stamp(rec["utc"])
                assert key == stamp(Path(rec["agri_relpath"]).stem)
                if split == "train": seen["train"].add(key)
                elif split in ("val", "validation"): seen["validation"].add(key)

    # New monthly val future timestamps that the old R saw in train or model
    # selection are excluded before scores. Manifest metadata only; no test IO.
    with (manifest / "hunan_8to16.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    val_target_keys = {stamp(Path(p).stem) for r in rows for p in r["data_relpaths"].split("|")[8:]}
    overlap_train = val_target_keys & seen["train"]
    overlap_val = val_target_keys & seen["validation"]
    safe_keys = val_target_keys - seen["train"] - seen["validation"]
    assert len(rows) == 3140 and len(val_target_keys) == 4025
    assert overlap_train.isdisjoint(overlap_val)
    assert len(safe_keys) > 3000

    source_hashes = {"simvp_code": sha(simvp_source / "triad_models.py"),
                     "r_arch": sha(solar / "scripts/train_cot_repaired.py"),
                     "compare_script": sha(Path(__file__)),
                     "profile_pbs": sha(Path(__file__).with_name("profile.pbs")),
                     "full_pbs": sha(Path(__file__).with_name("full.pbs"))}
    sim = r = rmean = rstd = shade_net = None
    if load_models:
        # Load models only after all source and support checks pass.
        sys.path.insert(0, str(simvp_source))
        from triad_models import SimVPBackbone
        sim_ck = torch.load(sim_path, map_location="cpu", weights_only=False)
        sim = SimVPBackbone(simvp_source).cuda().eval().requires_grad_(False)
        sim.load_state_dict(sim_ck["model"], strict=True)

        cot_source = solar / "scripts"
        if str(cot_source) not in sys.path: sys.path.insert(0, str(cot_source))
        from train_cot_repaired import COTUNet
        r_ck = torch.load(r_path, map_location="cpu", weights_only=False)
        r = COTUNet().cuda().eval().requires_grad_(False)
        r.load_state_dict(r_ck["model"], strict=True)
        rn = json.loads(r_norm_path.read_text())
        rmean = torch.tensor(rn["mean"], device="cuda", dtype=torch.float32)[None, :, None, None]
        rstd = torch.tensor(rn["std"], device="cuda", dtype=torch.float32)[None, :, None, None]

        build_args = argparse.Namespace(stage="diffusion", output=shade_run)
        shade_net = shade_train.build(build_args, info).cuda().eval()
        diff_ck = torch.load(shade_run / "diffusion/best.pt", map_location="cpu", weights_only=False)
        assert diff_ck["metadata"]["stage"] == "diffusion"
        shade_net.load_state_dict(diff_ck["model"], strict=True)
        shade_net.eval(); shade_net.autoencoder.eval(); shade_net.context_encoder.eval()
    checkpoint_hashes = dict(SHADECAST_SHA)
    checkpoint_hashes.update({"simvp": SIMVP_SHA, "R256": R_SHA})
    return dict(solar=solar, shade=shade, manifest=manifest, simvp_source=simvp_source,
                simvp_path=sim_path, r_path=r_path, rdir=rdir, pair=pair, r_norm_path=r_norm_path,
                shade_run=shade_run, norm=norm, info=info, rows=rows,
                r_train_seen=seen["train"], r_val_seen=seen["validation"], safe_keys=safe_keys,
                overlap_train=overlap_train, overlap_val=overlap_val,
                sim=sim, r=r, rmean=rmean, rstd=rstd, shade_train=shade_train, shade_net=shade_net,
                checkpoint_hashes=checkpoint_hashes,
                source_hashes=source_hashes)


class GeoMonthlyData:
    """Wrap official val-only MonthlyData and add raw geometry from its cache."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    @property
    def rows(self): return self.base.rows
    def __getitem__(self, i):
        item = self.base[i]
        paths = self.base.rows[i]["data_relpaths"].split("|")
        geo = []
        for rel in paths:
            with np.load(self.base.cache / (rel + ".npz"), allow_pickle=False) as z:
                geo.append(z["g"])
        item["geometry"] = np.ascontiguousarray(np.stack(geo).astype(np.float32))
        return item


def retrieve(model, fields, geometry, mean, std, device):
    """Return log1p(COT) for [T,C,H,W] AGRI fields; missing AGRI -> R train mean."""
    raw = np.concatenate((fields, geometry), axis=1)
    pred = []
    with torch.no_grad():
        for start in range(0, len(raw), 4):
            x = torch.as_tensor(raw[start:start+4], device=device, dtype=torch.float32)
            x = torch.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
            y = model(x).float()
            if not torch.isfinite(y).all(): raise FloatingPointError("nonfinite frozen-R output")
            pred.append(y[:, 0].cpu().numpy())
    return np.concatenate(pred, axis=0).astype(np.float32)


def sum_metrics(pred, truth, valid, target_safe, cotmask, cot_ref, cot_pred):
    # [method, channel, lead, y, x]; all methods share exact targets and masks.
    e = pred.astype(np.float64) - truth[None].astype(np.float64)
    nonleak = valid & target_safe[None, :, None, None]
    full = nonleak[None].astype(np.float64)
    safe = (nonleak & cotmask[None]).astype(np.float64)[None]
    std = np.asarray(sum_metrics.channel_std, dtype=np.float64)[None, :, None, None, None]
    out = {
        "agri_all_abs_z": (np.abs(e) * full).sum((-1, -2)),
        "agri_all_sq_z": (e * e * full).sum((-1, -2)),
        "agri_all_abs_phys": (np.abs(e) * full * std).sum((-1, -2)),
        "agri_all_sq_phys": (e * e * full * std * std).sum((-1, -2)),
        "agri_all_count": nonleak.sum((-1, -2)).astype(np.float64),
        "agri_safe_abs_z": (np.abs(e) * safe).sum((-1, -2)),
        "agri_safe_sq_z": (e * e * safe).sum((-1, -2)),
        "agri_safe_abs_phys": (np.abs(e) * safe * std).sum((-1, -2)),
        "agri_safe_sq_phys": (e * e * safe * std * std).sum((-1, -2)),
        "agri_safe_count": np.broadcast_to(safe.sum((-1, -2))[0][None], (len(METHODS), 13, 16)).astype(np.float64),
    }
    log_ref = cot_ref.astype(np.float64)
    log_pred = cot_pred.astype(np.float64)
    mask = cotmask[None].astype(np.float64)
    dlog = log_pred - log_ref[None]
    with np.errstate(over="raise", invalid="raise"):
        ref_phys = np.expm1(log_ref)
        pred_phys = np.expm1(log_pred)
    if not np.isfinite(ref_phys).all() or not np.isfinite(pred_phys).all():
        raise FloatingPointError("nonfinite physical COT retrieved from R")
    dphys = pred_phys - ref_phys[None]
    out.update(cot_log_abs=(np.abs(dlog) * mask).sum((-1, -2)),
               cot_log_sq=(dlog * dlog * mask).sum((-1, -2)),
               cot_phys_abs=(np.abs(dphys) * mask).sum((-1, -2)),
               cot_phys_sq=(dphys * dphys * mask).sum((-1, -2)),
               cot_count=np.broadcast_to(cotmask.sum((-1, -2))[None, :], (len(METHODS), 16)).astype(np.float64))
    ref_thick = log_ref >= THICK_LOG_COT
    pred_thick = log_pred >= THICK_LOG_COT
    out.update(thick_miss=((mask.astype(bool) & ref_thick[None] & ~pred_thick).sum((-1, -2)).astype(np.float64)),
               thick_ref=np.broadcast_to((mask.astype(bool) & ref_thick[None]).sum((-1, -2)),
                                         (len(METHODS), 16)).astype(np.float64),
               thick_false_alarm=((mask.astype(bool) & ~ref_thick[None] & pred_thick).sum((-1, -2)).astype(np.float64)),
               thick_nonref=np.broadcast_to((mask.astype(bool) & ~ref_thick[None]).sum((-1, -2)),
                                            (len(METHODS), 16)).astype(np.float64))
    return out


STAT_KEYS = ("agri_all_abs_z", "agri_all_sq_z", "agri_all_abs_phys", "agri_all_sq_phys", "agri_all_count",
             "agri_safe_abs_z", "agri_safe_sq_z", "agri_safe_abs_phys", "agri_safe_sq_phys", "agri_safe_count",
             "cot_log_abs", "cot_log_sq", "cot_phys_abs", "cot_phys_sq", "cot_count",
             "thick_miss", "thick_ref", "thick_false_alarm", "thick_nonref")


def save_chunk(folder, rank, chunk_id, ids, blocks, stats):
    path = folder / f"rank_{rank:02d}_chunk_{chunk_id:04d}.npz"
    tmp = path.with_suffix(".npz.tmp")
    payload = {"index": np.asarray(ids, np.int32), "month_block": np.asarray(blocks, np.int32)}
    payload.update({k: np.stack([x[k] for x in stats]) for k in STAT_KEYS})
    with tmp.open("wb") as f: np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def run_rank(a, contract, device, rank, world):
    from monthly_data import MonthlyData
    from models import PLMSSampler
    norm, info = contract["norm"], contract["info"]
    base = MonthlyData(a.data_root, contract["manifest"], a.cache, norm, "val", "diffusion")
    ds = GeoMonthlyData(base)
    if a.mode == "profile":
        n = min(a.profile_sequences, len(ds))
        ids = np.unique(np.linspace(0, len(ds)-1, n, dtype=np.int64)).tolist() if rank == 0 else []
        sampler = FixedSampler(ids)
    else:
        ids = None
        sampler = StridedSampler(len(ds), rank, world)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=a.workers,
                        pin_memory=True, persistent_workers=False, prefetch_factor=1)

    sim, r, shade = contract["sim"], contract["r"], contract["shade_net"]
    plms = PLMSSampler(shade)
    month_mean = torch.tensor(norm["mean"], device=device, dtype=torch.float32)[None, :, None, None]
    month_std = torch.tensor(norm["std"], device=device, dtype=torch.float32)[None, :, None, None]
    sum_metrics.channel_std = norm["std"]
    new_blocks = {"simvp": 0}
    pending_ids, pending_months, pending = [], [], []
    chunk_id = 0
    if a.mode == "full":
        prior = sorted(a.output.glob(f"rank_{rank:02d}_chunk_*.npz"))
        done = set()
        for p in prior:
            with np.load(p, allow_pickle=False) as z: done.update(z["index"].tolist())
        chunk_id = len(prior)
    else: done = set()
    started = time.monotonic(); processed = 0
    with torch.no_grad():
        for cpu in loader:
            index = int(cpu["index"].item())
            if index in done: continue
            image = cpu["image"].to(device, non_blocking=True)
            valid = cpu["valid"][0].numpy().astype(bool)
            geometry = cpu["geometry"][0].numpy().astype(np.float32)
            row = ds.rows[index]
            paths = row["data_relpaths"].split("|")
            target_safe = np.asarray([stamp(Path(p).stem) in contract["safe_keys"] for p in paths[8:]], dtype=bool)
            target_nonleak = np.asarray([stamp(Path(p).stem) not in
                                         (contract["r_train_seen"] | contract["r_val_seen"])
                                         for p in paths[8:]], dtype=bool)
            geom_day = (geometry[8:, 0] > COS80) & (geometry[8:, 2] > 0.5)
            cotmask = valid[:, 8:].all(0) & geom_day & target_safe[:, None, None]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                sim_pred = sim(image[:, :, :8].permute(0, 2, 1, 3, 4),
                               cpu["geometry"].to(device, non_blocking=True)).permute(0, 2, 1, 3, 4)
                latent_now = shade.context_encoder.nowcast_net
                now_lat = latent_now.out_proj(latent_now.latent_forward(image[:, :, :8]))
                now_pred = shade.autoencoder.decode(now_lat).float()
                context = shade.context_encoder(image[:, :, :8])
            assert sim_pred.shape == now_pred.shape == (1, 13, 16, 256, 256)
            assert torch.isfinite(sim_pred).all() and torch.isfinite(now_pred).all()

            mean_agri = None
            with shade.ema_scope():
                for start in range(0, a.members, a.member_batch):
                    count = min(a.member_batch, a.members - start)
                    cond = {k: v.expand(count, *v.shape[1:]).contiguous() for k, v in context.items()}
                    seed = (104729 + 1000003 * index + 9176 * start) % (2**31 - 1)
                    with torch.random.fork_rng(devices=[device.index]):
                        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            latent, _ = plms.sample(S=20, conditioning=cond, batch_size=count,
                                shape=(32, 4, 64, 64), eta=0.0, verbose=False, progbar=False)
                            forecast = shade.autoencoder.decode(latent).float()
                    if not torch.isfinite(forecast).all(): raise FloatingPointError("nonfinite diffusion AGRI")
                    chunk_sum = forecast.sum(0)
                    mean_agri = chunk_sum if mean_agri is None else mean_agri + chunk_sum
            mean_agri = (mean_agri / a.members).unsqueeze(0)

            # All models and metrics use [method, channel, lead, y, x].
            pred_z = torch.cat((sim_pred, now_pred, mean_agri), dim=0).float().cpu().numpy()
            truth_z = image[0, :, 8:].float().cpu().numpy()
            truth_phys = truth_z * norm["std"][:, None, None, None] + norm["mean"][:, None, None, None]
            pred_phys = pred_z * norm["std"][None, :, None, None, None] + norm["mean"][None, :, None, None, None]
            valid_future = valid[:, 8:]
            truth_for_r = np.where(valid_future, truth_phys, np.nan)
            pred_for_r = np.where(valid_future[None], pred_phys, np.nan)
            geom_future = geometry[8:]
            cot_ref = retrieve(r, truth_for_r.transpose(1, 0, 2, 3), geom_future, contract["rmean"], contract["rstd"], device)
            cot_pred = np.stack([retrieve(r, x.transpose(1, 0, 2, 3), geom_future,
                                           contract["rmean"], contract["rstd"], device) for x in pred_for_r])
            assert np.isfinite(cot_ref).all() and np.isfinite(cot_pred).all()
            metrics = sum_metrics(pred_z, truth_z, valid_future, target_nonleak, cotmask, cot_ref, cot_pred)
            month_block = int(row["BJT_start"][:7].replace("-", ""))
            pending_ids.append(index); pending_months.append(month_block); pending.append(metrics)
            processed += 1
            if len(pending_ids) >= a.chunk_size:
                save_chunk(a.output, rank, chunk_id, pending_ids, pending_months, pending)
                chunk_id += 1; pending_ids, pending_months, pending = [], [], []
            if processed % 10 == 0:
                torch.cuda.synchronize(device)
                print(json.dumps({"event": "COT_COMPARE_PROGRESS", "mode": a.mode, "rank": rank,
                    "index": index, "processed_rank": processed, "elapsed_s": round(time.monotonic()-started,1),
                    "safe_lead_frames": int(cotmask.reshape(16,-1).any(1).sum()),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device)/2**30,2)}), flush=True)
    if pending_ids:
        save_chunk(a.output, rank, chunk_id, pending_ids, pending_months, pending)
    torch.cuda.synchronize(device)
    print(json.dumps({"event": "RANK_COMPLETE", "mode": a.mode, "rank": rank,
                      "processed": processed, "seconds": round(time.monotonic()-started,2),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device)/2**30,2)}), flush=True)


def merge_chunks(folder, world, expected):
    payloads = []
    for rank in range(world):
        chunks = sorted(Path(folder).glob(f"rank_{rank:02d}_chunk_*.npz"))
        assert chunks, f"missing rank {rank} output chunks"
        for p in chunks:
            with np.load(p, allow_pickle=False) as z:
                payloads.append({k: z[k].copy() for k in z.files})
    keys = ("index", "month_block", *STAT_KEYS)
    merged = {k: np.concatenate([x[k] for x in payloads], axis=0) for k in keys}
    order = np.argsort(merged["index"])
    merged = {k: v[order] for k, v in merged.items()}
    assert len(merged["index"]) == expected and np.array_equal(merged["index"], np.arange(expected))
    np.savez_compressed(Path(folder) / "per_sample_totals.npz", **merged)
    return merged


def grouped_sum(a, keys, method, lo, hi, safe=False):
    prefix = "agri_safe" if safe else "agri_all"
    idx = method
    sl = slice(lo, hi)
    count = a[f"{prefix}_count"]
    # Full-domain counts are [sample, channel, lead], while R-safe counts
    # retain the method axis because that support is broadcast per method.
    # Select the requested method before reducing, so count has the same
    # [sample, channel, lead] layout in both branches.
    if safe:
        count = count[:, idx]
    return {
        "mae_z": float(a[f"{prefix}_abs_z"][:, idx, :, sl].sum() / count[:, :, sl].sum()),
        "rmse_z": float(np.sqrt(a[f"{prefix}_sq_z"][:, idx, :, sl].sum() / count[:, :, sl].sum())),
        "mae_physical_by_channel": (a[f"{prefix}_abs_phys"][:, idx, :, sl].sum((0, 2)) /
                                     count[:, :, sl].sum((0, 2))).tolist(),
        "rmse_physical_by_channel": np.sqrt(a[f"{prefix}_sq_phys"][:, idx, :, sl].sum((0, 2)) /
                                             count[:, :, sl].sum((0, 2))).tolist(),
        "pixel_channel_count": float(count[:, :, sl].sum()),
    }


def bootstrap_diff(blocks, method, kind, lo, hi, nboot=10000):
    codes = np.unique(blocks["month_block"])
    rng = np.random.default_rng(20261008)
    picks = rng.integers(0, len(codes), size=(nboot, len(codes)))
    totals = []
    for code in codes:
        mask = blocks["month_block"] == code
        if kind in ("agri_all", "agri_safe"):
            prefix = "agri_all" if kind == "agri_all" else "agri_safe"
            abs_by_method = blocks[prefix+"_abs_z"][mask, :, :, lo:hi].sum((0, 2, 3))
            sq_by_method = blocks[prefix+"_sq_z"][mask, :, :, lo:hi].sum((0, 2, 3))
            counts = blocks[prefix+"_count"][mask]
            if kind == "agri_safe":
                # Safe support has [sample, method, channel, lead]; count once
                # from method 0 (the common mask is shared by all methods).
                count = counts[:, 0, :, lo:hi].sum()
            else:
                count = counts[:, :, lo:hi].sum()
            count_by_method = np.full(len(METHODS), count, dtype=np.float64)
            vals = np.stack((abs_by_method, sq_by_method, count_by_method), axis=-1)
        elif kind == "cot_log":
            abs_by_method = blocks["cot_log_abs"][mask, :, lo:hi].sum((0, 2))
            sq_by_method = blocks["cot_log_sq"][mask, :, lo:hi].sum((0, 2))
            count_by_method = blocks["cot_count"][mask, :, lo:hi].sum((0, 2))
            vals = np.stack((abs_by_method, sq_by_method, count_by_method), axis=-1)
        elif kind == "thick_miss":
            miss_by_method = blocks["thick_miss"][mask, :, lo:hi].sum((0, 2))
            ref_by_method = blocks["thick_ref"][mask, :, lo:hi].sum((0, 2))
            vals = np.stack((miss_by_method, ref_by_method), axis=-1)
        else: raise ValueError(kind)
        totals.append(vals)
    sums = np.stack(totals, axis=0).astype(np.float64)  # block, method, metric component
    boot = sums[picks].sum(1)  # bootstrap, method, component
    if kind == "thick_miss":
        score = boot[:, :, 0] / np.maximum(boot[:, :, 1], 1)
    else:
        numer, denom = boot[:, :, 1], boot[:, :, 2]
        score = np.sqrt(numer / np.maximum(denom, 1))
    delta = score[:, method] - score[:, 0]
    if kind == "thick_miss":
        point_scores = sums[:, :, 0].sum(0) / np.maximum(sums[:, :, 1].sum(0), 1)
    else:
        point_scores = np.sqrt(sums[:, :, 1].sum(0) / np.maximum(sums[:, :, 2].sum(0), 1))
    return {"candidate_minus_simvp": float(point_scores[method]-point_scores[0]),
            "delta_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
            "bootstrap_draws": nboot, "clusters": int(len(codes)), "cluster_unit": "monthly validation block"}


def aggregate(folder, merged, contract, a):
    out = {}
    for mi, name in enumerate(METHODS):
        out[name] = {"agri_full_spatial_nonleak_validation": {}, "agri_r_safe_daylight": {}, "cot_proxy_vs_R_real_agri": {}, "retrieved_thick_proxy": {}}
        for group, (lo, hi) in GROUPS.items():
            out[name]["agri_full_spatial_nonleak_validation"][group] = grouped_sum(merged, (), mi, lo, hi, safe=False)
            out[name]["agri_r_safe_daylight"][group] = grouped_sum(merged, (), mi, lo, hi, safe=True)
            c = merged["cot_count"][:, mi, lo:hi].sum()
            out[name]["cot_proxy_vs_R_real_agri"][group] = {
                "log1p_cot_mae": float(merged["cot_log_abs"][:,mi,lo:hi].sum()/c),
                "log1p_cot_rmse": float(np.sqrt(merged["cot_log_sq"][:,mi,lo:hi].sum()/c)),
                "physical_cot_mae": float(merged["cot_phys_abs"][:,mi,lo:hi].sum()/c),
                "physical_cot_rmse": float(np.sqrt(merged["cot_phys_sq"][:,mi,lo:hi].sum()/c)),
                "common_pixels": float(c),
            }
            ref = merged["thick_ref"][:,mi,lo:hi].sum()
            nonref = merged["thick_nonref"][:,mi,lo:hi].sum()
            out[name]["retrieved_thick_proxy"][group] = {
                "reference_definition": "R(real future AGRI) >= 10 COT proxy",
                "miss_rate": float(merged["thick_miss"][:,mi,lo:hi].sum()/ref) if ref else None,
                "false_alarm_rate": float(merged["thick_false_alarm"][:,mi,lo:hi].sum()/nonref) if nonref else None,
                "thick_reference_pixels": float(ref), "non_thick_reference_pixels": float(nonref),
            }
    bootstrap = {}
    for group, (lo, hi) in GROUPS.items():
        bootstrap[group] = {}
        for m in (1, 2):
            name = METHODS[m]
            bootstrap[group][name] = {
                "agri_full_rmse_z": bootstrap_diff(merged, m, "agri_all", lo, hi),
                "agri_r_safe_rmse_z": bootstrap_diff(merged, m, "agri_safe", lo, hi),
                "cot_proxy_log1p_rmse": bootstrap_diff(merged, m, "cot_log", lo, hi),
                "retrieved_thick_proxy_miss_rate": bootstrap_diff(merged, m, "thick_miss", lo, hi),
            }
            bootstrap[group][name]["interpretation"] = "Paired candidate-minus-SimVP interval over the same sampled monthly blocks; not an independent-truth CI."
    summary = {
        "state": "COMPLETE_EXPLORATORY_R_PROXY_VALIDATION_COMPARISON",
        "experiment": "cot_simvp_shadecast_val_20261008",
        "split": "hunan_monthly_block_2d_buffer1d_seed42_v1/val_only",
        "validation_sequences": int(len(merged["index"])),
        "monthly_validation_blocks": int(len(np.unique(merged["month_block"]))),
        "manifest_sha256": MANIFEST_SHA, "normalization_sha256": NORM_SHA,
        "checkpoint_sha256": contract["checkpoint_hashes"],
        "source_sha256": contract["source_hashes"],
        "r_provenance": {"state": "OLDER_EXPLORATORY_CPP_PAIRED_R_NOT_MONTHLY_R",
            "purpose": "regional R feasibility and CPP-reference validation only",
            "source_cpp_physical_qa": "unresolved; CPP is retrieval reference, not independent truth",
            "normalization_sha256": sha(contract["r_norm_path"]),
            "excluded_new_val_unique_target_frames_seen_in_r_train": len(contract["overlap_train"]),
            "excluded_new_val_unique_target_frames_seen_in_r_checkpoint_validation": len(contract["overlap_val"]),
            "eligible_new_val_unique_target_frames": len(contract["safe_keys"]),
            "total_new_val_unique_target_frames": 4025},
        "methods": list(METHODS), "lead_minutes": list(range(15,241,15)),
        "cot_reference": "same frozen R applied to real future AGRI; retrieval proxy, not physical truth",
        "diffusion_output": "R applied to the 10-member AGRI ensemble-mean map; official PLMS, 20 steps, eta=0, EMA",
        "metrics": out, "monthly_block_bootstrap": bootstrap,
        "support": "All paired AGRI metrics exclude every target timestamp seen by old R training or checkpoint-selection validation; spatial support is the identical per-channel future QA mask. R-safe AGRI metrics and COT further require all-13-channel validity, day_mask, cosSOZ>cos80, and the same R-disjoint timestamps.",
        "cpp_used": False, "test_payloads_read": 0,
        "limitations": ["exploratory validation only; no new monthly R exists",
            "COT outputs are all from AGRI through an older exploratory R and are not independent COT observations",
            "CPP quality chain is unresolved and CPP files were not read",
            "R-train and R-selection-validation timestamp overlaps are excluded from every paired method",
            "monthly-block bootstrap reflects validation block variability, not R or CPP uncertainty",
            "no model weights were trained or selected in this run"]}
    write_json(Path(folder)/"summary.json", summary)
    write_json(Path(folder)/"complete.json", {"state": summary["state"], "validation_sequences": summary["validation_sequences"],
        "manifest_sha256": MANIFEST_SHA, "normalization_sha256": NORM_SHA,
        "checkpoint_sha256": summary["checkpoint_sha256"], "test_payloads_read": 0,
        "per_sample_totals_sha256": sha(Path(folder)/"per_sample_totals.npz"),
        "summary_sha256": sha(Path(folder)/"summary.json")})
    print(json.dumps({"event":"COT_COMPARE_COMPLETE", "n":summary["validation_sequences"], "summary":str(Path(folder)/"summary.json")}),flush=True)


def main():
    p = argparse.ArgumentParser()
    for n in ("solar-root", "shade-root", "data-root", "cache", "output"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--mode", choices=("audit", "profile", "full", "aggregate"), required=True)
    p.add_argument("--profile-sequences", type=int, default=32)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--member-batch", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=50)
    a = p.parse_args()
    assert a.steps == 20 and a.members == 10 and 1 <= a.member_batch <= a.members
    assert a.mode == "full" or int(os.environ.get("WORLD_SIZE", "1")) == 1
    a.output = Path(a.output); a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == "audit":
        contract = load_contract(a, load_models=False)
        audit = {"state":"PASS_METADATA_AND_PROVENANCE_PREFLIGHT_ONLY",
                 "manifest_sha256":MANIFEST_SHA,"normalization_sha256":NORM_SHA,
                 "validation_sequences":len(contract["rows"]),"val_unique_target_frames":4025,
                 "excluded_r_train_overlap_frames":len(contract["overlap_train"]),
                 "excluded_r_selection_val_overlap_frames":len(contract["overlap_val"]),
                 "eligible_r_safe_frames":len(contract["safe_keys"]),
                 "checkpoint_sha256":contract["checkpoint_hashes"],
                 "source_sha256":contract["source_hashes"],"cpp_used":False,"test_payloads_read":0,
                 "note":"Metadata/checkpoint provenance only; no payload inference or quality result."}
        write_json(a.output/"preflight_audit.json",audit)
        print(json.dumps(audit),flush=True)
        return
    if a.mode == "aggregate":
        # Recover after a completed inference whose post-processing failed.
        # This path deliberately loads only metadata/checkpoint provenance and
        # saved chunks; it does not initialize CUDA, models, or data payloads.
        contract = load_contract(a, load_models=False)
        run_contract = json.loads((a.output/"run_contract.json").read_text())
        assert run_contract["mode"] == "full" and run_contract["split"] == "val_only"
        assert run_contract["manifest_sha256"] == MANIFEST_SHA
        assert run_contract["norm_sha256"] == NORM_SHA
        assert run_contract["checkpoints"] == contract["checkpoint_hashes"]
        assert run_contract["cpp_used"] is False and run_contract["test_payloads_read"] == 0
        merged = merge_chunks(a.output, 4, 3140)
        aggregate(a.output, merged, contract, a)
        return
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert torch.cuda.is_available() and torch.cuda.device_count() == world
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    if world > 1: dist.init_process_group("nccl")
    contract = load_contract(a)
    if rank == 0:
        if (a.output/"complete.json").exists(): raise RuntimeError("completed output is immutable")
        run_contract = {"mode":a.mode,"split":"val_only","manifest_sha256":MANIFEST_SHA,"norm_sha256":NORM_SHA,
                        "checkpoints":contract["checkpoint_hashes"],"members":a.members,"steps":a.steps,
                        "sampler":"official PLMSSampler, eta=0, EMA denoiser","test_payloads_read":0,
                        "cpp_used":False,"source_sha256":contract["source_hashes"]["compare_script"],
                        "launch_script_sha256":contract["source_hashes"]["full_pbs" if a.mode == "full" else "profile_pbs"]}
        if (a.output/"run_contract.json").exists():
            assert json.loads((a.output/"run_contract.json").read_text()) == run_contract
        else: write_json(a.output/"run_contract.json",run_contract)
    if world > 1: dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    run_rank(a, contract, device, rank, world)
    elapsed = time.monotonic()-start
    peak = torch.cuda.max_memory_reserved(device)/2**30
    records = [None for _ in range(world)]
    records[rank] = {"elapsed_seconds":elapsed,"peak_reserved_gib":peak,"gpu":torch.cuda.get_device_name(device)}
    if world > 1: dist.all_gather_object(records, records[rank]);dist.barrier()
    if rank == 0:
        if a.mode == "profile":
            profile={"state":"COMPLETE_ONE_GPU_PROFILE_ONLY","profile_sequences":len(np.unique(np.linspace(0,3139,min(a.profile_sequences,3140),dtype=np.int64))),
                     "world_size":world,"elapsed_seconds":elapsed,"samples_per_second":len(np.unique(np.linspace(0,3139,min(a.profile_sequences,3140),dtype=np.int64)))/elapsed,
                     "peak_reserved_gib":peak,"gpu":records[0]["gpu"],"finite_outputs":True,
                     "manifest_sha256":MANIFEST_SHA,"normalization_sha256":NORM_SHA,
                     "checkpoint_sha256":contract["checkpoint_hashes"],"r_overlap_train_excluded":len(contract["overlap_train"]),
                     "r_overlap_selection_val_excluded":len(contract["overlap_val"]),"cpp_used":False,"test_payloads_read":0,
                     "note":"Engineering throughput/memory profile only; not a model-quality result."}
            write_json(a.output/"profile.json",profile);print(json.dumps(profile),flush=True)
        else:
            merged=merge_chunks(a.output,world,3140)
            aggregate(a.output,merged,contract,a)
    if world > 1: dist.barrier();dist.destroy_process_group()


if __name__ == "__main__": main()
