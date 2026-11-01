"""Frozen-R, no-training 64/128/256 upstream-source comparison.

For each sequence, retrieve one full-resolution COT field from the last real
historical AGRI image using the existing exploratory R256 checkpoint, estimate
motion only from the last two historical AGRI frames, then sample that field at
the backward-traced station 5x5 source coordinates. Future CPP is reference only.
The R input/context and motion are identical across domain sizes; only the
available source-window mask changes. Test data are never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates

STATION = (163, 148)
SIZES = (64, 128, 256)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def motion_shift(pair):
    # pair contains two raw C13 frames; scaling/normalization cancels in phase correlation.
    a, b = pair[0][0][10], pair[1][0][10]
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    a = np.nan_to_num(a - np.nanmean(a)) * win
    b = np.nan_to_num(b - np.nanmean(b)) * win
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-12)).real
    d = np.asarray(np.unravel_index(corr.argmax(), corr.shape), dtype=np.float64)
    d[d > a.shape[0] / 2] -= a.shape[0]
    return d


def bucket():
    return {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0,
            "thick_n": 0, "thick_sq": 0.0, "thick_miss": 0}


def update(b, pred, truth, valid):
    if not valid.any():
        return
    e = pred[valid] - truth[valid]
    thick = truth[valid] >= 30
    b["n"] += int(e.size)
    b["sum_abs"] += float(np.abs(e).sum())
    b["sum_sq"] += float(np.square(e).sum())
    b["sum_err"] += float(e.sum())
    b["thick_n"] += int(thick.sum())
    b["thick_sq"] += float(np.square(e[thick]).sum())
    b["thick_miss"] += int((pred[valid][thick] < 10).sum())


def finish(b):
    n, tn = b["n"], b["thick_n"]
    return {
        "n_valid_station_pixels": n,
        "mae_cot": b["sum_abs"] / n if n else None,
        "rmse_cot": (b["sum_sq"] / n) ** .5 if n else None,
        "bias_cot": b["sum_err"] / n if n else None,
        "thick_n_cot_ge30": tn,
        "thick_rmse_cot": (b["thick_sq"] / tn) ** .5 if tn else None,
        "thick_miss_rate_pred_lt10": b["thick_miss"] / tn if tn else None,
    }


def main():
    p = argparse.ArgumentParser()
    for arg in ("solar-root", "hunan-source", "data-root", "r-run", "r-cache", "ghi-bank", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refuse to overwrite: {a.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("This frozen-R inference requires the allocated GPU.")

    solar = a.solar_root
    cfg = json.loads((solar / "configs/s_frozen_hunan_seed42.json").read_text())
    r_code = solar / "scripts/train_cot_repaired.py"
    rnorm_path = a.r_cache / "norm.json"
    r_status = json.loads((a.r_run / "status.json").read_text())
    rstate = torch.load(a.r_run / "best.pt", map_location="cpu", weights_only=False)
    assert r_status["state"] == "COMPLETE_EXPLORATORY_REGIONAL_R" and r_status["test_used"] is False
    assert rstate["metadata"]["r_arch_sha256"] == sha(r_code)
    assert rstate["metadata"]["normalization_sha256"] == sha(rnorm_path)
    norm = json.loads(rnorm_path.read_text())

    model_lib = load_module(r_code, "regional_source_R")
    model = model_lib.COTUNet().cuda().eval().requires_grad_(False)
    model.load_state_dict(rstate["model"])

    bank_code = solar / "experiments/cot_dynamics_20260924/scripts/build_geometry24_forecast_cot_20260924.py"
    bank_lib = load_module(bank_code, "causal_inputs_for_source_domain")
    snorm = json.loads(Path(cfg["normalization"]).read_text())
    ds = bank_lib.CausalInputs(a.hunan_source, a.data_root, cfg["manifest"], snorm,
                               "/tmp/source_domain_frames_" + sha(Path(cfg["manifest"]))[:12], "val")
    seq_ids = np.load(a.ghi_bank / "val/sequence_indices.npy", allow_pickle=False).astype(int)
    assert len(seq_ids) == 128 and len(np.unique(seq_ids)) == 128
    assert json.loads((a.ghi_bank / "complete.json").read_text())["test_used"] is False

    # Validation CPP retrieval reference index; load shards lazily to avoid duplicating the bank.
    idx = json.loads((a.r_cache / "index.json").read_text())
    assert idx["test_used"] is False
    cpp_lookup = {}
    shard_dirs = {}
    for sh in idx["shards"]:
        d = a.r_cache / sh["directory"]
        shard_dirs[sh["directory"]] = d
        for rec in json.loads((d / "records.json").read_text()):
            if rec["split"] == "val" and rec["included_for_R"]:
                cpp_lookup[rec["utc"]] = (sh["directory"], int(rec["array_row"]))
    arrays = {}
    def get_cpp(shard):
        if shard not in arrays:
            d = shard_dirs[shard]
            arrays[shard] = (np.load(d / "cot.npy", mmap_mode="r"),
                             np.load(d / "mask.npy", mmap_mode="r"))
        return arrays[shard]

    # Prepare inputs and metadata. Each selected sequence contributes one last-history R field.
    inputs, histories, seq_meta = [], [], []
    mean = np.asarray(norm["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[:, None, None]
    for seq in seq_ids:
        row = ds.base.rows[int(seq)]
        rels = row["data_relpaths"].split("|")
        last_two = [ds.historical_frame(rel) for rel in rels[6:8]]
        latest = last_two[-1]
        raw = np.concatenate((np.where(latest[1], latest[0], np.nan), latest[2]), axis=0).astype(np.float32)
        x = (raw - mean) / std
        inputs.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        histories.append(last_two)
        seq_meta.append((int(seq), rels[8:24]))

    cot_fields = []
    with torch.no_grad():
        for start in range(0, len(inputs), a.batch):
            x = torch.from_numpy(np.stack(inputs[start:start + a.batch])).cuda()
            cot_fields.extend(torch.expm1(model(x).float()).cpu().numpy()[:, 0])
            del x
    cot_fields = np.asarray(cot_fields, dtype=np.float32)
    assert cot_fields.shape == (len(seq_ids), 256, 256)

    offsets = np.arange(-2, 3, dtype=np.float32)
    target_r, target_c = np.meshgrid(STATION[0] + offsets, STATION[1] + offsets, indexing="ij")
    rm = {size: [bucket() for _ in range(16)] for size in SIZES}
    incremental = {"64_to_128_new_sources": [bucket() for _ in range(16)],
                   "128_to_256_new_sources": [bucket() for _ in range(16)]}
    common64 = [bucket() for _ in range(16)]
    coverage = {size: [0, 0] for size in SIZES}  # supported CPP pixels / available CPP pixels
    lead_rows = []
    n_sequences_with_cpp = set()
    n_missing_cpp = 0

    for i, (seq, future_rels) in enumerate(seq_meta):
        shift = motion_shift(histories[i])
        cot = cot_fields[i]
        # Require all 13 AGRI inputs to be valid at a sampled source location.
        source_valid = np.asarray(histories[i][-1][1], dtype=bool).all(axis=0)
        for t, rel in enumerate(future_rels):
            stamp = Path(rel).stem
            if stamp not in cpp_lookup:
                n_missing_cpp += 1
                continue
            n_sequences_with_cpp.add(seq)
            shard, ar = cpp_lookup[stamp]
            cot_arr, mask_arr = get_cpp(shard)
            truth = np.asarray(cot_arr[ar], dtype=np.float32)
            valid_truth = np.asarray(mask_arr[ar], dtype=bool)
            tr = truth[STATION[0]-2:STATION[0]+3, STATION[1]-2:STATION[1]+3]
            tv = valid_truth[STATION[0]-2:STATION[0]+3, STATION[1]-2:STATION[1]+3] & np.isfinite(tr)
            sy, sx = target_r - (t + 1) * shift[0], target_c - (t + 1) * shift[1]
            vals = map_coordinates(cot, [sy.ravel(), sx.ravel()], order=1, mode="constant", cval=np.nan, prefilter=False).reshape(5, 5)
            src_ok = map_coordinates(source_valid.astype(np.float32), [sy.ravel(), sx.ravel()], order=1,
                                     mode="constant", cval=0.0, prefilter=False).reshape(5, 5) >= 0.999
            inside = {}
            for size in SIZES:
                top, left = STATION[0] - size // 2, STATION[1] - size // 2
                inside[size] = ((sy >= top) & (sy <= top + size - 1) &
                                (sx >= left) & (sx <= left + size - 1) & src_ok & np.isfinite(vals))
                use = tv & inside[size]
                coverage[size][0] += int(use.sum()); coverage[size][1] += int(tv.sum())
                update(rm[size][t], vals, tr, use)
            common = tv & inside[64]
            update(common64[t], vals, tr, common)
            add128 = tv & inside[128] & ~inside[64]
            add256 = tv & inside[256] & ~inside[128]
            update(incremental["64_to_128_new_sources"][t], vals, tr, add128)
            update(incremental["128_to_256_new_sources"][t], vals, tr, add256)
            lead_rows.append({"sequence": seq, "lead_minutes": (t+1)*15,
                              "motion_row_col_per_15min": shift.tolist(),
                              "cpp_valid_station5_pixels": int(tv.sum()),
                              "source_support_pixels": {str(size): int((tv & inside[size]).sum()) for size in SIZES}})

    a.output.mkdir(parents=True)
    summary = {
        "state": "COMPLETE_NO_TRAINING_REGIONAL_SOURCE_DOMAIN_COMPARISON",
        "test_used": False, "training_performed": False,
        "cohort": "same 128 fixed Sili validation sequences in regional GHI pilot bank",
        "sequences_with_at_least_one_cpp_target": len(n_sequences_with_cpp),
        "future_frames_without_cpp_match": n_missing_cpp,
        "station_neighborhood": "5x5 around Sili row=163,col=148 in the Hunan 256x256 grid",
        "source_windows": "64x64, 128x128, and 256x256, all centred on Sili; only source availability mask changes",
        "retrieval_and_motion": "same frozen exploratory R256 checkpoint on last real historical AGRI; same last-two-history C13 phase-correlation shift for all sizes; constant translation only, no thickness evolution",
        "reference": "CPP validation retrieval; independent producer/physical QA unresolved",
        "metrics_limit": "Per-size scores use each size's available CPP-valid source pixels; added-source bands show errors on newly available pixels. Common-support scores must be equal by construction. No result is a forecast-GHI skill claim.",
        "coverage_by_domain": {str(size): {"supported_cpp_pixels": values[0], "all_cpp_valid_station5_pixels": values[1],
                                            "coverage_fraction": values[0] / values[1] if values[1] else None}
                               for size, values in coverage.items()},
        "per_lead": {
            str(lead): {
                "coverage": {str(size): {
                    "n": rm[size][lead-1]["n"],
                    "mae_cot": finish(rm[size][lead-1])["mae_cot"],
                    "rmse_cot": finish(rm[size][lead-1])["rmse_cot"],
                    "bias_cot": finish(rm[size][lead-1])["bias_cot"],
                    "thick_n_cot_ge30": finish(rm[size][lead-1])["thick_n_cot_ge30"],
                    "thick_rmse_cot": finish(rm[size][lead-1])["thick_rmse_cot"],
                    "thick_miss_rate_pred_lt10": finish(rm[size][lead-1])["thick_miss_rate_pred_lt10"],
                } for size in SIZES},
                "common_support_64": finish(common64[lead-1]),
                "newly_added_sources_64_to_128": finish(incremental["64_to_128_new_sources"][lead-1]),
                "newly_added_sources_128_to_256": finish(incremental["128_to_256_new_sources"][lead-1]),
            } for lead in range(1, 17)
        },
        "lead_records": lead_rows,
    }
    (a.output / "source_domain_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"state": summary["state"], "coverage": summary["coverage_by_domain"],
                      "240min": summary["per_lead"]["16"], "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
