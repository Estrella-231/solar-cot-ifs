#!/usr/bin/env python3
"""Build a train/validation-only SimVP -> frozen-R station-patch bank.

Only the first eight (observed) AGRI frames of each sequence are read. Future
times contribute deterministic geometry sidecars only; forecast AGRI comes
exclusively from the frozen SimVP checkpoint. Future observed AGRI and CPP are
never loaded. This is an in-sample train-forecast screening bank, not OOF
paper-grade training input.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("sili", "zhujia")
EXPECTED = {
    "candidate": "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807",
    "simvp_checkpoint": "f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4",
    "simvp_norm": "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d",
    "r_checkpoint": "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001",
    "r_source": "7a2e5cb441cdb5132a537cd03580017801595bdea839719d1773398d267210ce",
    "crosswalk": "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--simvp-root", type=Path, required=True)
    p.add_argument("--r-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--crosswalk", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame-cache", type=Path, required=True)
    p.add_argument("--batch-sequences", type=int, default=8)
    p.add_argument("--r-batch-patches", type=int, default=1024)
    p.add_argument("--limit-sequences", type=int, default=0, help="nonzero is a smoke run only")
    p.add_argument("--preflight-only", action="store_true", help="metadata/hash gate; no model or payload load")
    p.add_argument("--resume", action="store_true", help="resume only a matching incomplete bank")
    a = p.parse_args()
    if a.output.exists() and not a.resume:
        raise FileExistsError(f"refusing to overwrite {a.output}")

    source = a.simvp_root / "code"
    manifests = a.simvp_root / "manifests_v1"
    seq_path = manifests / "hunan_8to16.csv"
    norm_path = manifests / "hunan_train_norm.json"
    simvp_ckpt = a.simvp_root / "runs/simvp_seed42_v1/training/best.pt"
    r_script = a.r_root / "scripts/train_station16_R.py"
    r_ckpt = a.r_root / "results/seed42_v1/best.pt"
    r_norm_path = a.r_root / "results/seed42_v1/normalization.json"
    r_keys_path = a.r_root / "results/new_monthly_R_seed42_cot_bank_v1/frame_keys.csv"
    candidate_path = a.r_root / "input/target_candidates_train_val.csv"
    source_hashes = {}
    for name, path, key in (("candidate", candidate_path, "candidate"),
                            ("SimVP checkpoint", simvp_ckpt, "simvp_checkpoint"),
                            ("SimVP normalization", norm_path, "simvp_norm"),
                            ("R checkpoint", r_ckpt, "r_checkpoint"), ("R source", r_script, "r_source")):
        actual = sha(path) if path.is_file() else None
        if actual != EXPECTED[key]:
            raise ValueError(f"{name} is missing or hash differs from audited source: {path}")
        source_hashes[key] = actual
    if a.crosswalk.resolve() != candidate_path.resolve():
        raise ValueError("--crosswalk must be the audited train/validation candidate table")
    if source_hashes["candidate"] != EXPECTED["crosswalk"]:
        raise ValueError("monthly train/validation candidate contract hash mismatch")
    if not r_keys_path.is_file():
        raise FileNotFoundError(r_keys_path)
    sequence_manifest_hash = sha(seq_path)

    with np.load(a.keys, allow_pickle=False) as z:
        split_code = z["split_code"].astype(np.uint8)
        seq_ix = z["sequence_index_within_split"].astype(np.int32)
        row_count = len(split_code)
        needed = {"train": set(), "val": set()}
        for sc, si in zip(split_code, seq_ix):
            needed["train" if sc == 0 else "val"].add(int(si))
    if row_count != 781504:
        raise ValueError(f"crosswalk row count changed: {row_count}")

    sequence_rows = {s: [] for s in ("train", "val")}
    with seq_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in sequence_rows:
                continue
            sequence_rows[split].append(row)
    for split, rows in sequence_rows.items():
        if not needed[split] or max(needed[split]) >= len(rows):
            raise ValueError(f"crosswalk sequence indices outside {split} manifest")
    selected = [(s, i, sequence_rows[s][i]) for s in ("train", "val") for i in sorted(needed[s])]
    if a.limit_sequences:
        selected = selected[:a.limit_sequences]
    if a.limit_sequences and row_count > a.limit_sequences * 16:
        raise ValueError("smoke-limited bank must not be represented as complete")

    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    if norm.get("validation_test_used") is not False:
        raise ValueError("SimVP normalization lacks train-only provenance")
    amean = np.asarray(norm["mean"], np.float32)
    astd = np.asarray(norm["std"], np.float32)
    if amean.shape != (13,) or astd.shape != (13,) or not np.isfinite(amean).all() or (astd <= 0).any():
        raise ValueError("invalid SimVP normalization")

    # Station coordinates and split are read from the already audited R frame-key
    # manifest; these are metadata, not future satellite pixels.
    frame_index = {}
    frame_coord = {}
    with r_keys_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["split"], r["target_time_utc"])
            frame_index[k] = int(r["index"])
            frame_coord[k] = {s: tuple(map(int, r[f"{s}_grid_rc"].split(","))) for s in STATIONS}

    # The complete row-by-row sequence/label crosswalk has already passed its
    # independent metadata audit. This lightweight preflight checks boundary
    # samples only; the full runtime bank builder verifies every frame key as
    # it is generated, avoiding a second cold scan of hundreds of thousands
    # of overlapping sequence entries while another PBS job is active.
    missing_frame_keys = 0
    preflight_rows = []
    for split in ("train", "val"):
        rows = [item for item in selected if item[0] == split]
        if rows:
            preflight_rows.extend((rows[0], rows[len(rows)//2], rows[-1]))
    for split, seq_ix0, row in preflight_rows:
        paths = row["data_relpaths"].split("|")
        if len(paths) != 24:
            raise ValueError("sequence manifest row must list 24 frames")
        for rel in paths[8:]:
            t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
            # AGRI filename stems and candidate frame keys are UTC. The +8 h
            # conversion is used only to locate BJT geometry sidecars.
            target = t.strftime("%Y-%m-%d %H:%M:%S")
            if (split, target) not in frame_coord:
                missing_frame_keys += 1
    if missing_frame_keys:
        raise ValueError(f"{missing_frame_keys} train/val forecast target frames lack station metadata")
    preflight = {"state": "PASS_FORECAST_BANK_METADATA_PREFLIGHT", "crosswalk_rows": row_count,
        "sequences_selected": len(selected), "train_sequences": len(needed["train"]),
        "validation_sequences": len(needed["val"]), "boundary_sequences_checked": len(preflight_rows),
        "forecast_target_frames_checked": len(preflight_rows)*16,
        "missing_station_frame_keys": missing_frame_keys, "test_payload_opened": False,
        "future_observed_agri_read": False, "cpp_read": False,
        "crosswalk_sha256": source_hashes["candidate"], "candidate_sha256": source_hashes["candidate"],
        "simvp_checkpoint_sha256": source_hashes["simvp_checkpoint"],
        "simvp_norm_sha256": source_hashes["simvp_norm"],
        "r_checkpoint_sha256": source_hashes["r_checkpoint"], "r_source_sha256": source_hashes["r_source"]}
    if a.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for production forecast bank")

    sys.path.insert(0, str(source))
    sys.path.insert(0, str(r_script.parent))
    from hunan_data import load_frame, resolve_aux
    from triad_models import SimVPBackbone
    from train_station16_R import COTUNet

    payload = torch.load(simvp_ckpt, map_location="cpu", weights_only=False)
    md = payload.get("metadata", {})
    if md.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1" or md.get("test_payloads_read") != 0:
        raise ValueError("SimVP checkpoint split/test provenance mismatch")
    simvp = SimVPBackbone(str(source)).cuda().eval()
    simvp.load_state_dict(payload["model"], strict=True)
    del payload
    rp = torch.load(r_ckpt, map_location="cpu", weights_only=False)
    if rp.get("metadata", {}).get("training_script_sha256") != sha(r_script):
        raise ValueError("R checkpoint/source binding mismatch")
    rnorm = json.loads(r_norm_path.read_text(encoding="utf-8"))
    rmean = np.asarray(rnorm["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or rstd.shape != (16, 1, 1) or (rstd <= 0).any():
        raise ValueError("invalid frozen-R normalization")
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(rp["model"], strict=True)
    del rp

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    contract = {"split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1", "rows": len(selected),
        "crosswalk_row_count_expected": row_count, "crosswalk_sha256": source_hashes["candidate"],
        "sequence_manifest_sha256": sequence_manifest_hash,
        "simvp_checkpoint_sha256": source_hashes["simvp_checkpoint"], "simvp_norm_sha256": source_hashes["simvp_norm"],
        "r_checkpoint_sha256": source_hashes["r_checkpoint"], "r_source_sha256": source_hashes["r_source"],
        "producer_sha256": sha(Path(__file__))}
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    n = len(selected)
    progress_path = a.output / "progress.json"
    if a.resume:
        if not a.output.is_dir() or not progress_path.is_file():
            raise ValueError("--resume requires an existing incomplete bank with progress.json")
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        if prior.get("state") != "BUILDING" or prior.get("contract_sha256") != contract_sha:
            raise ValueError("existing partial-bank contract differs; refusing resume")
        start_index = int(prior["sequences_complete"])
        if not 0 <= start_index <= n:
            raise ValueError("partial-bank progress index is out of range or already complete")
    else:
        a.output.mkdir(parents=True)
        start_index = 0
    shapes = {
        "history_agri_raw.npy": (n, 2, 2, 13, 3, 3),
        "history_geometry.npy": (n, 2, 2, 2, 3, 3),
        "history_cot_log1p.npy": (n, 2, 2, 1, 3, 3),
        "forecast_agri_raw.npy": (n, 2, 16, 13, 3, 3),
        "forecast_geometry.npy": (n, 2, 16, 2, 3, 3),
        "forecast_cot_log1p.npy": (n, 2, 16, 1, 3, 3),
    }
    arrays = {}
    for k, shape in shapes.items():
        path = a.output / k
        if a.resume:
            arr = np.load(path, mmap_mode="r+", allow_pickle=False)
            if arr.shape != shape or arr.dtype != np.float32:
                raise ValueError(f"partial bank array contract mismatch: {k}")
        else:
            arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        arrays[k] = arr
    index_path = a.output / "index.csv"
    index_file = index_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(index_file, fieldnames=("bank_index", "split", "sequence_index_within_split", "seq_id"))
    writer.writeheader()
    for bank_i, (split, seq_ix0, row) in enumerate(selected[:start_index]):
        writer.writerow({"bank_index": bank_i, "split": split,
                         "sequence_index_within_split": seq_ix0, "seq_id": row.get("seq_id", "")})
    index_file.flush()
    # Cache R(history) by actual AGRI frame key. Only observed history pixels
    # are passed to load_frame; future history entries are not consulted.
    hist_cache = {}
    started = time.monotonic()
    atomic_json(progress_path, {"state": "BUILDING", "sequences_complete": start_index,
                 "sequences_total": n, "forecast_instances": start_index*32,
                 "contract_sha256": contract_sha, "test_payload_opened": False,
                 "future_observed_agri_read": False, "elapsed_seconds_before_resume": 0.0})

    def cached_observed_frame(rel: str):
        # Reuse the same node-local cache format as the authorized monthly
        # training job. Overlapping forecast windows otherwise reread the same
        # eight observed AGRI images many times from shared storage.
        dst = a.frame_cache / (rel + ".npz")
        if dst.is_file():
            with np.load(dst, allow_pickle=False) as z:
                return z["a"], z["v"], z["g"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(dst) + ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dst.is_file():
                with np.load(dst, allow_pickle=False) as z:
                    return z["a"], z["v"], z["g"]
            image, valid, geom = load_frame(a.data_root, rel)
            tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
            with tmp.open("wb") as f:
                np.savez(f, a=image, v=valid, g=geom)
            tmp.replace(dst)
        return image, valid, geom

    @lru_cache(maxsize=128)
    def cached_future_geometry(rel: str):
        # Cache only deterministic geometry values, never the future AGRI
        # payload. Overlapping 8-to-16 windows share most target frame times.
        t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
        aux = resolve_aux(a.data_root, t + timedelta(hours=8))
        with np.load(aux, allow_pickle=False) as z:
            geom = np.stack([z[k] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
        if geom.shape != (3, 256, 256) or not np.isfinite(geom).all():
            raise ValueError(f"invalid deterministic geometry sidecar for {rel}")
        return geom

    for base in range(start_index, n, a.batch_sequences):
        batch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        batch_rows = selected[base:base + a.batch_sequences]
        xrows, grows, paths_rows, coords_rows, seq_meta = [], [], [], [], []
        for split, seq_ix0, row in batch_rows:
            paths = row["data_relpaths"].split("|")
            if len(paths) != 24:
                raise ValueError("sequence manifest row must list 24 frames")
            hist, geos, times = [], [], []
            for rel in paths[:8]:
                image, valid, geom = cached_observed_frame(rel)
                hist.append(np.where(valid, (image - amean[:, None, None]) / astd[:, None, None], 0).astype(np.float32))
                geos.append(geom.astype(np.float32))
                times.append(datetime.strptime(Path(rel).stem, "%Y%m%d%H%M"))
            # Future frames: read deterministic geometry sidecars only. Never
            # call load_frame() on paths[8:], because that would open AGRI.
            future_geo, target_times = [], []
            for rel in paths[8:]:
                t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
                future_geo.append(cached_future_geometry(rel))
                target_times.append(t.strftime("%Y-%m-%d %H:%M:%S"))
            geometry = geos + future_geo
            if len(geometry) != 24 or not np.isfinite(geometry).all():
                raise ValueError("invalid 24-frame geometry context")
            coord = []
            for target in target_times:
                k = (split, target)
                if k not in frame_coord:
                    raise ValueError(f"missing station-grid metadata for {k}")
                coord.append(frame_coord[k])
            xrows.append(np.stack(hist)); grows.append(np.stack(geometry)); paths_rows.append(paths)
            coords_rows.append(coord); seq_meta.append((split, seq_ix0, row.get("seq_id", ""), times))
        x = torch.from_numpy(np.stack(xrows)).cuda(non_blocking=True)
        g = torch.from_numpy(np.stack(grows)).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            forecast_norm = simvp(x, g).float().cpu().numpy()
        forecast_phys = forecast_norm * astd[None, None, :, None, None] + amean[None, None, :, None, None]
        if forecast_phys.shape != (len(batch_rows), 16, 13, 256, 256) or not np.isfinite(forecast_phys).all():
            raise ValueError("invalid SimVP future AGRI")

        r_inputs, destinations = [], []
        # Forecast inputs: predicted AGRI + corresponding future geometry.
        for bi, (split, seq_ix0, row) in enumerate(batch_rows):
            for lead in range(16):
                for si, station in enumerate(STATIONS):
                    gr, gc = coords_rows[bi][lead][station]; r0, c0 = gr - 8, gc - 8
                    if min(r0, c0) < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                        raise ValueError(f"station crop outside 256 grid: {station} {(gr, gc)}")
                    ag = forecast_phys[bi, lead, :, r0:r0+16, c0:c0+16]
                    ge = grows[bi][8 + lead][:, r0:r0+16, c0:c0+16]
                    rin = (np.concatenate((ag, ge), axis=0) - rmean) / rstd
                    if not np.isfinite(rin).all(): raise ValueError("nonfinite predicted-AGRI R input")
                    r_inputs.append(rin.astype(np.float32)); destinations.append(("forecast", bi, si, lead, ag.copy(), ge.copy(), (r0, c0)))
            # Last two observed history frames are inverted with the same R.
            for hi, path_ix in enumerate((6, 7)):
                rel = paths_rows[bi][path_ix]
                key = (split, rel)
                # Station coordinates are static over this analysis grid; use
                # the first aligned target-frame metadata for this sequence.
                for si, station in enumerate(STATIONS):
                    if (split, rel, si) in hist_cache:
                        continue
                    gr, gc = coords_rows[bi][0][station]; r0, c0 = gr - 8, gc - 8
                    ag = xrows[bi][path_ix, :, r0:r0+16, c0:c0+16] * astd[:, None, None] + amean[:, None, None]
                    ge = grows[bi][path_ix, :, r0:r0+16, c0:c0+16]
                    rin = (np.concatenate((ag, ge), axis=0) - rmean) / rstd
                    r_inputs.append(rin.astype(np.float32)); destinations.append(("history", bi, si, hi, ag.copy(), ge.copy(), key))
        # Run frozen R in bounded batches; scatter station-centered outputs.
        cot_result = np.empty((len(r_inputs), 1, 16, 16), np.float32)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            for lo in range(0, len(r_inputs), a.r_batch_patches):
                xb = torch.from_numpy(np.stack(r_inputs[lo:lo+a.r_batch_patches])).cuda(non_blocking=True)
                cot_result[lo:lo+len(xb)] = rmodel(xb).float().cpu().numpy()
        if not np.isfinite(cot_result).all() or (cot_result < 0).any():
            raise ValueError("invalid frozen-R nonnegative log1p COT")
        forecast_cot = np.full((len(batch_rows), 2, 16, 1, 3, 3), np.nan, np.float32)
        for ri, d in enumerate(destinations):
            mode, bi, si, ti, ag, ge, extra = d
            cot3 = cot_result[ri, :, 7:10, 7:10]
            if mode == "forecast":
                arrays["forecast_agri_raw.npy"][base + bi, si, ti] = ag[:, 7:10, 7:10]
                arrays["forecast_geometry.npy"][base + bi, si, ti] = ge[:2, 7:10, 7:10]
                forecast_cot[bi, si, ti] = cot3
            else:
                key = extra
                hist_cache[(key[0], key[1], si)] = (ag[:, 7:10, 7:10], ge[:2, 7:10, 7:10], cot3)
        for bi, (split, seq_ix0, row) in enumerate(batch_rows):
            for hi, path_ix in enumerate((6, 7)):
                rel = paths_rows[bi][path_ix]
                for si in range(2):
                    hkey = (split, rel, si)
                    if hkey not in hist_cache: raise ValueError(f"missing history COT cache {hkey}")
                    ha, hg, hc = hist_cache[hkey]
                    arrays["history_agri_raw.npy"][base+bi, si, hi] = ha
                    arrays["history_geometry.npy"][base+bi, si, hi] = hg
                    arrays["history_cot_log1p.npy"][base+bi, si, hi] = hc
            arrays["forecast_cot_log1p.npy"][base+bi] = forecast_cot[bi]
            writer.writerow({"bank_index": base+bi, "split": split,
                             "sequence_index_within_split": seq_ix0, "seq_id": row.get("seq_id", "")})
        if base % max(a.batch_sequences * 8, 1) == 0 or base + len(batch_rows) == n:
            for arr in arrays.values(): arr.flush()
            index_file.flush()
            atomic_json(a.output / "progress.json", {"state": "BUILDING", "sequences_complete": min(base+len(batch_rows), n),
                         "sequences_total": n, "forecast_instances": min(base+len(batch_rows), n)*32,
                        "contract_sha256": contract_sha,
                         "test_payload_opened": False, "future_observed_agri_read": False,
                         "last_batch_seconds": time.monotonic()-batch_started,
                         "last_batch_sequences_per_second": len(batch_rows)/max(time.monotonic()-batch_started, 1e-6),
                         "last_batch_peak_cuda_gib": torch.cuda.max_memory_allocated()/1024**3,
                         "elapsed_seconds": time.monotonic()-started})
            print(f"PROGRESS {min(base+len(batch_rows), n)}/{n}", flush=True)
    for arr in arrays.values(): arr.flush()
    index_file.flush(); index_file.close()
    if a.limit_sequences:
        state = "INCOMPLETE_SMOKE_ONLY"
    else:
        state = "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK"
    manifest = {"state": state, "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "rows": n, "crosswalk_row_count_expected": row_count, "forecast_instances": n*32,
        "layouts": {k: list(v) for k, v in shapes.items()}, "stations": list(STATIONS),
        "forecast_source": "frozen monthly Hunan SimVP 8-to-16; only observed 8 history AGRI frames read",
        "forecast_cot_source": "frozen monthly Hunan R applied to predicted AGRI; FP32; no CPP",
        "history_cot_source": "same frozen R applied to last two observed history AGRI frames",
        "normalization": "train-only monthly AGRI norm; frozen-R train-only norm",
        "in_sample_train_forecasts": True, "screening_pilot_only": True,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_read": False, "cpp_read": False,
        "simvp_checkpoint_sha256": source_hashes["simvp_checkpoint"],
        "simvp_norm_sha256": source_hashes["simvp_norm"],
        "r_checkpoint_sha256": source_hashes["r_checkpoint"], "r_source_sha256": source_hashes["r_source"],
        "crosswalk_sha256": source_hashes["candidate"],
        "producer_sha256": contract["producer_sha256"], "sequence_manifest_sha256": sequence_manifest_hash,
        "contract_sha256": contract_sha, "elapsed_seconds": time.monotonic()-started}
    atomic_json(a.output / "bank_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
