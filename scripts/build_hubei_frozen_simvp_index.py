#!/usr/bin/env python3
"""Build a train/validation-only frozen 8->16 SimVP index for Hubei.

Frame payloads are read only for frozen train/validation dates. Test and buffer
dates are excluded from both the CSV and the filesystem accesses in this tool.
Existing official aux sidecars are reused when available; missing sidecars are
derived from the same official 20-channel scene and written to an isolated root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
AGRI_CHANNELS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_frozen_split(path: Path) -> dict[str, str]:
    if sha256(path) != SPLIT_SHA:
        raise ValueError("Unexpected frozen split SHA-256")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["date_BJT"]: r["split"] for r in rows}


def load_supplement(path: Path) -> dict[datetime, Path]:
    out: dict[datetime, Path] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            bjt = datetime.strptime(row["time_bjt"], "%Y%m%d%H%M")
            out[bjt] = Path(row["path_full256"])
    return out


def load_trainval(path: Path) -> dict[datetime, str]:
    out: dict[datetime, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "val"}:
                continue
            bjt = datetime.strptime(row["time_bjt"], "%Y%m%d%H%M")
            if bjt in out:
                raise ValueError(f"Duplicate train/val key {bjt}")
            out[bjt] = row["split"]
    return out


def official_data_path(root: Path, bjt: datetime) -> Path:
    utc = bjt - timedelta(hours=8)
    return root / utc.strftime("%Y") / utc.strftime("%Y%m") / utc.strftime("%Y%m%d") / f"{utc:%Y%m%d%H%M}.npy"


def build_aux_index(root: Path, months: list[str]) -> dict[datetime, tuple[Path, int]]:
    """Scan each monthly sidecar directory once (avoid repeated NFS glob scans)."""
    out: dict[datetime, tuple[Path, int]] = {}
    for month in months:
        folder = root / month
        if not folder.is_dir():
            continue
        for path in folder.glob("*_0.0_0.0_0.0_aux.npz"):
            parts = path.stem.split("_")
            if len(parts) != 6 or parts[1] not in {"0", "1"} or parts[2:5] != ["0.0", "0.0", "0.0"]:
                continue
            try:
                bjt = datetime.strptime(parts[0], "%Y%m%d%H%M")
            except ValueError:
                continue
            if bjt.strftime("%Y%m") != month:
                continue
            if bjt in out:
                raise ValueError(f"Multiple solar sidecars for {bjt} under {root}")
            out[bjt] = (path, int(parts[1]))
    return out


def load_or_build_aux(scene_path: Path, bjt: datetime,
                      native_index: dict[datetime, tuple[Path, int]],
                      supplement_index: dict[datetime, tuple[Path, int]],
                      supplement_aux: Path) -> tuple[Path, int]:
    # The official filename already records the center-pixel day flag. Avoid
    # opening a full 256x256 scene when a native or prior derived sidecar exists.
    if bjt in native_index:
        return native_index[bjt]
    if bjt in supplement_index:
        return supplement_index[bjt]

    month = bjt.strftime("%Y%m")
    scene = np.load(scene_path, mmap_mode="r", allow_pickle=False)
    if scene.shape != (20, 256, 256) or scene.dtype != np.float32:
        raise ValueError(f"Unexpected scene contract at {scene_path}: {scene.shape}/{scene.dtype}")
    day = int(scene[19, 128, 128] > 0.5)
    name = f"{bjt:%Y%m%d%H%M}_{day}_0.0_0.0_0.0_aux.npz"
    target_path = supplement_aux / month / name
    soz = np.asarray(scene[18], dtype=np.float32)
    soa = np.asarray(scene[17], dtype=np.float32)
    saa = np.asarray(scene[15], dtype=np.float32)
    day_mask = np.asarray(scene[19], dtype=np.float32)
    cos_soz = np.cos(np.deg2rad(soz)).astype(np.float32)
    cos_raa = np.cos(np.deg2rad(soa - saa)).astype(np.float32)
    if not (np.isfinite(cos_soz).all() and np.isfinite(cos_raa).all() and np.isfinite(day_mask).all()):
        raise ValueError(f"Non-finite solar auxiliary field at {scene_path}")
    atomic_npz(target_path, cosSOZ=cos_soz, cosRAA=cos_raa, day_mask=day_mask)
    supplement_index[bjt] = (target_path, day)
    return target_path, day


def derive_aux_worker(job: tuple[str, str, str]) -> tuple[str, str, int]:
    """Build one missing auxiliary sidecar; each worker writes one unique key."""
    scene_path_s, bjt_s, supplement_aux_s = job
    scene_path = Path(scene_path_s)
    bjt = datetime.strptime(bjt_s, "%Y%m%d%H%M")
    scene = np.load(scene_path, mmap_mode="r", allow_pickle=False)
    if scene.shape != (20, 256, 256) or scene.dtype != np.float32:
        raise ValueError(f"Unexpected scene contract at {scene_path}: {scene.shape}/{scene.dtype}")
    day = int(scene[19, 128, 128] > 0.5)
    name = f"{bjt:%Y%m%d%H%M}_{day}_0.0_0.0_0.0_aux.npz"
    target = Path(supplement_aux_s) / bjt.strftime("%Y%m") / name
    soz = np.asarray(scene[18], dtype=np.float32)
    soa = np.asarray(scene[17], dtype=np.float32)
    saa = np.asarray(scene[15], dtype=np.float32)
    day_mask = np.asarray(scene[19], dtype=np.float32)
    cos_soz = np.cos(np.deg2rad(soz)).astype(np.float32)
    cos_raa = np.cos(np.deg2rad(soa - saa)).astype(np.float32)
    if not (np.isfinite(cos_soz).all() and np.isfinite(cos_raa).all() and np.isfinite(day_mask).all()):
        raise ValueError(f"Non-finite solar auxiliary field at {scene_path}")
    atomic_npz(target, cosSOZ=cos_soz, cosRAA=cos_raa, day_mask=day_mask)
    return bjt_s, str(target), day


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-split", type=Path, required=True)
    p.add_argument("--trainval-manifest", type=Path, required=True)
    p.add_argument("--full256-manifest", type=Path, required=True)
    p.add_argument("--original-data-root", type=Path, required=True)
    p.add_argument("--original-aux-root", type=Path, required=True)
    p.add_argument("--supplement-aux-root", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    p.add_argument("--pre-length", type=int, default=8)
    p.add_argument("--aft-length", type=int, default=16)
    p.add_argument("--stride-minutes", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    for output in (a.output_csv, a.report_json):
        if output.exists():
            raise FileExistsError(output)

    frozen = read_frozen_split(a.frozen_split)
    keys = load_trainval(a.trainval_manifest)
    supplement = load_supplement(a.full256_manifest)
    months = sorted({bjt.strftime("%Y%m") for bjt in keys})
    native_aux_index = build_aux_index(a.original_aux_root, months)
    supplement_aux_index = build_aux_index(a.supplement_aux_root, months)
    records: dict[datetime, dict] = {}
    aux_jobs: list[tuple[str, str, str]] = []
    source_counts = Counter()
    aux_derived = 0
    for bjt, split in sorted(keys.items()):
        if frozen.get(bjt.strftime("%Y-%m-%d")) != split:
            raise ValueError(f"Frozen split mismatch at {bjt}")
        native = official_data_path(a.original_data_root, bjt)
        if native.is_file():
            scene_path = native
            source_counts["original_full_scene"] += 1
        else:
            scene_path = supplement.get(bjt)
            if scene_path is None or not scene_path.is_file():
                raise FileNotFoundError(f"No full scene for train/val key {bjt}")
            source_counts["supplement_full_scene"] += 1
        aux_record = native_aux_index.get(bjt) or supplement_aux_index.get(bjt)
        if aux_record is None:
            aux_jobs.append((str(scene_path), bjt.strftime("%Y%m%d%H%M"), str(a.supplement_aux_root)))
            records[bjt] = {"split": split, "data": str(scene_path), "aux": None, "day": None}
        else:
            aux_path, day = aux_record
            if str(aux_path).startswith(str(a.supplement_aux_root)):
                aux_derived += 1
            records[bjt] = {"split": split, "data": str(scene_path), "aux": str(aux_path), "day": day}

    if aux_jobs:
        with ProcessPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futures = [pool.submit(derive_aux_worker, job) for job in aux_jobs]
            for n, future in enumerate(as_completed(futures), 1):
                stamp, aux_path, day = future.result()
                bjt = datetime.strptime(stamp, "%Y%m%d%H%M")
                records[bjt]["aux"] = aux_path
                records[bjt]["day"] = day
                aux_derived += 1
                if n % 500 == 0 or n == len(futures):
                    print(f"AUX_SIDECARS_DERIVED {n}/{len(futures)}", flush=True)

    by_day: dict[str, list[datetime]] = defaultdict(list)
    for bjt in records:
        by_day[bjt.strftime("%Y%m%d")].append(bjt)
    total = a.pre_length + a.aft_length
    step = timedelta(minutes=a.stride_minutes)
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for day, times in sorted(by_day.items()):
        time_set = set(times)
        for start in sorted(times):
            seq = [start + i * step for i in range(total)]
            if seq[-1].strftime("%Y%m%d") != day or not all(t in time_set for t in seq):
                continue
            rs = [records[t] for t in seq]
            if len({r["split"] for r in rs}) != 1:
                raise ValueError(f"A sequence crosses frozen split at {start}")
            if not all(r["day"] == 1 for r in rs):
                continue
            if rs[0]["split"] not in splits:
                continue
            splits[rs[0]["split"]].append({
                "data_paths": "|".join(r["data"] for r in rs),
                "aux_paths": "|".join(r["aux"] for r in rs),
            })

    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{a.output_csv.name}.", suffix=".tmp", dir=a.output_csv.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "data_paths", "aux_paths"])
            writer.writeheader()
            for split in ("train", "val"):
                for row in splits[split]:
                    writer.writerow({"split": split, **row})
        os.replace(tmp, a.output_csv)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    report = {
        "status": "FROZEN_HUBEI_SIMVP_TRAINVAL_INDEX_BUILT_TEST_CLOSED",
        "frozen_split_sha256": sha256(a.frozen_split),
        "trainval_manifest_sha256": sha256(a.trainval_manifest),
        "full256_manifest_sha256": sha256(a.full256_manifest),
        "output_csv_sha256": sha256(a.output_csv),
        "trainval_frame_keys": len(records),
        "full_scene_sources": dict(source_counts),
        "derived_aux_sidecars": aux_derived,
        "sequence_contract": {"history_frames": a.pre_length, "future_frames": a.aft_length,
                               "cadence_minutes": a.stride_minutes, "same_bjt_day": True,
                               "all_24_frames_day_flag_1": True},
        "sequences": {"train": len(splits["train"]), "val": len(splits["val"]), "test": 0},
        "test_or_buffer_payloads_read": 0,
        "output_csv": str(a.output_csv),
        "supplement_aux_root": str(a.supplement_aux_root),
        "aux_contract": "cosSOZ=cos(SOZ); cosRAA=cos(SOA-SAA); day_mask from official 20-channel scene",
        "normalization": "not computed by this index builder; fit separately from train only before training",
    }
    a.report_json.parent.mkdir(parents=True, exist_ok=True)
    a.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
