"""Bounded validation/train-cache audit of geometry used by the COT/GHI heads.

This audit reads no test data and does not invoke S or R.  It quantifies the
difference between the current head's patch-mean solar geometry and the
station-centre geometry on evenly spaced frozen-cache shards.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main(root, output):
    root = root.resolve(); output = output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite geometry audit")
    pack = root / "data/head_pack_trainval_20260908_v1"
    cache = root / "data/frozen_forecast_trainval_20260907_v1"
    pack_audit = json.loads((pack / "audit.json").read_text())
    shards = json.loads((pack / "source_shards.json").read_text())
    if pack_audit["test_used"] or not pack_audit["S_frozen"] or not pack_audit["R_frozen"]:
        raise RuntimeError("input contract drift")
    selected = []
    for split in ("train", "val"):
        candidates = [item for item in shards if item["split"] == split]
        for index in np.linspace(0, len(candidates) - 1, 4, dtype=int):
            selected.append(candidates[index])
    all_delta, all_cos, all_mask = [], [], []
    receipts = []
    for item in selected:
        path = cache / item["split"] / item["name"]
        if sha(path) != item["sha256"]:
            raise RuntimeError("source shard hash drift")
        with np.load(path, allow_pickle=False) as value:
            geometry = value["geometry"].astype(np.float64)
        if geometry.ndim != 6 or geometry.shape[2:] != (16, 3, 16, 16):
            raise RuntimeError("geometry source shape drift")
        centre = geometry[..., 8, 8]
        mean = geometry.mean(axis=(-1, -2))
        delta = centre - mean
        all_delta.append(delta.reshape(-1, 3))
        all_cos.append(geometry[..., :2, :, :].reshape(-1, 2))
        all_mask.append(geometry[..., 2, :, :].reshape(-1))
        receipts.append({"split": item["split"], "name": item["name"], "sha256": item["sha256"]})
    delta = np.concatenate(all_delta); cosine = np.concatenate(all_cos); mask = np.concatenate(all_mask)
    report = {"state": "PASS_BOUNDED_FROZEN_CACHE_GEOMETRY_AUDIT", "test_used": False,
              "source": "evenly spaced 4 train + 4 validation frozen forecast shards; no outcome-based selection",
              "source_shards": receipts, "head_geometry": "target-lead patch mean cosSOZ/cosRAA plus day_mask",
              "station_geometry": "same target-lead geometry at [8,8]", "head_A_AC_identity":
              "both paired groups index the same data[g,station,lead]; only COT tensor is zeroed or retained",
              "cosine_range": {"min": cosine.min(axis=0).tolist(), "max": cosine.max(axis=0).tolist()},
              "day_mask_values": np.unique(mask).tolist(),
              "centre_minus_patch_mean": {"mean_abs": np.abs(delta).mean(axis=0).tolist(),
                                           "p99_abs": np.quantile(np.abs(delta), .99, axis=0).tolist(),
                                           "max_abs": np.abs(delta).max(axis=0).tolist(),
                                           "fraction_abs_gt_0p01": (np.abs(delta) > .01).mean(axis=0).tolist(),
                                           "fraction_abs_gt_0p10": (np.abs(delta) > .10).mean(axis=0).tolist()},
              "boundary": "this verifies cache range, target-lead alignment representation and patch-vs-centre magnitude; it does not independently prove astronomy or aux-file generation"}
    output.parent.mkdir(parents=True, exist_ok=True); atomic(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
