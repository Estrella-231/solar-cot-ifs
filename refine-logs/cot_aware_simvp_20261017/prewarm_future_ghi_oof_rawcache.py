#!/usr/bin/env python3
"""Prewarm only OOF-fold training AGRI frames into the node-local raw cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--root", required=True)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--folds", nargs="+", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--receipt", required=True, type=Path)
    a = p.parse_args()
    if a.workers != 8 or len(set(a.folds)) != len(a.folds):
        raise ValueError("expected eight workers and unique fold IDs")
    split_root = a.experiment / "results/future_ghi_oof_manifests_v1"
    records = []
    unique = set()
    for fold in a.folds:
        if fold not in range(5):
            raise ValueError(f"invalid fold {fold}")
        d = split_root / f"fold_{fold}"
        audit = json.loads((d / "manifest_audit.json").read_text())
        assert audit["frame_disjoint_splits"] is True
        assert audit["train_val_frame_intersection"] == 0
        assert audit["sequence_counts"]["test"] == 0
        assert audit["validation_test_used"] is False
        frames_path = d / "train_frames.txt"
        rels = [x.strip() for x in frames_path.read_text().splitlines() if x.strip()]
        if not rels or len(rels) != len(set(rels)):
            raise ValueError(f"empty/duplicated train frame list for fold {fold}")
        unique.update(rels)
        records.append({"fold": fold, "train_frame_count": len(rels),
                        "train_frames_sha256": sha(frames_path),
                        "manifest_sha256": audit["manifest_sha256"]})
    sys.path.insert(0, str(a.source))
    from train_monthly_simvp_20261002 import RawCache

    raw = RawCache(a.source, a.root, a.cache)
    rels = sorted(unique)
    a.cache.mkdir(parents=True, exist_ok=True)
    print("RAW_CACHE_PREWARM_BEGIN", json.dumps({"folds": records,
          "unique_train_frames": len(rels), "workers": a.workers,
          "test_payload_opened": False, "validation_payload_opened": False}), flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(raw.frame, rel) for rel in rels]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0 or done == len(rels):
                print("RAW_CACHE_PREWARM_PROGRESS", done, len(rels), flush=True)
    missing = [rel for rel in rels if not (a.cache / (rel + ".npz")).is_file()]
    if missing:
        raise RuntimeError(f"cache verification failed for {len(missing)} frames")
    receipt = {"state": "COMPLETE_OOF_TRAIN_ONLY_RAWCACHE_PREWARM",
               "folds": records, "unique_train_frames": len(rels),
               "cache_path": str(a.cache), "cache_files_verified": len(rels),
               "test_payload_opened": False, "validation_payload_opened": False,
               "cpp_opened": False}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("RAW_CACHE_PREWARM_COMPLETE", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
