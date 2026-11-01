"""Warm only train/validation AGRI frame cache before a GPU run.

The worker copies each unique frame through the same frozen Hunan loader and
the same node-local .npz format used by the SimVP trainer. It never opens test
rows or labels. This is a performance helper; it does not alter model inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frame_task(task):
    source, root, cache, rel = task
    sys.path.insert(0, source)
    load_frame = importlib.import_module("hunan_data").load_frame
    dst = Path(cache) / (rel + ".npz")
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with dst.with_suffix(dst.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dst.exists():
            return False
        if shutil.disk_usage(cache).free < 25 * 2**30:
            raise RuntimeError("node-local cache reserve below 25 GiB")
        a, valid, geo = load_frame(root, rel)
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            import numpy as np
            np.savez(f, a=a, v=valid, g=geo)
        tmp.replace(dst)
    return True


def main():
    p = argparse.ArgumentParser()
    for name in ("source", "root", "manifest", "cache", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if not 1 <= a.workers <= 6:
        raise ValueError("workers must be between 1 and 6 on the allocated 8-CPU node")
    rows = list(csv.DictReader(a.manifest.open(newline="", encoding="utf-8")))
    trainval = [r for r in rows if r.get("split") in ("train", "val")]
    if not trainval or any(r.get("split") not in ("train", "val", "test", "buffer") for r in rows):
        raise ValueError("unexpected or empty split manifest")
    rels = sorted({rel for row in trainval for rel in row["data_relpaths"].split("|")})
    if not rels or any(not rel or Path(rel).is_absolute() for rel in rels):
        raise ValueError("invalid train/validation frame paths")
    a.cache.mkdir(parents=True, exist_ok=True)
    free0 = shutil.disk_usage(a.cache).free
    if free0 < 25 * 2**30:
        raise RuntimeError("node-local cache reserve below 25 GiB")
    existing = sum((a.cache / (rel + ".npz")).is_file() for rel in rels)
    started = time.time()
    tasks = ((str(a.source), str(a.root), str(a.cache), rel) for rel in rels)
    created = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, did_create in enumerate(pool.map(frame_task, tasks, chunksize=8), 1):
            created += int(did_create)
            if i % 500 == 0:
                print(json.dumps({"event": "prewarm_progress", "checked": i,
                                  "total": len(rels), "created": created}), flush=True)
    missing = [rel for rel in rels if not (a.cache / (rel + ".npz")).is_file()]
    if missing:
        raise RuntimeError(f"prewarm incomplete: {len(missing)} missing cache frames")
    result = {
        "status": "PASS_TRAINVAL_FRAME_CACHE_WARM",
        "manifest_sha256": digest(a.manifest),
        "rows_train_val": len(trainval),
        "unique_train_val_frames": len(rels),
        "existing_before": existing,
        "created": created,
        "missing_after": 0,
        "workers": a.workers,
        "elapsed_seconds": time.time() - started,
        "cache_gib": sum((a.cache / (rel + ".npz")).stat().st_size for rel in rels) / 2**30,
        "test_payloads_read": 0,
        "manifest_sha256_scope": "manifest metadata only; frame payloads restricted to train/validation",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(a.output)
    print("FRAME_CACHE_PREWARM_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
