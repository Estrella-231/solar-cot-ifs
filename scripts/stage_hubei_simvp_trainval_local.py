#!/usr/bin/env python3
"""Stage only indexed Hubei train/validation arrays to node-local /tmp."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_one(kind: str, source: Path, dest: Path) -> dict[str, str | int]:
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    fd, tmp_s = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            for block in iter(lambda: src.read(8 << 20), b""):
                dst.write(block)
                h.update(block)
            dst.flush()
            os.fsync(dst.fileno())
        if os.path.getsize(tmp_s) != source.stat().st_size:
            raise IOError(f"Size mismatch while copying {source}")
        os.replace(tmp_s, dest)
    except Exception:
        try:
            os.unlink(tmp_s)
        except FileNotFoundError:
            pass
        raise
    return {"kind": kind, "source": str(source), "destination": str(dest),
            "size_bytes": dest.stat().st_size, "sha256_during_copy": h.hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--stage-root", type=Path,
                   default=Path("/tmp/zjnu_hubei_simvp_stage_20261016"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    root = a.stage_root.resolve()
    if root.parent != Path("/tmp").resolve():
        raise ValueError(f"Staging is restricted to a direct child of /tmp, got {root}")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {root}")
    if a.workers < 1:
        raise ValueError("workers must be >= 1")

    rows: list[dict[str, str]] = []
    data_sources: dict[str, Path] = {}
    aux_sources: dict[str, Path] = {}
    with a.index_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "val"}:
                raise ValueError(f"Unexpected split row: {row['split']}")
            dpaths = row["data_paths"].split("|")
            apaths = row["aux_paths"].split("|")
            if len(dpaths) != 24 or len(apaths) != 24:
                raise ValueError("Every sequence must have 24 aligned data/aux paths")
            for src_s in dpaths:
                src = Path(src_s)
                old = data_sources.setdefault(src.name, src)
                if old != src:
                    raise ValueError(f"Full-scene basename collision: {old} vs {src}")
            for src_s in apaths:
                src = Path(src_s)
                old = aux_sources.setdefault(src.name, src)
                if old != src:
                    raise ValueError(f"Auxiliary basename collision: {old} vs {src}")
            rows.append({"split": row["split"],
                         "data_paths": row["data_paths"],
                         "aux_paths": row["aux_paths"]})
    if not rows:
        raise ValueError("Empty train/validation index")

    tasks = []
    for kind, sources in (("data", data_sources), ("aux", aux_sources)):
        for name, source in sorted(sources.items()):
            tasks.append((kind, source, root / kind / name))
    total_bytes = sum(src.stat().st_size for _, src, _ in tasks)
    free_bytes = shutil.disk_usage(root.parent).free
    if total_bytes > free_bytes * 0.65:
        raise OSError(f"Stage needs {total_bytes} bytes but /tmp has {free_bytes} free; keeping 35% reserve")

    root.mkdir(parents=True, exist_ok=False)
    path_map: dict[str, str] = {}
    manifest_rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(copy_one, kind, src, dest): (kind, src, dest)
                   for kind, src, dest in tasks}
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            manifest_rows.append(record)
            path_map[record["source"]] = record["destination"]
            if n % 500 == 0 or n == len(futures):
                print(f"STAGED_FILES {n}/{len(futures)} bytes={sum(int(x['size_bytes']) for x in manifest_rows)}", flush=True)

    staged_index = root / "index_trainval.csv"
    with staged_index.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "data_paths", "aux_paths"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "split": row["split"],
                "data_paths": "|".join(path_map[p] for p in row["data_paths"].split("|")),
                "aux_paths": "|".join(path_map[p] for p in row["aux_paths"].split("|")),
            })
    manifest_path = root / "stage_manifest.csv"
    with manifest_path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "source", "destination", "size_bytes", "sha256_during_copy"])
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda x: (x["kind"], x["source"])))
    report = {
        "status": "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED",
        "source_index": str(a.index_csv),
        "source_index_sha256": sha256(a.index_csv),
        "stage_root": str(root),
        "sequences": {split: sum(r["split"] == split for r in rows) for split in ("train", "val")},
        "test_sequences": 0,
        "unique_full_scenes": len(data_sources),
        "unique_aux_sidecars": len(aux_sources),
        "staged_files": len(manifest_rows),
        "staged_bytes": sum(int(x["size_bytes"]) for x in manifest_rows),
        "stage_index": str(staged_index),
        "stage_index_sha256": sha256(staged_index),
        "stage_manifest": str(manifest_path),
        "stage_manifest_sha256": sha256(manifest_path),
        "test_or_buffer_payloads_staged": 0,
        "workers": a.workers,
        "cache_is_ephemeral": True,
    }
    report_path = root / "stage_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
