#!/usr/bin/env python3
"""Package only the frozen Hubei train/validation AGRI manifest for GPU staging."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
from pathlib import Path


SOURCE_ROOT = Path("/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/data16_label")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-tar", required=True, type=Path)
    p.add_argument("--output-report", required=True, type=Path)
    a = p.parse_args()
    if a.output_tar.exists() or a.output_report.exists():
        raise FileExistsError("refusing to overwrite an existing AGRI staging artifact")
    with a.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("manifest must contain only frozen train/validation rows")
    keys = [r["time_bjt"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate time_bjt in AGRI manifest")
    root = SOURCE_ROOT.resolve()
    files = []
    total_bytes = 0
    for row in rows:
        src = Path(row["path_agri"]).resolve(strict=True)
        if root not in src.parents or src.suffix != ".npz":
            raise ValueError(f"AGRI path escapes approved train/val root: {src}")
        if not src.is_file():
            raise FileNotFoundError(src)
        size = src.stat().st_size
        total_bytes += size
        files.append((src, f"{row['time_bjt']}.npz", size))

    a.output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output_tar, mode="w") as tf:
        for src, member, _ in files:
            tf.add(src, arcname=member, recursive=False)
    report = {
        "state": "HUBEI_AGRI_TRAINVAL_ONLY_ARCHIVE_COMPLETE_TEST_CLOSED",
        "manifest_sha256": sha(a.manifest),
        "source_root": str(root),
        "rows": len(rows),
        "rows_by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
        "archive_members": len(files),
        "source_bytes": total_bytes,
        "archive_bytes": a.output_tar.stat().st_size,
        "archive_sha256": sha(a.output_tar),
        "member_naming": "<time_bjt>.npz",
        "test_or_buffer_rows_staged": 0,
        "test_or_buffer_payloads_read": 0,
    }
    a.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
