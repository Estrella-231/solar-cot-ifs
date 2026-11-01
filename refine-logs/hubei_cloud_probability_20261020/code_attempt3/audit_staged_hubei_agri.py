#!/usr/bin/env python3
"""Verify the train/validation-only Hubei AGRI staging manifest and sample payloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--manifest-report", required=True, type=Path)
    p.add_argument("--stage-report", required=True, type=Path)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = json.loads(a.manifest_report.read_text(encoding="utf-8"))
    stage = json.loads(a.stage_report.read_text(encoding="utf-8"))
    if report.get("staged_manifest_state") != "HUBEI_AGRI_TRAINVAL_STAGED_MANIFEST_COMPLETE_TEST_CLOSED":
        raise ValueError("staged manifest state mismatch")
    if report.get("test_payloads_read") != 0 or report.get("test_or_buffer_payloads_staged") != 0:
        raise ValueError("staged manifest violates test/buffer boundary")
    if stage.get("state") != "HUBEI_AGRI_TRAINVAL_ONLY_ARCHIVE_COMPLETE_TEST_CLOSED":
        raise ValueError("source archive state mismatch")
    if stage.get("test_or_buffer_rows_staged") != 0 or stage.get("test_or_buffer_payloads_read") != 0:
        raise ValueError("archive report violates test/buffer boundary")
    archive_sha = sha(a.archive)
    if archive_sha != stage.get("archive_sha256"):
        raise ValueError("staged archive SHA-256 mismatch")
    with a.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 22061 or len({r["time_bjt"] for r in rows}) != len(rows):
        raise ValueError("staged manifest key count/uniqueness mismatch")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    if counts != {"train": 20206, "val": 1855}:
        raise ValueError(f"frozen train/val counts changed: {counts}")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("non-train/validation row found")
    root = Path(report["staged_agri_root"]).resolve()
    for r in rows:
        path = Path(r["path_agri"]).resolve(strict=True)
        if root not in path.parents or path.name != f"{r['time_bjt']}.npz":
            raise ValueError(f"unexpected staged AGRI path for key {r['time_bjt']}")
    train_ix = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val_ix = [i for i, r in enumerate(rows) if r["split"] == "val"]
    sample_ix = sorted({train_ix[0], train_ix[-1], val_ix[0], val_ix[-1]})
    samples = []
    required = {"patch", "cosSOZ", "cosRAA", "day_mask"}
    for i in sample_ix:
        r = rows[i]
        with np.load(r["path_agri"], allow_pickle=False) as z:
            if not required.issubset(z.files):
                raise ValueError(f"missing AGRI/geometry fields at {r['time_bjt']}")
            patch = z["patch"]
            if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
                raise ValueError(f"invalid AGRI patch shape at {r['time_bjt']}: {patch.shape}")
            if any(z[k].shape != (16, 16) for k in ("cosSOZ", "cosRAA", "day_mask")):
                raise ValueError(f"invalid geometry shape at {r['time_bjt']}")
            if not all(np.isfinite(z[k]).all() for k in ("patch", "cosSOZ", "cosRAA", "day_mask")):
                raise ValueError(f"nonfinite staged payload at {r['time_bjt']}")
        samples.append({"time_bjt": r["time_bjt"], "split": r["split"], "patch_shape": list(patch.shape)})
    result = {
        "state": "PASS_HUBEI_AGRI_STAGED_TRAINVAL_AUDIT_TEST_CLOSED",
        "rows": len(rows), "rows_by_split": counts,
        "manifest_sha256": sha(a.manifest), "archive_sha256": archive_sha,
        "sample_payloads": samples, "test_payloads_read": 0,
        "test_or_buffer_payloads_staged": 0,
    }
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
