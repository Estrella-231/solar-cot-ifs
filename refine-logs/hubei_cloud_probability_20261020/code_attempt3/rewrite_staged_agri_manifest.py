#!/usr/bin/env python3
"""Bind the audited Hubei train/val key list to a verified staged AGRI tree."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", required=True, type=Path)
    p.add_argument("--source-report", required=True, type=Path)
    p.add_argument("--agri-root", required=True, type=Path)
    p.add_argument("--output-manifest", required=True, type=Path)
    p.add_argument("--output-report", required=True, type=Path)
    a = p.parse_args()
    if a.output_manifest.exists() or a.output_report.exists():
        raise FileExistsError("refusing to overwrite a staged manifest")
    report = json.loads(a.source_report.read_text(encoding="utf-8"))
    if report.get("state") != "HUBEI_FULL_AGRI_TRAINVAL_PATH_INDEXED_METADATA_ONLY_TEST_CLOSED":
        raise ValueError("source manifest is not the audited train/val-only manifest")
    if report.get("test_payloads_read") != 0:
        raise ValueError("source manifest report says test values were read")
    if report.get("test_or_buffer_payloads_staged") not in (None, 0):
        raise ValueError("source manifest report is not test/buffer closed")
    if report.get("buffer_payloads_read") not in (None, 0):
        raise ValueError("source manifest report says buffer values were read")
    with a.source_manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != int(report["trainval_paths_written"]):
        raise ValueError("source manifest/report row count mismatch")
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("source manifest contains non-train/validation rows")
    keys = [r["time_bjt"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate time key in source manifest")
    for row in rows:
        path = a.agri_root / f"{row['time_bjt']}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        row["path_agri"] = str(path)
    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.output_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time_bjt", "split", "path_agri"])
        w.writeheader()
        w.writerows(rows)
    output_report = dict(report)
    output_report.update({
        "staged_manifest_state": "HUBEI_AGRI_TRAINVAL_STAGED_MANIFEST_COMPLETE_TEST_CLOSED",
        "source_manifest_sha256": sha(a.source_manifest),
        "source_report_sha256": sha(a.source_report),
        "staged_manifest_sha256": sha(a.output_manifest),
        "staged_agri_root": str(a.agri_root),
        "test_payloads_read": 0,
        "test_or_buffer_payloads_staged": 0,
    })
    a.output_report.write_text(json.dumps(output_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output_report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
