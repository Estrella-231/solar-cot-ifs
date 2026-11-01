#!/usr/bin/env python3
"""Verify a train/validation-only Hubei SimVP stage and rebase its index.

The source stage report and per-file SHA-256 manifest are preserved.  This
utility writes a separate PBS-path index and a verification receipt; it never
reads or stages test/buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "train": 5153,
    "val": 509,
    "files": 23468,
    "bytes": 66508197403,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="PBS stage root containing data/, aux/, and source metadata")
    ap.add_argument("--expected-source-root", default="/tmp/zjnu_hubei_simvp_stage_20261016")
    args = ap.parse_args()
    root = args.root.resolve()
    report_path = root / "stage_report.json"
    manifest_path = root / "stage_manifest.csv"
    index_path = root / "index_trainval.csv"
    for p in (report_path, manifest_path, index_path):
        if not p.is_file():
            raise SystemExit(f"MISSING_REQUIRED_FILE {p}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED":
        raise SystemExit(f"BAD_STAGE_STATUS {report.get('status')!r}")
    if report.get("test_sequences") != 0 or report.get("test_or_buffer_payloads_staged") != 0:
        raise SystemExit("TEST_OR_BUFFER_PAYLOAD_PRESENT_IN_STAGE_REPORT")
    seq = report.get("sequences", {})
    if seq.get("train") != EXPECTED["train"] or seq.get("val") != EXPECTED["val"]:
        raise SystemExit(f"UNEXPECTED_SEQUENCE_COUNTS {seq}")
    if report.get("staged_files") != EXPECTED["files"] or report.get("staged_bytes") != EXPECTED["bytes"]:
        raise SystemExit("UNEXPECTED_STAGE_SIZE")
    if sha256(manifest_path) != report.get("stage_manifest_sha256"):
        raise SystemExit("SOURCE_MANIFEST_SHA256_MISMATCH")

    checked = 0
    total_bytes = 0
    seen: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kind = row["kind"]
            if kind not in {"data", "aux"}:
                raise SystemExit(f"BAD_MANIFEST_KIND {kind!r}")
            name = Path(row["destination"]).name
            rel = f"{kind}/{name}"
            if rel in seen:
                raise SystemExit(f"DUPLICATE_MANIFEST_PATH {rel}")
            seen.add(rel)
            p = root / rel
            if not p.is_file():
                raise SystemExit(f"MISSING_STAGED_PAYLOAD {rel}")
            size = p.stat().st_size
            if size != int(row["size_bytes"]):
                raise SystemExit(f"SIZE_MISMATCH {rel} expected={row['size_bytes']} actual={size}")
            digest = sha256(p)
            if digest != row["sha256_during_copy"]:
                raise SystemExit(f"SHA256_MISMATCH {rel}")
            checked += 1
            total_bytes += size
    if checked != EXPECTED["files"] or total_bytes != EXPECTED["bytes"]:
        raise SystemExit(f"MANIFEST_TOTAL_MISMATCH files={checked} bytes={total_bytes}")

    src_prefix = args.expected_source_root.rstrip("/") + "/"
    dst_prefix = str(root).rstrip("/") + "/"
    out_index = root / "index_trainval_pbs.csv"
    rows = 0
    with index_path.open(newline="", encoding="utf-8") as src, out_index.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames != ["split", "data_paths", "aux_paths"]:
            raise SystemExit(f"UNEXPECTED_INDEX_COLUMNS {reader.fieldnames}")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["split"] not in {"train", "val"}:
                raise SystemExit(f"NON_TRAINVAL_INDEX_ROW {row['split']!r}")
            for col in ("data_paths", "aux_paths"):
                paths = row[col].split("|")
                if not paths or len(paths) != 24:
                    raise SystemExit(f"BAD_24_FRAME_SEQUENCE row={rows} col={col}")
                rebased = []
                for old_path in paths:
                    if not old_path.startswith(src_prefix):
                        raise SystemExit(f"UNEXPECTED_SOURCE_PATH {old_path}")
                    new_path = dst_prefix + old_path[len(src_prefix):]
                    if not Path(new_path).is_file():
                        raise SystemExit(f"INDEX_TARGET_MISSING {new_path}")
                    rebased.append(new_path)
                row[col] = "|".join(rebased)
            writer.writerow(row)
            rows += 1
    if rows != EXPECTED["train"] + EXPECTED["val"]:
        raise SystemExit(f"UNEXPECTED_INDEX_ROWS {rows}")

    receipt = {
        "status": "PBS_TRAINVAL_STAGE_SHA256_AND_INDEX_REBASE_PASSED_TEST_CLOSED",
        "stage_root": str(root),
        "source_stage_status": report["status"],
        "train_sequences": seq["train"],
        "validation_sequences": seq["val"],
        "test_sequences": 0,
        "test_or_buffer_payloads_staged": 0,
        "payload_files_sha256_verified": checked,
        "payload_bytes_verified": total_bytes,
        "source_manifest_sha256": sha256(manifest_path),
        "rebased_index": str(out_index),
        "rebased_index_sha256": sha256(out_index),
        "index_rows": rows,
    }
    receipt_path = root / "stage_verification_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
