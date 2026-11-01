#!/usr/bin/env python3
"""Remap only the physical root of a frozen train/val AGRI manifest.

The time keys and split assignments remain byte-for-byte values from the
audited source manifest. Test/buffer rows are rejected; all mapped files must
exist before a new manifest is emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--expected-rows", type=int, default=24705)
    p.add_argument("--expected-split-sha256", required=True)
    a = p.parse_args()
    if a.output_manifest.exists() or a.report.exists():
        raise FileExistsError("Refusing to overwrite a manifest or report")

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    mapped: set[str] = set()
    with a.source_manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["time_bjt", "split", "path_agri"]:
            raise ValueError(f"Unexpected manifest columns: {reader.fieldnames}")
        for row in reader:
            if row["split"] not in ("train", "val"):
                raise ValueError(f"Refusing non-train/val row: {row}")
            old = Path(row["path_agri"])
            if "data16_label" in old.parts:
                pos = old.parts.index("data16_label")
                rel = Path(*old.parts[pos + 1 :])
                source_class = "legacy_data16_label"
            elif len(old.parts) >= 3 and old.parts[-3] == "data":
                # Supplemental June-July files were generated under the audited
                # full_opt/data/YYYYMM tree and staged by their month/basename.
                rel = Path(*old.parts[-2:])
                source_class = "audited_supplement_data"
            else:
                raise ValueError(f"Unrecognized audited train/val source path: {old}")
            if len(rel.parts) != 2 or len(rel.parts[0]) != 6 or not rel.parts[0].isdigit():
                raise ValueError(f"Unexpected monthly-relative path: {rel}")
            new = a.stage_root / rel
            if not new.is_file():
                missing.append(str(new))
            key = row["time_bjt"]
            if key in mapped:
                raise ValueError(f"Duplicate time key: {key}")
            mapped.add(key)
            counts[row["split"]] += 1
            source_counts[source_class] += 1
            rows.append({"time_bjt": key, "split": row["split"], "path_agri": str(new)})

    if len(rows) != a.expected_rows or len(mapped) != a.expected_rows:
        raise ValueError(f"Unexpected row/key count: rows={len(rows)} keys={len(mapped)}")
    if missing:
        raise FileNotFoundError(f"{len(missing)} staged AGRI files missing; first={missing[0]}")
    if counts["train"] + counts["val"] != a.expected_rows:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")

    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.output_manifest.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_bjt", "split", "path_agri"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "state": "HUBEI_TRAINVAL_MANIFEST_PBS_STAGE_PATHS_VERIFIED_TEST_CLOSED",
        "source_manifest": str(a.source_manifest),
        "source_manifest_sha256": sha256(a.source_manifest),
        "output_manifest": str(a.output_manifest),
        "output_manifest_sha256": sha256(a.output_manifest),
        "stage_root": str(a.stage_root),
        "expected_split_sha256": a.expected_split_sha256,
        "rows": len(rows),
        "unique_time_bjt": len(mapped),
        "split_counts": dict(counts),
        "source_class_counts": dict(source_counts),
        "mapped_trainval_files_checked": len(rows),
        "missing_files": 0,
        "non_trainval_rows": 0,
        "key_and_split_values_preserved": True,
    }
    with a.report.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
