#!/usr/bin/env python3
"""Create a train/validation-only transfer tar for the triggered Hubei AL arm."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED_SPLIT = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
EXPECTED_FRAMES = 24705


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data16-root", type=Path, required=True)
    p.add_argument("--supplement-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--file-list", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    manifest = a.base / "manifests/agri_paths_trainval.csv"
    pack = json.loads((a.base / "pack/pack_report.json").read_text())
    features = json.loads((a.base / "FEATURE_REPORT.json").read_text())
    if pack.get("state") != "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED" or pack.get("test_and_buffer_values_read") != 0:
        raise ValueError("GHI pack is not train/validation-only")
    if features.get("state") != "HUBEI_STATION_3X3_RAW_FEATURES_TRAINVAL_COMPLETE_TEST_CLOSED" or features.get("test_payloads_read") != 0:
        raise ValueError("Feature report is not train/validation-only")
    if pack.get("split_sha256") != EXPECTED_SPLIT or features.get("split_sha256") != EXPECTED_SPLIT:
        raise ValueError("Frozen split hash mismatch")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_FRAMES or len({r["time_bjt"] for r in rows}) != EXPECTED_FRAMES:
        raise ValueError("Unexpected train/validation AGRI frame count or duplicate key")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Refusing to stage a test/buffer key")
    relpaths = {"data16": [], "supplement": []}
    source_counts = {"data16": 0, "supplement": 0}
    total_bytes = 0
    for row in rows:
        src = Path(row["path_agri"])
        if src.is_relative_to(a.data16_root):
            group = "data16"
            rel = src.relative_to(a.data16_root)
        elif src.is_relative_to(a.supplement_root):
            group = "supplement"
            rel = src.relative_to(a.supplement_root)
        else:
            raise ValueError(f"Source path is outside the two audited train/validation roots: {src}")
        if len(rel.parts) < 2 or not rel.parts[0].isdigit() or len(rel.parts[0]) != 6:
            raise ValueError(f"Unexpected monthly relative path: {rel}")
        if not src.is_file():
            raise FileNotFoundError(src)
        relpaths[group].append(rel.as_posix())
        source_counts[group] += 1
        total_bytes += src.stat().st_size
    staged_names = [f"data16_label/{x}" for values in relpaths.values() for x in values]
    if len(set(staged_names)) != EXPECTED_FRAMES:
        raise ValueError("Duplicate data16_label source file")
    list_base = a.file_list.with_name(a.file_list.stem + "_data16.txt")
    list_supp = a.file_list.with_name(a.file_list.stem + "_supplement.txt")
    if any(x.exists() for x in (a.archive, a.file_list, a.report, list_base, list_supp)):
        raise FileExistsError("Refusing to overwrite a staging artifact")
    a.file_list.parent.mkdir(parents=True, exist_ok=True)
    list_base.write_text("\n".join(relpaths["data16"]) + "\n", encoding="utf-8")
    list_supp.write_text("\n".join(relpaths["supplement"]) + "\n", encoding="utf-8")

    subprocess.run(["tar", "-cf", str(a.archive), "--transform=s|^|data16_label/|",
                    "-C", str(a.data16_root), "-T", str(list_base)], check=True)
    subprocess.run(["tar", "-rf", str(a.archive), "--transform=s|^|data16_label/|",
                    "-C", str(a.supplement_root), "-T", str(list_supp)], check=True)
    subprocess.run(["tar", "-rf", str(a.archive), "-C", str(a.base),
                    "pack", "assets/frozen_R_20261016", "FEATURE_REPORT.json", "R_BANK_REPORT.json",
                    "manifests/agri_paths_trainval.csv"], check=True)
    result = {
        "state": "HUBEI_AL_TRAINVAL_INPUT_TRANSFER_ARCHIVE_READY_TEST_CLOSED",
        "split_sha256": EXPECTED_SPLIT, "unique_trainval_agri_frames": EXPECTED_FRAMES,
        "source_file_counts": source_counts,
        "trainval_agri_bytes_uncompressed": total_bytes, "test_or_buffer_payloads_staged": 0,
        "archive_bytes": a.archive.stat().st_size, "archive_sha256": sha(a.archive),
        "file_list_data16_sha256": sha(list_base), "file_list_supplement_sha256": sha(list_supp),
        "pack_report_sha256": sha(a.base / "pack/pack_report.json"),
        "feature_report_sha256": sha(a.base / "FEATURE_REPORT.json"),
        "r_bank_report_sha256": sha(a.base / "R_BANK_REPORT.json"),
        "frozen_R_checkpoint_sha256": sha(a.base / "assets/frozen_R_20261016/best.pt"),
        "frozen_R_source_sha256": sha(a.base / "assets/frozen_R_20261016/train_cot_repaired.py"),
        "frozen_R_norm_sha256": sha(a.base / "assets/frozen_R_20261016/norm.json"),
    }
    a.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
