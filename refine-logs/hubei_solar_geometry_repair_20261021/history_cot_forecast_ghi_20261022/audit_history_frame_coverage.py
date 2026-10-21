#!/usr/bin/env python3
"""Audit exact train/val history-frame keys for the Hubei historical-COT GHI run.

This reads only CSV/JSON metadata. With --check-files it also stats only paths
referenced by train/validation history frames; it never loads image payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_bjt(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H%M")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-manifest", type=Path, required=True)
    p.add_argument("--ghi-rows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check-files", action="store_true")
    a = p.parse_args()

    stage_rows = read_csv(a.stage_manifest)
    ghi_rows = read_csv(a.ghi_rows)
    stage: dict[str, dict[str, str]] = {}
    duplicate_stage_keys: list[str] = []
    for row in stage_rows:
        bjt = row["bjt"]
        if bjt in stage:
            duplicate_stage_keys.append(bjt)
        stage[bjt] = row

    by_init: dict[str, list[dict[str, str]]] = defaultdict(list)
    target_key_counts: Counter[tuple[str, str, str, str]] = Counter()
    invalid_rows: list[str] = []
    for row in ghi_rows:
        if row["split"] not in {"train", "val"}:
            invalid_rows.append(f"non_train_val:{row.get('index')}:{row['split']}")
            continue
        init = parse_bjt(row["init_time_bjt"])
        target = parse_bjt(row["target_time_bjt"])
        lead = int(row["lead_minutes"])
        if target != init + timedelta(minutes=lead) or lead not in range(15, 241, 15):
            invalid_rows.append(f"bad_time:{row.get('index')}:{row['init_time_bjt']}:{row['target_time_bjt']}:{lead}")
        target_key_counts[(row["station"], row["init_time_bjt"], row["target_time_bjt"], str(lead))] += 1
        by_init[row["init_time_bjt"]].append(row)

    init_results = []
    required_frame_keys: set[str] = set()
    missing: list[dict[str, str]] = []
    split_mismatch: list[dict[str, str]] = []
    source_split_mixed: list[str] = []
    checked_files = 0
    missing_files: list[str] = []
    history_offsets = (105, 90, 75, 60, 45, 30, 15, 0)
    for init_key, rows in sorted(by_init.items()):
        splits = {r["split"] for r in rows}
        if len(splits) != 1:
            source_split_mixed.append(init_key)
            continue
        sample_split = next(iter(splits))
        frame_keys = [(parse_bjt(init_key) - timedelta(minutes=off)).strftime("%Y%m%d%H%M")
                      for off in history_offsets]
        init_missing = []
        for frame_key in frame_keys:
            required_frame_keys.add(frame_key)
            frame = stage.get(frame_key)
            if frame is None:
                missing.append({"init_time_bjt": init_key, "history_time_bjt": frame_key, "reason": "no_stage_row"})
                init_missing.append(frame_key)
                continue
            if frame["split"] != sample_split:
                split_mismatch.append({"init_time_bjt": init_key, "sample_split": sample_split,
                                       "history_time_bjt": frame_key, "frame_split": frame["split"]})
            expected_utc = (parse_bjt(frame_key) - timedelta(hours=8)).strftime("%Y%m%d%H%M")
            if frame["utc"] != expected_utc:
                invalid_rows.append(f"bjt_utc_mismatch:{frame_key}:{frame['utc']}:{expected_utc}")
            if Path(frame["data_path"]).stem != expected_utc:
                invalid_rows.append(f"data_path_timestamp_mismatch:{frame_key}:{frame['data_path']}")
            if Path(frame["aux_path"]).name[:12] != frame_key:
                invalid_rows.append(f"aux_path_timestamp_mismatch:{frame_key}:{frame['aux_path']}")
            if a.check_files:
                for key in ("data_path", "aux_path"):
                    q = Path(frame[key])
                    checked_files += 1
                    if not q.is_file():
                        missing_files.append(str(q))
        init_results.append({"init_time_bjt": init_key, "split": sample_split,
                             "ghi_rows": len(rows), "stations": sorted({r['station'] for r in rows}),
                             "source_roles": sorted({r['source_role'] for r in rows}),
                             "history_frames_required": 8, "history_missing_count": len(init_missing)})

    split_row_counts = Counter(r["split"] for r in ghi_rows)
    station_split = Counter((r["split"], r["station"]) for r in ghi_rows)
    stage_split = Counter(r["split"] for r in stage_rows)
    result = {
        "state": "HISTORY_FRAME_COVERAGE_AUDIT",
        "stage_manifest_path": str(a.stage_manifest), "ghi_rows_path": str(a.ghi_rows),
        "stage_manifest_sha256": sha256(a.stage_manifest), "ghi_rows_sha256": sha256(a.ghi_rows),
        "stage_rows": len(stage_rows), "stage_rows_by_split": dict(stage_split),
        "ghi_rows": len(ghi_rows), "ghi_rows_by_split": dict(split_row_counts),
        "ghi_rows_by_split_station": {f"{s}/{st}": n for (s, st), n in sorted(station_split.items())},
        "unique_init_times": len(by_init), "unique_history_frame_times_required": len(required_frame_keys),
        "missing_history_keys": len(missing), "history_split_mismatches": len(split_mismatch),
        "duplicate_stage_keys": len(duplicate_stage_keys),
        "duplicate_station_init_target_lead_keys": sum(n - 1 for n in target_key_counts.values() if n > 1),
        "mixed_split_inits": len(source_split_mixed), "invalid_rows": len(invalid_rows),
        "check_files": bool(a.check_files), "history_file_paths_checked": checked_files,
        "missing_history_files": len(missing_files), "missing_history_file_examples": missing_files[:20],
        "missing_history_examples": missing[:20], "history_split_mismatch_examples": split_mismatch[:20],
        "invalid_row_examples": invalid_rows[:20], "test_payloads_read": 0, "buffer_payloads_read": 0,
    }
    result["state"] = "PASS" if not any((missing, split_mismatch, duplicate_stage_keys,
                                           source_split_mixed, invalid_rows, missing_files)) else "FAIL"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
