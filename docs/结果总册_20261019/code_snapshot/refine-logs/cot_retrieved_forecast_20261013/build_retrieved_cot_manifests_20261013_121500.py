#!/usr/bin/env python3
"""Build train/val-only UTC-keyed frame and 8-to-16 COT sequence manifests.

This is a metadata-only preparation step. It never opens AGRI, CPP, COT, or
test payloads. Rows assigned to test are skipped before their frame paths are
parsed. The monthly manifest's .npy filename timestamps are UTC; BJT is UTC+8.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


UTC = timezone.utc
BJT = timezone(timedelta(hours=8))
CADENCE = timedelta(minutes=15)


def parse_frame_time(relpath: str) -> datetime:
    stem = Path(relpath).stem
    return datetime.strptime(stem, "%Y%m%d%H%M").replace(tzinfo=UTC)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--day-assignment", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    assigned: dict[str, str] = {}
    with args.day_assignment.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            assigned[row["date_BJT"]] = row["split"]

    frames: dict[str, dict] = {}
    sequences: list[dict] = []
    role_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    skipped_test = skipped_other = 0
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split == "test":
                skipped_test += 1
                continue
            if split not in {"train", "val"}:
                skipped_other += 1
                continue

            paths = row["data_relpaths"].split("|")
            if len(paths) != 24:
                raise ValueError(f"{row['seq_id']}: expected 24 frames, got {len(paths)}")
            times = [parse_frame_time(p) for p in paths]
            if any(b - a != CADENCE for a, b in zip(times, times[1:])):
                raise ValueError(f"{row['seq_id']}: frames are not 15-minute contiguous")
            start_bjt = times[0].astimezone(BJT)
            if start_bjt.strftime("%Y-%m-%d %H:%M:%S") != row["BJT_start"]:
                raise ValueError(f"{row['seq_id']}: UTC filename and BJT_start disagree")

            for i, (p, t) in enumerate(zip(paths, times)):
                bjt_date = t.astimezone(BJT).date().isoformat()
                if assigned.get(bjt_date) != split:
                    raise ValueError(
                        f"{row['seq_id']}: frame {iso(t)} maps to {assigned.get(bjt_date)!r}, "
                        f"not sequence split {split!r}"
                    )
                key = iso(t)
                prior = frames.get(key)
                if prior is not None and prior["split"] != split:
                    raise ValueError(f"cross-split duplicate frame key: {key}")
                if prior is None:
                    prior = frames[key] = {
                        "frame_utc": key,
                        "frame_bjt": iso(t.astimezone(BJT)),
                        "split": split,
                        "data_relpath": p,
                        "history_occurrences": 0,
                        "future_occurrences": 0,
                    }
                role_counts[key][0 if i < 8 else 1] += 1

            init_time = times[7]
            sequences.append({
                "split": split,
                "seq_id": row["seq_id"],
                "init_time_utc": iso(init_time),
                "init_time_bjt": iso(init_time.astimezone(BJT)),
                "history_frame_utc": "|".join(iso(t) for t in times[:8]),
                "target_frame_utc": "|".join(iso(t) for t in times[8:]),
                "history_relpaths": "|".join(paths[:8]),
                "target_relpaths": "|".join(paths[8:]),
            })

    for key, rec in frames.items():
        rec["history_occurrences"], rec["future_occurrences"] = role_counts[key]

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame_path = args.outdir / "unique_frames_train_val.csv"
    seq_path = args.outdir / "sequences_8to16_train_val.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(frames.values())).keys()))
        writer.writeheader()
        writer.writerows(frames[k] for k in sorted(frames))
    with seq_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sequences[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(sequences, key=lambda x: (x["split"], x["init_time_utc"])))

    summary = {
        "status": "PASS_METADATA_ONLY_TRAIN_VAL_MANIFESTS",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "input_manifest_sha256": __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest(),
        "day_assignment_sha256": __import__("hashlib").sha256(args.day_assignment.read_bytes()).hexdigest(),
        "test_payload_access": False,
        "test_rows_skipped_without_parsing_frame_paths": skipped_test,
        "other_rows_skipped": skipped_other,
        "unique_frames": {s: sum(x["split"] == s for x in frames.values()) for s in ("train", "val")},
        "sequences": {s: sum(x["split"] == s for x in sequences) for s in ("train", "val")},
        "cross_split_frame_keys": 0,
        "frame_count": len(frames),
        "outputs": [frame_path.name, seq_path.name],
        "limitation": "metadata index only; no payload existence/readability, R coverage, or COT quality established",
    }
    (args.outdir / "manifest_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
