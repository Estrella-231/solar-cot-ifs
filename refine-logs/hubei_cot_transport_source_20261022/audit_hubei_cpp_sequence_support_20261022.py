#!/usr/bin/env python3
"""Audit whether the existing Hubei 16x16 CPP reference bank supports 8->16 sequences.

This reads only the already extracted train/validation timestamp manifest. It does not
open CPP/HDF payloads, labels, test, or buffer data, and it does not authorize training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


CADENCE_MIN = 15
HISTORY_PLUS_FUTURE_FRAMES = 24  # 8 history frames including init + 16 future frames


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_bjt(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H%M")


def count_windows(length: int, stride_frames: int) -> int:
    if length < HISTORY_PLUS_FUTURE_FRAMES:
        return 0
    return (length - HISTORY_PLUS_FUTURE_FRAMES) // stride_frames + 1


def contiguous_segments(times: list[datetime]) -> list[list[datetime]]:
    out: list[list[datetime]] = []
    current: list[datetime] = []
    for t in times:
        if current and t - current[-1] != timedelta(minutes=CADENCE_MIN):
            out.append(current)
            current = []
        current.append(t)
    if current:
        out.append(current)
    return out


def sequence_summary(times: list[datetime]) -> dict:
    times = sorted(times)
    segments = contiguous_segments(times)
    lengths = [len(s) for s in segments]
    init_dates = Counter()
    for seg in segments:
        for i in range(max(0, len(seg) - HISTORY_PLUS_FUTURE_FRAMES + 1)):
            init_time = seg[i + 7]
            init_dates[init_time.strftime("%Y-%m-%d")] += 1
    return {
        "frames": len(times),
        "unique_dates": len({t.date().isoformat() for t in times}),
        "date_range_bjt": [times[0].isoformat(), times[-1].isoformat()] if times else None,
        "continuous_segments": len(segments),
        "segments_at_least_24_frames": sum(n >= HISTORY_PLUS_FUTURE_FRAMES for n in lengths),
        "longest_segment_frames": max(lengths, default=0),
        "windows_stride_15min_overlapping": sum(count_windows(n, 1) for n in lengths),
        "windows_stride_60min": sum(count_windows(n, 4) for n in lengths),
        "windows_nonoverlapping_6h": sum(count_windows(n, 24) for n in lengths),
        "init_dates_with_windows": len(init_dates),
        "windows_by_init_date_stride_15min": dict(sorted(init_dates.items())),
    }


def audit(input_csv: Path, agri_manifest: Path | None, cpp_cutoff_bjt: str) -> dict:
    by_split: dict[str, list[datetime]] = defaultdict(list)
    full_support_by_split: dict[str, list[datetime]] = defaultdict(list)
    support_counts: dict[str, list[int]] = defaultdict(list)
    key_split: dict[datetime, str] = {}
    seen_rows: set[tuple[str, datetime]] = set()
    errors: list[str] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_bjt", "time_cpp_utc", "split", "path_cpp_correct_utc", "valid_cot_pixels", "valid_clp_pixels"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            split = row["split"].strip().lower()
            if split not in {"train", "val"}:
                errors.append(f"row {row_num}: unexpected split {split!r}")
                continue
            bjt = parse_bjt(row["time_bjt"])
            expected_utc = bjt - timedelta(hours=8)
            got_utc = datetime.strptime(row["time_cpp_utc"], "%Y%m%d%H%M%S")
            if got_utc != expected_utc:
                errors.append(f"row {row_num}: BJT/UTC mismatch {bjt} vs {got_utc}")
            expected_name = f"FY4B_AGRI_{expected_utc:%Y%m%d%H%M%S}.nc"
            if Path(row["path_cpp_correct_utc"]).name != expected_name:
                errors.append(f"row {row_num}: CPP basename mismatch")
            valid_cot = int(row["valid_cot_pixels"])
            valid_clp = int(row["valid_clp_pixels"])
            if valid_cot not in {0, 256} or valid_clp not in {0, 256}:
                errors.append(f"row {row_num}: partial local support requires an explicit mask-aware sequence contract")
            if (valid_cot == 256) != (valid_clp == 256):
                errors.append(f"row {row_num}: COT and CLP validity support differ")
            if (split, bjt) in seen_rows:
                errors.append(f"row {row_num}: duplicate {split} timestamp {bjt}")
            seen_rows.add((split, bjt))
            if bjt in key_split and key_split[bjt] != split:
                errors.append(f"row {row_num}: timestamp crosses train/val split: {bjt}")
            key_split[bjt] = split
            by_split[split].append(bjt)
            support_counts[split].append(valid_cot)
            if valid_cot == 256 and valid_clp == 256:
                full_support_by_split[split].append(bjt)

    split_report = {}
    for split in ("train", "val"):
        times = sorted(by_split[split])
        if len(times) != len(set(times)):
            errors.append(f"{split}: duplicate timestamps")
        full_times = sorted(full_support_by_split[split])
        split_report[split] = {
            **sequence_summary(full_times),
            "manifest_frames": len(times),
            "full_support_frames_16x16": len(full_times),
            "zero_support_frames_excluded": sum(v == 0 for v in support_counts[split]),
            "minimum_valid_cot_pixels_per_frame": min(support_counts[split], default=0),
            "maximum_valid_cot_pixels_per_frame": max(support_counts[split], default=0),
        }

    agri_window_potential = None
    if agri_manifest is not None:
        agri_times: dict[str, list[datetime]] = defaultdict(list)
        with agri_manifest.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"time_bjt", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("AGRI manifest must contain time_bjt and split")
            for row_num, row in enumerate(reader, start=2):
                split = row["split"].strip().lower()
                if split not in {"train", "val"}:
                    errors.append(f"AGRI row {row_num}: unexpected split {split!r}")
                    continue
                agri_times[split].append(parse_bjt(row["time_bjt"]))
        cutoff = parse_bjt(cpp_cutoff_bjt)
        agri_window_potential = {
            "manifest": str(agri_manifest),
            "manifest_sha256": sha256(agri_manifest),
            "known_latest_cpp_product_bjt": cutoff.isoformat(),
            "by_split": {},
            "caveat": "Counts are AGRI key candidates only; raw CPP per-frame file/QC/time/grid coverage was not verified, so these are not confirmed CPP-supervised windows.",
        }
        for split in ("train", "val"):
            ts = sorted(agri_times[split])
            if len(ts) != len(set(ts)):
                errors.append(f"AGRI {split}: duplicate timestamps")
            all_summary = sequence_summary(ts)
            within_cutoff = [t for t in ts if t <= cutoff]
            cutoff_summary = sequence_summary(within_cutoff)
            agri_window_potential["by_split"][split] = {
                "all_trainval_manifest": all_summary,
                "agri_frames_at_or_before_known_cpp_product_cutoff": len(within_cutoff),
                "agri_sequence_candidates_at_or_before_cutoff": cutoff_summary,
            }

    return {
        "state": "PASS_FULL_SUPPORT_SEQUENCE_AUDIT_TEST_BUFFER_CLOSED" if not errors else "FAIL_FULL_SUPPORT_SEQUENCE_AUDIT",
        "input_manifest": str(input_csv),
        "input_manifest_sha256": sha256(input_csv),
        "agri_manifest_sequence_potential": agri_window_potential,
        "scope": "existing Hubei station-local 16x16 CPP train/validation manifest; zero-support frames excluded from candidate sequences",
        "payloads_opened": 0,
        "test_buffer_payloads_opened": 0,
        "cadence_minutes": CADENCE_MIN,
        "history_frames_including_init": 8,
        "future_frames": 16,
        "required_contiguous_frames_per_window": HISTORY_PLUS_FUTURE_FRAMES,
        "split_timestamp_overlap": len(set(by_split["train"]) & set(by_split["val"])),
        "by_split": split_report,
        "errors": errors,
        "interpretation": (
            "A passing result establishes only that station-local 16x16 CPP references have contiguous train/validation timestamps. "
            "It does not establish full-region CPP coverage, independent cloud truth, adequate independent validation sample size, or forecast skill."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--agri-manifest", type=Path)
    parser.add_argument("--cpp-cutoff-bjt", default="202603021800")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, args.agri_manifest, args.cpp_cutoff_bjt)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    train = result["by_split"]["train"]
    val = result["by_split"]["val"]
    lines = [
        "# Hubei CPP sequence support audit (train/validation only)",
        "",
        f"State: `{result['state']}`",
        "",
        "The audit used only the existing timestamp manifest; it opened no CPP payload, label, test, or buffer data.",
        "",
        "| Split | Manifest frames | Full-support frames | Zero-support excluded | Dates | 24-frame windows, 15-min stride | 60-min stride | Non-overlap stride | Longest segment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Train | {train['manifest_frames']} | {train['full_support_frames_16x16']} | {train['zero_support_frames_excluded']} | {train['unique_dates']} | {train['windows_stride_15min_overlapping']} | {train['windows_stride_60min']} | {train['windows_nonoverlapping_6h']} | {train['longest_segment_frames']} |",
        f"| Validation | {val['manifest_frames']} | {val['full_support_frames_16x16']} | {val['zero_support_frames_excluded']} | {val['unique_dates']} | {val['windows_stride_15min_overlapping']} | {val['windows_stride_60min']} | {val['windows_nonoverlapping_6h']} | {val['longest_segment_frames']} |",
        "",
        "Only frames with all 256 COT and CLP pixels valid are used to form candidate sequences; zero-support frames are excluded and break continuity. A sample window consists of 8 history frames including initialization plus 16 future frames at 15-minute cadence. Overlapping windows are highly dependent; the independent uncertainty unit remains the target date. Validation counts therefore support only a feasibility screen, not a final skill claim.",
        "",
        "The manifest SHA-256 is `" + result["input_manifest_sha256"] + "`.",
        "",
        result["interpretation"],
        "",
    ]
    if result["agri_manifest_sequence_potential"] is not None:
        potential = result["agri_manifest_sequence_potential"]
        lines.extend([
            "## AGRI timestamp opportunity before the last known CPP product time",
            "",
            "This second view counts allowed AGRI keys and candidate 24-frame sequences only. It does not assert that CPP exists or passes QC for every key.",
            "",
            "| Split | AGRI frames ≤ cutoff | Candidate 24-frame windows | 60-min stride | Candidate init dates |",
            "|---|---:|---:|---:|---:|",
        ])
        for split in ("train", "val"):
            p = potential["by_split"][split]
            q = p["agri_sequence_candidates_at_or_before_cutoff"]
            lines.append(f"| {split.title()} | {p['agri_frames_at_or_before_known_cpp_product_cutoff']} | {q['windows_stride_15min_overlapping']} | {q['windows_stride_60min']} | {q['init_dates_with_windows']} |")
        lines.extend(["", potential["caveat"], ""])
        lines.append("The train/validation-only AGRI manifest SHA-256 is `" + potential["manifest_sha256"] + "`.")
        lines.append("")
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": result["state"], "train": train, "val": val, "errors": result["errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
