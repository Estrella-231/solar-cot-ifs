#!/usr/bin/env python3
"""Inventory Hubei candidate test-truth source files without decoding GHI cells.

The receipt records exact local paths, file sizes, byte hashes, frozen-date
container assignment, and timestamp-only complete-window support. It does not
decode radiation cells, build truth, read AGRI payloads, or authorize scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from openpyxl import load_workbook


APR_MAY_RECORDED_SHA256 = "d4781696f42231bd2ebf54e05c43eed98ecbf084506be1ab78a0c14b3631d50a"
APR_MAY_DIGEST_EVIDENCE = (
    "solar-cot-ifs/refine-logs/hubei_cot_independent_20261014/"
    "PROGRESS_20261017_COT_SOURCE_REBUILD.md (2026-10-18 source-workbook audit entry)"
)
STATION_SHEET_ALIASES = {
    "Niushou": "牛首供电所",
    "Wangzhai": "王寨供电所",
    "Mizhuang": "米庄供电所",
}


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    return {
        "path": str(resolved),
        "name": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def load_expected_sources(report_path: Path) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entries = report.get("source_files")
    if not isinstance(entries, list):
        raise ValueError("Train/validation source report lacks source_files list")
    expected: dict[str, str] = {}
    for item in entries:
        name, digest = item.get("name"), item.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Malformed source_files entry in train/validation report")
        if name in expected:
            raise ValueError(f"Duplicate source name in report: {name}")
        expected[name] = digest.lower()
    return expected


def read_test_dates(split_path: Path) -> list[str]:
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"date_BJT", "split"}.issubset(reader.fieldnames):
            raise ValueError("Frozen split must contain date_BJT and split columns")
        dates: list[str] = []
        seen: set[str] = set()
        for row in reader:
            if row["split"].strip().lower() != "test":
                continue
            raw = row["date_BJT"].strip().replace("-", "")
            day = datetime.strptime(raw, "%Y%m%d").strftime("%Y%m%d")
            if day in seen:
                raise ValueError(f"Duplicate test date: {day}")
            seen.add(day)
            dates.append(day)
    return sorted(dates)


def zip_member_hashes(path: Path, expected_names: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    members: dict[str, str] = {}
    with zipfile.ZipFile(path, "r") as archive:
        members_by_basename: dict[str, list[str]] = defaultdict(list)
        for name in archive.namelist():
            basename = Path(name).name
            if basename in expected_names:
                members_by_basename[basename].append(name)
        for basename in sorted(expected_names):
            matches = members_by_basename.get(basename, [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one ZIP member named {basename}; found {len(matches)}")
            member_name = matches[0]
            with archive.open(member_name, "r") as member:
                hashes[basename] = sha256_stream(member)
            members[basename] = member_name
    return hashes, members


def workbook_time_coverage(source: BinaryIO, test_dates: set[str]) -> dict[str, object]:
    """Read only column Q (the previously audited timestamp column)."""
    workbook = load_workbook(source, read_only=True, data_only=True)
    try:
        by_sheet: dict[str, dict[str, object]] = {}
        for sheet in workbook.worksheets:
            first: datetime | None = None
            last: datetime | None = None
            rows_with_time = 0
            frozen_test_dates_seen: set[str] = set()
            for row in sheet.iter_rows(min_col=17, max_col=17, values_only=True):
                value = row[0]
                if value is None or value == "":
                    continue
                if isinstance(value, datetime):
                    stamp = value
                else:
                    raw = str(value).strip()
                    try:
                        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                rows_with_time += 1
                if first is None or stamp < first:
                    first = stamp
                if last is None or stamp > last:
                    last = stamp
                day = stamp.strftime("%Y%m%d")
                if day in test_dates:
                    frozen_test_dates_seen.add(day)
            by_sheet[sheet.title] = {
                "timestamp_rows": rows_with_time,
                "first_timestamp": first.isoformat(sep=" ") if first else None,
                "last_timestamp": last.isoformat(sep=" ") if last else None,
                "frozen_test_dates_with_any_timestamp": sorted(frozen_test_dates_seen),
            }
        return by_sheet
    finally:
        workbook.close()


def archive_member_time_coverage(path: Path, member_name: str, test_dates: set[str]) -> dict[str, object]:
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open(member_name, "r") as member:
            data = member.read()
    return workbook_time_coverage(io.BytesIO(data), test_dates)


def date_source(day: str) -> tuple[str, str]:
    date = datetime.strptime(day, "%Y%m%d").date()
    if date.year == 2025 and 8 <= date.month <= 12 or date.year == 2026 and 1 <= date.month <= 3:
        return "monthly_archive", f"天气数据_{date:%Y%m}.xlsx"
    if date.isoformat() <= "2026-05-31" and date.isoformat() >= "2026-04-01":
        return "apr_may_workbook", "天气数据_2026040100-2026053123.xlsx"
    if "2026-06-01" <= date.isoformat() <= "2026-07-06":
        return "jun_jul6_workbook", "天气数据_2026-06-01-2026-07-06.xlsx"
    if "2026-07-07" <= date.isoformat() <= "2026-07-14":
        return "jul7_14_workbook", "天气数据_2026-07-07-2026-07-14.xlsx"
    raise ValueError(f"No declared source window for frozen test date {day}")


def read_timestamp_support(path: Path, test_dates: set[str]) -> dict[str, object]:
    expected_stations = {"Niushou", "Wangzhai", "Mizhuang"}
    counts: dict[str, int] = defaultdict(int)
    targets: dict[str, set[str]] = {station: set() for station in expected_stations}
    rows_test_date = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"station", "target_time_BJT", "complete15"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Timestamp support must contain {sorted(required)}")
        for row in reader:
            station = row["station"].strip()
            if station not in expected_stations:
                continue
            target = row["target_time_BJT"].strip()
            try:
                dt = datetime.fromisoformat(target.replace("Z", "+00:00"))
            except ValueError:
                # Some source files carry naive BJT strings, which are intentionally kept naive.
                dt = datetime.fromisoformat(target)
            day = dt.strftime("%Y%m%d")
            if day not in test_dates:
                continue
            rows_test_date += 1
            if row["complete15"].strip() not in {"0", "1"}:
                raise ValueError(f"Unexpected complete15 value: {row['complete15']!r}")
            if row["complete15"].strip() == "1":
                stamp = dt.isoformat(timespec="minutes")
                if stamp in targets[station]:
                    raise ValueError(f"Duplicate station target key: {station} {stamp}")
                targets[station].add(stamp)
                counts[station] += 1
    common = set.intersection(*(targets[s] for s in sorted(expected_stations)))
    return {
        "metadata_only": True,
        "test_date_support_rows_seen": rows_test_date,
        "complete15_candidate_targets_by_station": dict(sorted(counts.items())),
        "three_station_common_complete15_candidate_targets": len(common),
        "common_target_set_sha256": hashlib.sha256("\n".join(sorted(common)).encode("utf-8")).hexdigest(),
        "radiation_values_read": False,
        "interpretation": "timestamp completeness candidates only; not validated labels or authorized forecast keys",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--monthly-archive", required=True, type=Path)
    parser.add_argument("--apr-may-workbook", required=True, type=Path)
    parser.add_argument("--jun-jul6-workbook", required=True, type=Path)
    parser.add_argument("--jul7-14-workbook", required=True, type=Path)
    parser.add_argument("--timestamp-support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "trainval_source_report": args.source_report,
        "frozen_split": args.split,
        "monthly_archive": args.monthly_archive,
        "apr_may_workbook": args.apr_may_workbook,
        "jun_jul6_workbook": args.jun_jul6_workbook,
        "jul7_14_workbook": args.jul7_14_workbook,
        "timestamp_support": args.timestamp_support,
    }
    for label, path in inputs.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} must be an existing non-symlink file: {path}")

    expected = load_expected_sources(args.source_report)
    monthly_names = {f"天气数据_{year}{month:02}.xlsx" for year, months in ((2025, range(8, 13)), (2026, range(1, 4))) for month in months}
    missing_report_names = monthly_names.difference(expected)
    if missing_report_names:
        raise ValueError(f"Source report missing monthly members: {sorted(missing_report_names)}")

    observed_monthly, member_paths = zip_member_hashes(args.monthly_archive, monthly_names)
    mismatched_monthly = {
        name: {"expected": expected[name], "observed": observed_monthly[name]}
        for name in sorted(monthly_names) if expected[name] != observed_monthly[name]
    }
    if mismatched_monthly:
        raise ValueError(f"Monthly ZIP member hashes differ from train/validation source report: {mismatched_monthly}")

    standalone_names = {
        "jun_jul6_workbook": "天气数据_2026-06-01-2026-07-06.xlsx",
        "jul7_14_workbook": "天气数据_2026-07-07-2026-07-14.xlsx",
    }
    standalone_records: dict[str, dict[str, object]] = {}
    for key, name in standalone_names.items():
        if name not in expected:
            raise ValueError(f"Source report missing standalone source {name}")
        record = file_record(inputs[key])
        if record["sha256"] != expected[name]:
            raise ValueError(f"Source hash mismatch for {name}: {record['sha256']} != {expected[name]}")
        record["matches_trainval_source_report"] = True
        standalone_records[key] = record

    apr_may_record = file_record(args.apr_may_workbook)
    apr_may_record["matches_trainval_source_report"] = None
    apr_may_record["matches_pinned_digest"] = apr_may_record["sha256"] == APR_MAY_RECORDED_SHA256
    apr_may_record["pinned_digest_sha256"] = APR_MAY_RECORDED_SHA256
    apr_may_record["digest_evidence_path"] = APR_MAY_DIGEST_EVIDENCE
    if not apr_may_record["matches_pinned_digest"]:
        raise ValueError(f"April-May source hash differs from pinned progress-log digest: {apr_may_record['sha256']}")

    test_dates = read_test_dates(args.split)
    test_date_set = set(test_dates)
    date_map: dict[str, dict[str, str]] = {}
    container_counts: dict[str, int] = defaultdict(int)
    for day in test_dates:
        source_key, source_name = date_source(day)
        date_map[day] = {"source_key": source_key, "source_name": source_name}
        container_counts[source_key] += 1
    expected_counts = {"monthly_archive": 16, "apr_may_workbook": 4, "jun_jul6_workbook": 2}
    if dict(container_counts) != expected_counts:
        raise ValueError(f"Frozen test-date to declared source-window counts differ: {dict(container_counts)}")

    source_time_coverage: dict[str, dict[str, object]] = {}
    for name in sorted(monthly_names):
        source_time_coverage[name] = archive_member_time_coverage(
            args.monthly_archive, member_paths[name], test_date_set
        )
    standalone_source_paths = {
        "apr_may_workbook": args.apr_may_workbook,
        **{key: inputs[key] for key in standalone_names},
    }
    for key, path in standalone_source_paths.items():
        source = path.open("rb")
        try:
            source_time_coverage[path.name] = workbook_time_coverage(source, test_date_set)
        finally:
            source.close()

    date_time_evidence: dict[str, dict[str, object]] = {}
    for day, mapping in date_map.items():
        source_key, source_name = mapping["source_key"], mapping["source_name"]
        if source_key == "monthly_archive":
            sheet_coverage = source_time_coverage[source_name]
        else:
            sheet_coverage = source_time_coverage[source_name]
        station_hits = {
            station: (
                day in sheet_coverage.get(sheet_name, {}).get("frozen_test_dates_with_any_timestamp", [])
            )
            for station, sheet_name in STATION_SHEET_ALIASES.items()
        }
        date_time_evidence[day] = {
            "source_name": source_name,
            "station_sheet_timestamp_present": station_hits,
            "all_three_station_sheets_present": len(station_hits) == 3 and all(station_hits.values()),
        }

    member_records = {
        name: {"archive_member": member_paths[name], "size_bytes": None,
               "sha256": observed_monthly[name], "matches_trainval_source_report": True}
        for name in sorted(observed_monthly)
    }
    support = read_timestamp_support(args.timestamp_support, set(test_dates))
    report = {
        "state": "HUBEI_TEST_TRUTH_SOURCE_CANDIDATE_INVENTORY_METADATA_ONLY",
        "scope": "exact source paths and byte hashes; frozen test dates mapped to declared source windows; timestamp-only complete-window support",
        "inputs": {key: file_record(path) for key, path in inputs.items()},
        "monthly_archive_members": member_records,
        "apr_may_workbook_digest_status": apr_may_record,
        "standalone_workbooks": standalone_records,
        "test_target_dates_bjt": test_dates,
        "date_to_source_window": date_map,
        "test_dates_by_source_window": dict(sorted(container_counts.items())),
        "source_workbook_timestamp_column_q_metadata": source_time_coverage,
        "date_to_station_timestamp_presence": date_time_evidence,
        "timestamp_support": support,
        "test_radiation_cells_decoded": 0,
        "test_radiation_values_read": False,
        "test_truth_pack_created": False,
        "test_evaluation_authorized": False,
        "instrument_unit_metadata_verified": False,
        "notes": [
            "The worksheet timestamp column Q was scanned without reading the radiation-value column; frozen test dates are mapped to exact source filenames and per-station timestamp presence is reported.",
            "April-May digest is pinned from a prior progress record and is not independently listed in the train/validation REPORT.json or B0 audit; it is not called an independent source audit.",
            "Timestamp support counts only indicate complete minute-bin candidates and do not establish radiation-value validity, instrument units, test independence, or forecast-key release.",
            "The inherited W/m2 operating contract and right-closed 15-minute label rule are not independent instrument calibration evidence.",
            "No test or buffer radiation cells, AGRI pixels, CPP/COT payloads, or predictions were decoded by this utility.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "state": report["state"],
        "test_dates": len(test_dates),
        "source_window_counts": dict(sorted(container_counts.items())),
        "monthly_members_hash_matched": len(observed_monthly),
        "timestamp_support": support,
        "test_radiation_cells_decoded": 0,
        "test_evaluation_authorized": False,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
