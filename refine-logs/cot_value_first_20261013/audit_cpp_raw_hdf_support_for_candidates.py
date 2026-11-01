"""Metadata-only audit of raw FY-4B HDF inputs behind candidate AGRI frames.

This checks filenames under the CPP portable package's configured AGRI root;
it does not open or decode HDF payloads and does not read test candidates.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--agri-root", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    a = p.parse_args()

    by_frame: dict[str, dict] = {}
    with a.manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rel = row["agri_frame_relpath"]
            frame = Path(rel).stem
            frame_utc = datetime.strptime(frame, "%Y%m%d%H%M")
            target_utc = datetime.strptime(row["target_time_utc"], "%Y-%m-%d %H:%M:%S")
            if frame_utc != target_utc:
                raise ValueError(
                    f"monthly manifest frame path is not the candidate UTC target: "
                    f"{rel} vs {row['target_time_utc']}"
                )
            item = by_frame.setdefault(frame, {"frame": frame, "split": row["split"], "relpath": rel})
            if item["split"] != row["split"]:
                raise ValueError(f"frame has conflicting split: {frame}")

    result = []
    month_counts: Counter[tuple[str, str, str]] = Counter()
    for frame, item in sorted(by_frame.items()):
        # The monthly SimVP manifest stores frame paths in UTC. The candidate
        # builder verifies this by deriving BJT as frame_time + 8 h and checking
        # that the resulting BJT date matches the frozen split calendar. Do not
        # apply the matched-.npz BJT-to-UTC rule to these .npy frame paths.
        stamp_utc = datetime.strptime(frame, "%Y%m%d%H%M")
        day = a.agri_root / f"{stamp_utc:%Y}" / f"{stamp_utc:%Y%m%d}"
        hits = sorted(day.glob(f"*{frame}*")) if day.is_dir() else []
        item.update(
            month=f"{stamp_utc:%Y-%m}",
            hdf_frame_utc=frame,
            hdf_present=bool(hits),
            hdf_count=len(hits),
            hdf_path=str(hits[0]) if hits else "",
        )
        result.append(item)
        month_counts[(item["split"], item["month"], "present" if hits else "absent")] += 1

    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "split", "month", "relpath", "hdf_frame_utc", "hdf_present", "hdf_count", "hdf_path"])
        writer.writeheader()
        writer.writerows(result)
    counts = Counter((r["split"], "present" if r["hdf_present"] else "absent") for r in result)
    report = {
        "state": "RAW_FY4B_AGRI_HDF_FILENAME_SUPPORT_AUDITED",
        "candidate_manifest": str(a.manifest),
        "agri_root": str(a.agri_root),
        "unique_candidate_frames": len(result),
        "counts": {f"{split}/{status}": n for (split, status), n in sorted(counts.items())},
        "monthly_counts": {f"{split}/{month}/{status}": n for (split, month, status), n in sorted(month_counts.items())},
        "metadata_only": True,
        "hdf_payload_opened": False,
        "test_payload_opened": False,
        "matching_rule": "monthly SimVP manifest .npy frame timestamp is UTC (candidate builder derives BJT=UTC+8 h); match the same UTC in the HDF year/date directory and filename; the separate matched-.npz BJT rule does not apply",
        "csv": str(a.output_csv),
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
