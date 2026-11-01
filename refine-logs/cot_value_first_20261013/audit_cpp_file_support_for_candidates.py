"""Audit CPP filename/size support for the new train/val target candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--ghi-labels", type=Path, required=True)
    p.add_argument("--cpp-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)

    candidates = []
    frames = {}
    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] not in {"train", "val"}:
                raise ValueError("test/non-train-val row in candidate manifest")
            t = datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S")
            stamp = t.strftime("%Y%m%d%H%M")
            path = a.cpp_root / t.strftime("%Y") / t.strftime("%Y%m%d") / f"FY4B_AGRI_{stamp}00.nc"
            entry = frames.setdefault(stamp, {"split": r["split"], "utc": t, "bjt": r["target_time_bjt"], "path": path})
            if entry["split"] != r["split"]:
                raise ValueError(f"target frame crosses splits: {stamp}")
            candidates.append(r)

    frame_rows = {}
    monthly = Counter()
    split_counts = Counter()
    for stamp, entry in frames.items():
        try:
            size = entry["path"].stat().st_size
            present = True
        except FileNotFoundError:
            size = 0
            present = False
        split, month = entry["split"], entry["utc"].strftime("%Y-%m")
        split_counts[(split, "candidate_frames")] += 1
        split_counts[(split, "cpp_present" if present else "cpp_absent")] += 1
        monthly[(split, month, "cpp_present" if present else "cpp_absent")] += 1
        frame_rows[stamp] = {**entry, "present": present, "size": size}

    labels = {}
    with a.ghi_labels.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = (r["station"], r["target_time_bjt"])
            if k in labels:
                raise ValueError(f"duplicate GHI label key {k}")
            labels[k] = r
    station_counts = Counter()
    monthly_station = Counter()
    for r in candidates:
        label = labels[(r["station"], r["target_time_bjt"])]
        stamp = datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
        entry = frame_rows[stamp]
        split, month = r["split"], entry["utc"].strftime("%Y-%m")
        if label["main_mask"] == "True":
            station_counts[(split, r["station"], "main_mask_rows")] += 1
            monthly_station[(split, month, "main_mask_rows")] += 1
            if entry["present"]:
                station_counts[(split, r["station"], "main_mask_with_cpp")] += 1
                monthly_station[(split, month, "main_mask_with_cpp")] += 1

    a.output_dir.mkdir(parents=True)
    rows_path = a.output_dir / "cpp_file_support_by_frame.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "target_time_utc", "target_time_bjt", "month_bjt", "cpp_present", "cpp_size_bytes", "cpp_path"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for stamp, e in sorted(frame_rows.items()):
            w.writerow({"split": e["split"], "target_time_utc": e["utc"].isoformat(sep=" "),
                "target_time_bjt": e["bjt"], "month_bjt": e["utc"].strftime("%Y-%m"),
                "cpp_present": e["present"], "cpp_size_bytes": e["size"], "cpp_path": str(e["path"])})
    report = {"state": "NEW_MONTHLY_TRAIN_VAL_CPP_FILE_SUPPORT_AUDITED",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "candidate_manifest_sha256": sha(a.candidates), "ghi_kt_label_table_sha256": sha(a.ghi_labels),
        "cpp_root": str(a.cpp_root), "counts": {"/".join(k): v for k,v in sorted(split_counts.items())},
        "main_mask_station_counts": {"/".join(k): v for k,v in sorted(station_counts.items())},
        "monthly_file_support": {"/".join(k): v for k,v in sorted(monthly.items())},
        "monthly_main_mask_station_support": {"/".join(k): v for k,v in sorted(monthly_station.items())},
        "file_support_csv_sha256": sha(rows_path), "file_support_csv": rows_path.name,
        "metadata_only": True, "cpp_payload_opened": False, "test_payload_opened": False}
    (a.output_dir / "cpp_support_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
