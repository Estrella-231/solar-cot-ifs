"""Metadata-only mapping of Hubei R AGRI inputs to fixed-v2 CPP under frozen dates."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-split", required=True)
    p.add_argument("--day-assignment", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    splits = {}
    with open(a.day_assignment, encoding="utf-8-sig", newline="") as f:
        splits = {r["date_BJT"]: r["split"] for r in csv.DictReader(f)}
    input_hash = sha(a.old_split)
    counters = Counter()
    seen = set()
    missing_agri = []
    missing_cpp_fixed = []
    wrong_old_cpp = []
    rows_out = []
    with open(a.old_split, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            time_text = row["time_bjt"]
            t = datetime.strptime(time_text, "%Y%m%d%H%M")
            frozen_split = splits.get(t.date().isoformat())
            counters[(frozen_split or "unassigned", "all_manifest_rows")] += 1
            if frozen_split not in {"train", "val", "test", "buffer"}:
                raise ValueError(f"Unassigned date {t.date()}")
            key = time_text
            if key in seen:
                raise ValueError(f"Duplicate timestamp {key}")
            seen.add(key)
            old_cpp = Path(row["path_cpp"])
            if "/cpp_16/" not in str(old_cpp):
                wrong_old_cpp.append(str(old_cpp))
                fixed_cpp = old_cpp
            else:
                fixed_cpp = Path(str(old_cpp).replace("/cpp_16/", "/cpp_16_fixed_v2/", 1))
            if not Path(row["path_agri"]).is_file():
                missing_agri.append(key)
            if not fixed_cpp.is_file():
                missing_cpp_fixed.append(key)
            counters[(frozen_split, "agri_path_present")] += int(Path(row["path_agri"]).is_file())
            counters[(frozen_split, "fixedv2_cpp_path_present")] += int(fixed_cpp.is_file())
            counters[(frozen_split, "old_split_test_but_frozen_trainval")] += int(row.get("split") == "test" and frozen_split in {"train", "val"})
            if frozen_split in {"train", "val"}:
                rows_out.append({"time_bjt": time_text, "split": frozen_split,
                                 "path_agri": row["path_agri"], "path_cpp_fixed_v2": str(fixed_cpp),
                                 "historical_split_read_only": row.get("split", "")})
    with open(a.output, "w", encoding="utf-8", newline="") as f:
        fields = ["time_bjt", "split", "path_agri", "path_cpp_fixed_v2", "historical_split_read_only"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)
    report = {
        "status": "TRAIN_VAL_R_PATH_MAPPING_COMPLETE_METADATA_ONLY" if not missing_agri and not missing_cpp_fixed else "R_PATH_MAPPING_HAS_GAPS",
        "source_manifest_sha256": input_hash,
        "frozen_split_assignment_sha256": sha(a.day_assignment),
        "output_sha256": sha(a.output),
        "rows_written_train_val_only": len(rows_out),
        "unique_source_timestamps": len(seen),
        "array_or_CPP_payloads_read": 0,
        "test_CPP_values_read": 0,
        "path_existence_checks_only": True,
        "counts_by_frozen_split": {
            split: {name: counters[(split, name)] for name in
                    ("all_manifest_rows", "agri_path_present", "fixedv2_cpp_path_present", "old_split_test_but_frozen_trainval")}
            for split in ("train", "val", "test", "buffer")
        },
        "missing_agri_trainval_count": sum(1 for r in rows_out if not Path(r["path_agri"]).is_file()),
        "missing_fixedv2_cpp_trainval_count": sum(1 for r in rows_out if not Path(r["path_cpp_fixed_v2"]).is_file()),
        "wrong_historical_cpp_root_count": len(wrong_old_cpp),
        "output_scope": "Only frozen train/val path rows written. Test paths checked for metadata existence only.",
        "required_next": "Before payload use, independently verify fixed_v2 file schema, AGRI channel/geometry schema, no-NaN-target mask, train-only normalization, and real-batch/reload/FP64/profile gates.",
    }
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
