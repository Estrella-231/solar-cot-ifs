"""Freeze a candidate split after metadata-only window-support gates pass."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-dir", required=True)
    p.add_argument("--support-report", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    candidate = Path(a.candidate_dir)
    cand_report = load_json(candidate / "REPORT.json")
    support = load_json(a.support_report)
    if not str(cand_report.get("status", "")).startswith("CANDIDATE"):
        raise SystemExit("Candidate report status is not a candidate")
    if support.get("GHI_values_read") != 0 or support.get("test_GHI_values_read") != 0:
        raise SystemExit("Refusing to freeze: GHI/test values were accessed")
    if not support.get("status", "").startswith("METADATA_ONLY"):
        raise SystemExit("Refusing to freeze: support report is not metadata-only")
    by_split = support["by_split"]
    for split in ("val", "test"):
        for station in ("Niushou", "Wangzhai", "Mizhuang"):
            if by_split[split]["stations"][station]["complete15_targets"] <= 0:
                raise SystemExit(f"No complete timestamp windows: {split}/{station}")
    assignment = candidate / "day_assignment.csv"
    rows = list(csv.DictReader(assignment.open(encoding="utf-8-sig", newline="")))
    if any(r["date_BJT"] == "2025-08-01" and r["split"] == "test" for r in rows):
        raise SystemExit("Schema-touched date must not be test")
    if not cand_report.get("shared_R_S_H_date_map", False):
        raise SystemExit("R/S/H date map is not shared")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    frozen_map = out / "day_assignment.csv"
    shutil.copyfile(assignment, frozen_map)
    report = {
        "status": "FROZEN_METADATA_ONLY_GHI15_SUPPORT_PASSED",
        "split_id": cand_report["split_id"].replace("candidate_v1", "frozen_v1"),
        "source_candidate_report": str(candidate / "REPORT.json"),
        "source_candidate_report_sha256": sha256(candidate / "REPORT.json"),
        "source_support_report": str(Path(a.support_report)),
        "source_support_report_sha256": sha256(a.support_report),
        "day_assignment_sha256": sha256(frozen_map),
        "counts": cand_report["counts"],
        "date_range_BJT": cand_report["date_range_BJT"],
        "shared_R_S_H_date_map": True,
        "selection_uses_GHI_CPP_values": False,
        "test_GHI_values_read": 0,
        "test_timestamp_support_read": True,
        "timestamp_support_by_split": by_split,
        "missing_support_days_do_not_change_date_assignment": True,
        "freeze_scope": "Date membership only; no labels, values, normalization, or model results frozen.",
    }
    (out / "FROZEN_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
