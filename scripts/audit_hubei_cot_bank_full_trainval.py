"""Independent coverage/shape audit for the full frozen Hubei train/val COT bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--support-out", required=True, type=Path)
    a = p.parse_args()

    manifest, bank_rows, labels = map(read_csv, (a.manifest, a.bank / "rows.csv", a.labels))
    bank_report = json.loads((a.bank / "BANK_REPORT.json").read_text(encoding="utf-8"))
    if bank_report.get("state") != "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED":
        raise ValueError("Bank state is not full train/validation, test-closed")
    if bank_report.get("test_payloads_read") != 0 or bank_report.get("test_path_rows_in_manifest") != 0:
        raise ValueError("Test boundary violation in bank report")
    if [r["time_bjt"] for r in manifest] != [r["time_bjt"] for r in bank_rows]:
        raise ValueError("Bank timestamps/order differ from frozen AGRI manifest")
    if any(m["split"] != b["split"] or m["path_agri"] != b["path_agri"] for m, b in zip(manifest, bank_rows)):
        raise ValueError("Bank row split/path does not exactly match the frozen manifest")
    if any(r["split"] not in ("train", "val") for r in manifest + bank_rows + labels):
        raise ValueError("Non-train/validation row found")

    arr = np.load(a.bank / "cot_pred_trainval_full.npy", mmap_mode="r", allow_pickle=False)
    if arr.shape != (len(bank_rows), 1, 16, 16):
        raise ValueError(f"Unexpected COT bank shape: {arr.shape}")
    finite, nonnegative, n = True, True, 0
    vmin, vmax = float("inf"), float("-inf")
    for i in range(0, len(arr), 256):
        x = np.asarray(arr[i:i + 256])
        finite &= bool(np.isfinite(x).all())
        nonnegative &= bool((x >= 0).all())
        n += int(x.size)
        vmin, vmax = min(vmin, float(x.min())), max(vmax, float(x.max()))
    if not finite or not nonnegative:
        raise ValueError("COT bank contains non-finite or negative predictions")

    index = {(r["time_bjt"], r["split"]): i for i, r in enumerate(bank_rows)}
    counts = Counter(r["split"] for r in bank_rows)
    by_station_split = {}
    support_rows = []
    complete_by_time = defaultdict(set)
    for row in labels:
        if row.get("valid_daylight_clear_sky_label") != "1":
            continue
        dt = datetime.strptime(row["target_time_BJT"], "%Y-%m-%d %H:%M:%S")
        key = (dt.strftime("%Y%m%d%H%M"), row["split"])
        idx = index.get(key)
        item = by_station_split.setdefault((row["station"], row["split"]), {"eligible_labels": 0, "matched_agri_cot": 0})
        item["eligible_labels"] += 1
        if idx is None:
            continue
        item["matched_agri_cot"] += 1
        complete_by_time[key].add(row["station"])
        support_rows.append({"station": row["station"], "target_time_BJT": row["target_time_BJT"],
                             "split": row["split"], "cot_bank_row": idx,
                             "valid_daylight_clear_sky_label": 1})

    a.support_out.parent.mkdir(parents=True, exist_ok=True)
    with a.support_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["station", "target_time_BJT", "split", "cot_bank_row",
                                          "valid_daylight_clear_sky_label"])
        w.writeheader()
        w.writerows(support_rows)
    station_split = {f"{s}/{sp}": v for (s, sp), v in sorted(by_station_split.items())}
    common_3station = sum(len(stations) == 3 for stations in complete_by_time.values())
    result = {
        "state": "PASS_FULL_AGRI_TRAINVAL_BANK_KEYED_TEST_CLOSED",
        "manifest_rows": len(manifest), "bank_rows": len(bank_rows), "bank_rows_by_split": dict(counts),
        "bank_rows_match_manifest_exactly": True, "prediction_shape": list(arr.shape),
        "prediction_finite_nonnegative": finite and nonnegative, "prediction_min_max": [vmin, vmax],
        "prediction_element_count": n,
        "daylight_clear_label_support_by_station_split": station_split,
        "three_station_common_target_times_in_split": common_3station,
        "matched_support_rows_written": len(support_rows),
        "support_csv_sha256": sha(a.support_out), "bank_prediction_sha256": sha(a.bank / "cot_pred_trainval_full.npy"),
        "bank_rows_sha256": sha(a.bank / "rows.csv"), "manifest_sha256": sha(a.manifest),
        "test_payloads_read": 0, "ghi_values_used_for_this_audit": False,
        "source_time_semantics_gate": "not decided by filename-key equality; see B0 audit",
        "ghi_unit_gate": "open; this is support/key audit only, not training authorization",
        "train_cot_predictions_are_oof": False,
        "cpp_reference": "cpp_16_fixed_v2 is retrieval supervision, not independent truth",
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
