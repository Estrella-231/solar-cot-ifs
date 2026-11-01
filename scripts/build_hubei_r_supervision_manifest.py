"""Make explicit R-only supervision eligibility from the closed train/val pack."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    pack, out = Path(a.pack), Path(a.output)
    status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED":
        raise ValueError("Unexpected pack state")
    with (pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    mask = np.load(pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    valid_pixels = mask.reshape(len(rows), -1).sum(axis=1)
    if len(rows) != len(valid_pixels) or any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Pack rows are not closed train/validation data")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "split", "time_bjt", "valid_cpp_pixels", "r_supervision_eligible", "reason"])
        for row, count in zip(rows, valid_pixels):
            eligible = int(count > 0)
            reason = "has_finite_cpp_target_pixels" if eligible else "no_finite_cpp_target_pixels"
            writer.writerow([row["index"], row["split"], row["time_bjt"], int(count), eligible, reason])
    split_report = {}
    for split in ("train", "val"):
        ids = [i for i, row in enumerate(rows) if row["split"] == split]
        split_report[split] = {
            "all_agri_rows_for_inference": len(ids),
            "r_supervision_eligible": int((valid_pixels[ids] > 0).sum()),
            "r_supervision_ineligible_explicit": int((valid_pixels[ids] == 0).sum()),
            "valid_cpp_pixels": int(valid_pixels[ids].sum()),
        }
    report = {
        "state": "HUBEI_R_SUPERVISION_SUPPORT_EXPLICIT_TEST_CLOSED",
        "source_pack_status_sha256": sha(pack / "pack_status.json"),
        "source_rows_sha256": sha(pack / "rows.csv"),
        "source_mask_sha256": sha(pack / "mask.npy"),
        "manifest_sha256": sha(out),
        "test_payloads_read": 0,
        "split": split_report,
        "policy": "Only rows with at least one valid CPP pixel contribute R training/validation loss; preserve every AGRI row for later COT-bank inference. No row is silently removed from the shared GHI sample universe.",
    }
    report_path = out.with_name("R_SUPERVISION_SUPPORT.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
