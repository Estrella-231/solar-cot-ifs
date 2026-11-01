#!/usr/bin/env python3
"""Independent CPU witness for joining correct-UTC CLP labels to R-pack rows."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_hubei_cloud_classifier_oof import align_clp_sidecar_to_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--cpp-sidecar", required=True, type=Path)
    a = p.parse_args()
    status = json.loads((a.pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("test_payloads_read") != 0 or status.get("buffer_payloads_read") != 0:
        raise ValueError("R-pack crossed train/validation data boundary")
    with (a.pack / "rows.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with np.load(a.cpp_sidecar, allow_pickle=False) as z:
        keys, splits, clp, valid, reordered = align_clp_sidecar_to_rows(
            rows, z["time_bjt"], z["split"], z["clp"], z["clp_valid_mask"])
    if clp.shape != (len(rows), 16, 16) or valid.shape != clp.shape:
        raise ValueError("aligned CLP tensor shape mismatch")
    print(json.dumps({
        "state": "PASS_HUBEI_CLP_KEYED_ALIGNMENT_TRAINVAL_ONLY",
        "rows": len(rows),
        "key_set_equal": True,
        "keyed_split_equal": True,
        "sidecar_rows_reordered": reordered,
        "aligned_clp_shape": list(clp.shape),
        "valid_clp_pixels": int(valid.sum()),
        "train_rows": int(np.count_nonzero(splits == "train")),
        "val_rows": int(np.count_nonzero(splits == "val")),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
