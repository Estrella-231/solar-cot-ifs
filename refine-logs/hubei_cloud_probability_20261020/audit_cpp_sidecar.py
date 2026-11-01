#!/usr/bin/env python3
"""Independent key/time/shape/value audit for the corrected Hubei CPP sidecar."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--extraction-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.extraction_report.read_text(encoding="utf-8"))
    with a.manifest.open(newline="", encoding="utf-8-sig") as f:
        manifest = list(csv.DictReader(f))
    with a.rows.open(newline="", encoding="utf-8-sig") as f:
        extracted = list(csv.DictReader(f))
    if report.get("state") != "COMPLETE_CORRECT_UTC_CPP_TRAIN_VAL_SIDEcar_TEST_CLOSED":
        raise ValueError("extractor report is not complete")
    if report.get("failures") or len(manifest) != len(extracted) or len(manifest) != report.get("rows"):
        raise ValueError("row count or failure ledger mismatch")
    expected_keys = [r["time_bjt"] for r in manifest]
    expected_splits = [r["split"] for r in manifest]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("manifest contains duplicate keys")
    for i, (m, r) in enumerate(zip(manifest, extracted)):
        if r["time_bjt"] != m["time_bjt"] or r["split"] != m["split"] or int(r["index"]) != i:
            raise ValueError(f"row order/key/split mismatch at row {i}")
        expected_utc = (datetime.strptime(m["time_bjt"], "%Y%m%d%H%M") - timedelta(hours=8)).strftime("%Y%m%d%H%M%S")
        if r["time_cpp_utc"] != expected_utc:
            raise ValueError(f"UTC conversion mismatch at {m['time_bjt']}")
    with np.load(a.sidecar, allow_pickle=False) as z:
        keys = z["time_bjt"].astype(str).tolist()
        splits = z["split"].astype(str).tolist()
        cot = np.asarray(z["cot"], dtype=np.float32)
        cm = np.asarray(z["cot_valid_mask"], dtype=bool)
        clp = np.asarray(z["clp"], dtype=np.uint8)
        lm = np.asarray(z["clp_valid_mask"], dtype=bool)
        lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    n = len(manifest)
    expected_shape = (n, 16, 16)
    if keys != expected_keys or splits != expected_splits:
        raise ValueError("NPZ keys/splits do not match source manifest order")
    if cot.shape != expected_shape or cm.shape != expected_shape or clp.shape != expected_shape or lm.shape != expected_shape:
        raise ValueError("sidecar array shapes mismatch")
    if lat.shape != (16, 16) or lon.shape != (16, 16):
        raise ValueError("sidecar geolocation shapes mismatch")
    if not np.array_equal(cm, np.isfinite(cot) & (cot >= 0) & (cot <= 100)):
        raise ValueError("COT valid-mask semantics mismatch")
    if not np.array_equal(lm, np.isin(clp, (0, 1, 2))):
        raise ValueError("CLP valid-mask semantics mismatch")
    if np.any(cot[cm] < 0) or np.any(cot[cm] > 100) or set(np.unique(clp[lm]).tolist()) - {0, 1, 2}:
        raise ValueError("valid physical/class values out of expected ranges")
    by_split = {}
    for split in ("train", "val"):
        ix = np.asarray([s == split for s in splits])
        by_split[split] = {
            "rows": int(ix.sum()), "cot_valid_pixels": int(cm[ix].sum()),
            "clp_valid_pixels": int(lm[ix].sum()),
            "clp_class_counts": {str(k): int(np.count_nonzero(lm[ix] & (clp[ix] == k))) for k in (0, 1, 2)},
            "cot_range": [float(np.min(cot[ix][cm[ix]])), float(np.max(cot[ix][cm[ix]]))],
        }
    if report.get("sidecar_sha256") != sha(a.sidecar):
        raise ValueError("sidecar SHA does not match extractor report")
    result = {
        "state": "PASS_CORRECT_UTC_CPP_SIDECAR_KEYS_TIME_VALUES_TEST_CLOSED",
        "rows": n, "split_counts": by_split,
        "manifest_sha256": sha(a.manifest), "rows_sha256": sha(a.rows),
        "sidecar_sha256": sha(a.sidecar), "extraction_report_sha256": sha(a.extraction_report),
        "source_report_failures": len(report.get("failures", [])),
        "time_rule": "CPP UTC=BJT-8h; checked row by row",
        "coordinate_error_deg": report.get("max_coordinate_error_deg"),
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "reference_caveat": "CPP is a retrieval reference, not independent cloud truth",
    }
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
