#!/usr/bin/env python3
"""Require raw and conservative-gate GHI packs to differ only in COT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rm = json.loads((a.raw / "pack_manifest.json").read_text())
    gm = json.loads((a.gate / "pack_manifest.json").read_text())
    for m, name in ((rm, "raw"), (gm, "gate")):
        if m.get("state") != "COMPLETE" or m.get("test_payload_opened") is not False:
            raise ValueError(f"{name} pack completion/test gate")
        if (m.get("rows"), m.get("train"), m.get("val")) != (54937, 50458, 4479):
            raise ValueError(f"{name} pack row counts changed: {(m.get('rows'), m.get('train'), m.get('val'))}")
    if rm.get("R_checkpoint_sha256") != gm.get("R_checkpoint_sha256"):
        raise ValueError("R source differs between packs")
    if rm.get("label_sha256") != gm.get("label_sha256") or rm.get("candidate_sha256") != gm.get("candidate_sha256"):
        raise ValueError("candidate/label source differs")
    raw_rows = list(csv.DictReader((a.raw / "rows.csv").open(newline="", encoding="utf-8")))
    gate_rows = list(csv.DictReader((a.gate / "rows.csv").open(newline="", encoding="utf-8")))
    if len(raw_rows) != len(gate_rows):
        raise ValueError("row count differs")
    key = lambda r: (r["station"], r["target_time_utc"])
    if [key(x) for x in raw_rows] != [key(x) for x in gate_rows]:
        raise ValueError("sample keys/order differ")
    if sha(a.raw / "rows.csv") != sha(a.gate / "rows.csv"):
        raise ValueError("labels/row metadata differ")
    equal_arrays = {}
    for name in ("agri_raw.npy", "agri_valid.npy", "geometry.npy"):
        rp, gp = np.load(a.raw / name, mmap_mode="r", allow_pickle=False), np.load(a.gate / name, mmap_mode="r", allow_pickle=False)
        equal_arrays[name] = bool(rp.shape == gp.shape and np.array_equal(rp, gp, equal_nan=True))
        if not equal_arrays[name]:
            raise ValueError(f"non-COT input differs: {name}")
    rc = np.load(a.raw / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    gc = np.load(a.gate / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if rc.shape != gc.shape or not np.isfinite(rc).all() or not np.isfinite(gc).all():
        raise ValueError("COT arrays invalid or shape mismatch")
    if np.array_equal(rc, gc):
        raise ValueError("gate did not modify COT")
    train = np.asarray([r["split"] == "train" for r in gate_rows])
    if not np.isfinite(float(gm["COT_norm_mean"])) or float(gm["COT_norm_std"]) <= 0:
        raise ValueError("gate normalization invalid")
    if not np.isfinite(float(rm["COT_norm_mean"])) or float(rm["COT_norm_std"]) <= 0:
        raise ValueError("raw normalization invalid")
    result = {
        "state": "PASS_PAIRED_RAW_GATE_PACKS",
        "rows": len(raw_rows), "train": int(train.sum()), "val": int((~train).sum()),
        "unique_sample_keys": len({key(r) for r in raw_rows}),
        "rows_csv_sha256": sha(a.raw / "rows.csv"),
        "non_cot_arrays_identical": equal_arrays,
        "raw_cot_mean_std_train": [rm["COT_norm_mean"], rm["COT_norm_std"]],
        "gate_cot_mean_std_train": [gm["COT_norm_mean"], gm["COT_norm_std"]],
        "raw_cot_sha256": sha(a.raw / "cot_log1p.npy"),
        "gate_cot_sha256": sha(a.gate / "cot_log1p.npy"),
        "test_payloads_read": 0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
