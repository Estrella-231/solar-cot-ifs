#!/usr/bin/env python3
"""Independent hash/shape/mask audit for the rebuilt correct-UTC Hubei R pack."""
import argparse
import csv
import hashlib
import json
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
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.pack / "pack_status.json").read_text())
    if status.get("state") != "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED":
        raise ValueError("wrong corrected-UTC R pack state")
    if status.get("test_payloads_read") != 0 or status.get("buffer_payloads_read") != 0:
        raise ValueError("test/buffer boundary failure")
    expected = json.loads((a.pack / "SHA256.json").read_text())
    actual = {n: sha(a.pack / n) for n in expected}
    if actual != expected:
        raise ValueError("pack file SHA mismatch")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8-sig")))
    x = np.load(a.pack / "x_norm.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(a.pack / "target.npy", mmap_mode="r", allow_pickle=False)
    m = np.load(a.pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    n = len(rows)
    if len({r["time_bjt"] for r in rows}) != n or any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("invalid row keys or split")
    if x.shape != (n, 16, 16, 16) or y.shape != (n, 1, 16, 16) or m.shape != y.shape:
        raise ValueError("array shape mismatch")
    if m.dtype != np.bool_ or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite arrays or wrong mask dtype")
    if np.any((y < 0) | (y > 1)) or np.any(y[~m] != 0):
        raise ValueError("target/mask physical contract failure")
    split_counts = {}
    for split in ("train", "val"):
        ix = np.asarray([r["split"] == split for r in rows])
        split_counts[split] = {"rows": int(ix.sum()), "supervised_rows": int(m[ix].reshape(-1, 256).any(axis=1).sum()),
                               "valid_pixels": int(m[ix].sum())}
    if split_counts != status["counts_by_split"]:
        raise ValueError("independent split/support counts differ from pack report")
    result = {"state": "PASS_CORRECT_UTC_R_PACK_INDEPENDENT_AUDIT_TEST_CLOSED",
              "rows": n, "counts_by_split": split_counts,
              "array_shapes": {"x_norm": list(x.shape), "target": list(y.shape), "mask": list(m.shape)},
              "hashes_match": True, "pack_status_sha256": sha(a.pack / "pack_status.json"),
              "test_payloads_read": 0, "buffer_payloads_read": 0,
              "CPP_reference_caveat": "retrieval supervision, not independent truth"}
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
