"""Pack frozen forecast-COT sequences without using test data or reference CPP.

The frozen forecast cache already holds `cot_log1p` with shape
[sequence, station, forecast_lead, 16, 16].  This script copies those exact
maps into split-local sequence packs.  It never creates a row-wise duplicated
trajectory tensor and never reads GHI or CPP reference products.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def main(root, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise RuntimeError("refuse to overwrite trajectory sidecar")
    base = root / "data/head_pack_trainval_20260908_v1"
    cache = root / "data/frozen_forecast_trainval_20260907_v1"
    audit = json.loads((base / "audit.json").read_text())
    if audit["test_used"] or not audit["S_frozen"] or not audit["R_frozen"]:
        raise RuntimeError("only frozen, test-free source cache is admissible")
    shards = json.loads((base / "source_shards.json").read_text())
    out.mkdir(parents=True)
    result = {"state": "RUNNING", "test_used": False,
              "source_cache_contract_sha256": audit["source_cache_contract_sha256"],
              "base_audit_sha256": sha(base / "audit.json"), "splits": {}}
    for split, split_value, sequence_total in (("train", 0, 27675), ("val", 1, 4258)):
        target = np.lib.format.open_memmap(out / f"{split}_forecast_cot_log1p.npy",
                                           mode="w+", dtype=np.float32,
                                           shape=(sequence_total, 2, 16, 1, 16, 16))
        written = np.zeros(sequence_total, dtype=np.bool_)
        for spec in (s for s in shards if s["split"] == split):
            source = cache / split / spec["name"]
            if sha(source) != spec["sha256"]:
                raise RuntimeError("source shard hash drift: " + str(source))
            with np.load(source, allow_pickle=False) as item:
                indices, cot = item["indices"], item["cot_log1p"]
                if cot.shape != (len(indices), 2, 16, 16, 16):
                    raise RuntimeError("forecast COT shape drift")
                if (indices < 0).any() or (indices >= sequence_total).any() or written[indices].any():
                    raise RuntimeError("sequence index overlap/range drift")
                target[indices, :, :, 0] = cot
                written[indices] = True
        target.flush()
        # Only sequences referenced by the head pack must be available to
        # training. The source cache may contain additional label-free entries.
        seq = np.load(base / "sequence.npy", mmap_mode="r")
        spl = np.load(base / "split.npy", mmap_mode="r")
        needed = np.unique(seq[spl == split_value])
        if not written[needed].all():
            raise RuntimeError("head row references a missing forecast trajectory")
        result["splits"][split] = {"sequence_slots": sequence_total,
                                    "source_sequences_written": int(written.sum()),
                                    "head_sequences_required": int(len(needed)),
                                    "sha256": sha(out / f"{split}_forecast_cot_log1p.npy")}
        atomic(out / "progress.json", result)
    result["state"] = "COMPLETE_FORECAST_COT_TRAJECTORY_SIDECAR"
    result["layout"] = "[sequence, station, forecast_lead, 1, 16, 16] log1p(COT)"
    result["causality"] = "trainer masks forecast leads strictly after each row target lead"
    result["history_cot"] = "not present in source cache; must be built from observed 8-frame AGRI with frozen R"
    atomic(out / "audit.json", result)
    print(result["state"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out)
