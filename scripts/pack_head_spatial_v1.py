"""Build a test-free spatial-COT sidecar for the immutable head pack."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(project, output):
    project, output = project.resolve(), output.resolve()
    base = project / "data/head_pack_trainval_20260908_v1"
    cache = project / "data/frozen_forecast_trainval_20260907_v1"
    require(output.parent == project / "data" and not output.exists(), "new sidecar path required")
    base_audit = json.loads((base / "audit.json").read_text())
    require(base_audit["state"] == "COMPLETE" and not base_audit["test_used"], "base pack not test-free complete")
    for spec in base_audit["arrays"].values():
        require(sha(base / spec["name"]) == spec["sha256"], "base array hash: " + spec["name"])
    require(sha(base / "rows.csv") == base_audit["rows_csv_sha256"], "base rows hash")
    require(sha(cache / "contract.json") == base_audit["source_cache_contract_sha256"], "frozen cache contract drift")
    rows = {}
    with (base / "rows.csv").open(newline="") as handle:
        for row_id, row in enumerate(csv.DictReader(handle)):
            split = row["split"]
            require(split in ("train", "validation"), "test or unknown row in base pack")
            key = (split, int(row["cache_index"]), int(row["station_index"]), int(row["lead_index"]))
            require(key not in rows, "duplicate base row identity")
            rows[key] = row_id
    require(len(rows) == base_audit["rows"] == 637902, "base row count")
    output.mkdir()
    c_map = np.lib.format.open_memmap(output / "cot_map.npy", mode="w+", dtype="float32",
                                      shape=(len(rows), 1, 16, 16))
    filled = np.zeros(len(rows), dtype=bool)
    total = square = 0.0
    train_pixels = 0
    receipts = []
    started = time.monotonic()
    for split, folder in (("train", "train"), ("validation", "val")):
        complete = json.loads((cache / folder / "complete.json").read_text())
        require(complete["state"] == "COMPLETE" and not complete["test_used"], "source split state")
        for entry in complete["shards"]:
            path = cache / folder / entry["name"]
            receipt = json.loads(path.with_suffix(".json").read_text())
            require(receipt == entry and sha(path) == entry["sha256"], "source receipt/hash " + str(path))
            with np.load(path, allow_pickle=False) as payload:
                indices, cot = payload["indices"], payload["cot_log1p"]
                require(cot.ndim == 5 and cot.shape[1:] == (2, 16, 16, 16), "COT shape")
                require(cot.dtype == np.float32 and np.isfinite(cot).all() and (cot >= 0).all(), "COT values")
                for local, cache_index in enumerate(indices.tolist()):
                    for station in range(2):
                        for lead in range(16):
                            # The frozen cache contains nighttime/unlabelled tuples.  They
                            # are intentionally absent from the head pack; only a missing
                            # key for a labelled base row would be an error at final fill.
                            row_id = rows.get((split, cache_index, station, lead))
                            if row_id is None:
                                continue
                            require(not filled[row_id], "duplicate labelled source identity")
                            value = cot[local, station, lead]
                            c_map[row_id, 0] = value
                            filled[row_id] = True
                            if split == "train":
                                v = value.astype(np.float64, copy=False)
                                total += float(v.sum())
                                square += float(np.square(v).sum())
                                train_pixels += v.size
            receipts.append({"split": split, **entry})
            if len(receipts) % 50 == 0:
                atomic(output / "status.json", {"state": "PACKING", "shards": len(receipts),
                       "rows": int(filled.sum()), "seconds": time.monotonic() - started})
    require(filled.all() and len(receipts) == 1332, "incomplete spatial sidecar")
    mean = total / train_pixels
    std = float(np.sqrt(max(square / train_pixels - mean**2, 0.0)))
    require(np.isfinite(mean) and np.isfinite(std) and std > 1e-12, "bad train COT normalization")
    for start in range(0, len(rows), 2048):
        c_map[start:start + 2048] = (c_map[start:start + 2048] - mean) / std
    require(np.isfinite(c_map).all(), "nonfinite normalized COT")
    c_map.flush()
    norm = {"feature": "log1p(COT) spatial map; source cache already log1p", "mean": mean, "std": std,
            "fit_split": "train", "fit_pixels": train_pixels, "normalization": "one scalar over all train rows and pixels",
            "patch": [16, 16], "test_used": False}
    atomic(output / "norm.json", norm)
    atomic(output / "source_shards.json", receipts)
    artifacts = {"cot_map.npy": {"shape": [len(rows), 1, 16, 16], "dtype": "float32",
                                  "sha256": sha(output / "cot_map.npy")}}
    audit = {"state": "COMPLETE_SPATIAL_COT_SIDECAR", "status": "PASS_TEST_FREE_BINDING",
             "rows": len(rows), "test_used": False, "S_frozen": True, "R_frozen": True,
             "base_pack": str(base), "base_pack_audit_sha256": sha(base / "audit.json"),
             "base_rows_sha256": sha(base / "rows.csv"), "source_cache": str(cache),
             "source_cache_contract_sha256": sha(cache / "contract.json"), "source_shards": len(receipts),
             "source_shards_sha256": sha(output / "source_shards.json"), "all_rows_filled_once": True,
             "all_source_hashes_verified": True, "normalization": norm,
             "norm_sha256": sha(output / "norm.json"), "arrays": artifacts,
             "script_sha256": sha(__file__), "seconds": time.monotonic() - started}
    atomic(output / "audit.json", audit)
    atomic(output / "status.json", {"state": "COMPLETE", "rows": len(rows), "seconds": audit["seconds"]})
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output)
