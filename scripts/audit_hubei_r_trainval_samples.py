"""Read a bounded train/val-only sample of AGRI and fixed-v2 CPP payloads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]


def array_summary(x):
    a = np.asarray(x)
    finite = np.isfinite(a)
    clean = a[finite]
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "finite_fraction": float(finite.mean()),
            "finite_min": float(clean.min()) if clean.size else None,
            "finite_max": float(clean.max()) if clean.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    chosen = {}
    with open(a.manifest, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            chosen.setdefault(row["split"], row)
    if set(chosen) != {"train", "val"}:
        raise ValueError(f"Expected only train/val rows, got {list(chosen)}")
    report = {"status": "BOUNDED_TRAIN_VAL_PAYLOAD_SCHEMA_AUDIT", "test_payloads_opened": 0, "samples": {}}
    for split, row in chosen.items():
        with np.load(row["path_agri"], allow_pickle=True) as z:
            keys = sorted(z.files)
            required = {"patch", "cosSOZ", "cosRAA", "day_mask"}
            if not required.issubset(keys):
                raise ValueError(f"{split} AGRI missing required keys: {sorted(required-set(keys))}")
            patch = z["patch"]
            channels = np.concatenate((patch[AGRI_CH], z["cosSOZ"][None], z["cosRAA"][None], z["day_mask"][None]), axis=0)
            agri_summary = {"keys": keys, "patch": array_summary(patch),
                            "cosSOZ": array_summary(z["cosSOZ"]), "cosRAA": array_summary(z["cosRAA"]),
                            "day_mask": array_summary(z["day_mask"]), "r_input_16ch": array_summary(channels)}
        cpp = np.load(row["path_cpp_fixed_v2"], allow_pickle=False)
        if cpp.ndim != 3 or cpp.shape[0] < 1:
            raise ValueError(f"Unexpected CPP shape {cpp.shape}")
        cot = np.asarray(cpp[0], dtype=np.float32)
        finite = np.isfinite(cot)
        valid = finite & (cot >= 0.0) & (cot <= 100.0)
        cpp_summary = {"array": array_summary(cpp), "cot_plane": array_summary(cot),
                       "finite_0_100_fraction": float(valid.mean()),
                       "finite_nan_fraction": float((~finite).mean()),
                       "effective_mask_pixels": int(valid.sum())}
        report["samples"][split] = {"time_bjt": row["time_bjt"], "agri": agri_summary, "cpp_fixed_v2": cpp_summary}
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
