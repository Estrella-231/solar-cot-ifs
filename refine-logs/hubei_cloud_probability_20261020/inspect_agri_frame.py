#!/usr/bin/env python3
"""Inspect one already-authorized train/val AGRI frame and station crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.frame, allow_pickle=False)
    fields = {}
    for key in z.files:
        value = np.asarray(z[key])
        rec = {"shape": list(value.shape), "dtype": str(value.dtype),
               "finite_fraction": float(np.isfinite(value).mean()) if np.issubdtype(value.dtype, np.number) else None}
        if np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)]
            rec.update({"min": float(finite.min()) if finite.size else None,
                        "max": float(finite.max()) if finite.size else None,
                        "zero_fraction": float(np.mean(finite == 0)) if finite.size else None,
                        "nonzero_count": int(np.count_nonzero(finite))})
            if value.ndim >= 2 and value.shape[-2:] == (16, 16):
                crops = {}
                for station, (rr, cc) in CENTERS.items():
                    crop = value[..., rr - 1:rr + 2, cc - 1:cc + 2]
                    crop_f = crop[np.isfinite(crop)]
                    crops[station] = {"shape": list(crop.shape),
                                      "zero_fraction": float(np.mean(crop_f == 0)) if crop_f.size else None,
                                      "nonzero_count": int(np.count_nonzero(crop_f)),
                                      "min": float(crop_f.min()) if crop_f.size else None,
                                      "max": float(crop_f.max()) if crop_f.size else None}
                rec["station_crops"] = crops
        fields[key] = rec
    if "patch" in z.files:
        patch = np.asarray(z["patch"])
        if patch.ndim == 3 and patch.shape[-2:] == (16, 16):
            fields["patch"]["per_channel"] = [
                {"channel": i, "zero_fraction": float(np.mean(patch[i] == 0)),
                 "nonzero_count": int(np.count_nonzero(patch[i])),
                 "min": float(np.nanmin(patch[i])), "max": float(np.nanmax(patch[i])),
                 "median": float(np.nanmedian(patch[i]))}
                for i in range(patch.shape[0])]
    report = {"state": "TRAIN_FRAME_INSPECTED_ONLY", "source": str(a.frame), "fields": fields,
              "test_payloads_read": 0, "buffer_payloads_read": 0}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
