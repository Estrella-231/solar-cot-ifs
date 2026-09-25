"""Pair a small raw CPP regional pilot with the original Hunan AGRI frames."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


STATIONS = {"sili": (163, 148), "zhujia": (73, 150)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ("data-root", "manifest", "cpp-pilot", "hunan-data-code", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite regional AGRI/CPP audit")
    spec = importlib.util.spec_from_file_location("regional_hunan_data", a.hunan_data_code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = json.loads((a.cpp_pilot / "manifest.json").read_text())
    if meta["test_used"] is not False or meta["manifest_sha256"] != sha(a.manifest):
        raise RuntimeError("CPP pilot/AGRI manifest contract drift")
    by_stem = {}
    with a.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in ("train", "val"):
                continue
            for rel in row["data_relpaths"].split("|"):
                key = (row["split"], Path(rel).stem)
                if key in by_stem and by_stem[key] != rel:
                    raise RuntimeError(f"conflicting AGRI relpath: {key}")
                by_stem[key] = rel
    report = {"state": "REGIONAL_AGRI_CPP_PAIR_PILOT", "test_used": False,
              "reference": "original raw CPP retrieval; no independent physical QA",
              "cpp_pilot_manifest_sha256": sha(a.cpp_pilot / "manifest.json"),
              "agri_manifest_sha256": sha(a.manifest), "splits": {}}
    for split in ("train", "val"):
        archive = a.cpp_pilot / meta["splits"][split]["archive"]
        if sha(archive) != meta["splits"][split]["archive_sha256"]:
            raise RuntimeError(f"{split} CPP archive drift")
        with np.load(archive, allow_pickle=False) as arr:
            utcs = [x.decode("ascii") for x in arr["utc"]]
            cot, cpp_valid = arr["cot"], arr["valid"]
        if cot.shape != (len(utcs), 256, 256) or cpp_valid.shape != cot.shape:
            raise RuntimeError(f"{split} CPP shape mismatch")
        rows = []
        for i, stem in enumerate(utcs):
            agri, agri_valid, geom = module.load_frame(a.data_root, by_stem[(split, stem)])
            if agri.shape != (13, 256, 256) or geom.shape != (3, 256, 256):
                raise RuntimeError("AGRI/geometry shape mismatch")
            image_valid = agri_valid.all(axis=0)
            day = geom[2] > 0.5
            paired = cpp_valid[i] & image_valid & day
            record = {"utc": stem, "valid_cpp_fraction": float(cpp_valid[i].mean()),
                      "valid_agri_fraction": float(image_valid.mean()),
                      "paired_day_fraction": float(paired.mean()), "stations": {}}
            for name, (cy, cx) in STATIONS.items():
                area = (slice(cy-32, cy+32), slice(cx-32, cx+32))
                mask = paired[area]
                local = cot[i][area]
                record["stations"][name] = {
                    "paired_day_fraction_64": float(mask.mean()),
                    "mean_cpp_cot_valid": float(local[mask].mean()) if mask.any() else None}
            rows.append(record)
        report["splits"][split] = {"frames": len(rows),
                                   "mean_paired_day_fraction": float(np.mean([r["paired_day_fraction"] for r in rows])),
                                   "records": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({s: {"frames": x["frames"],
                          "mean_paired_day_fraction": x["mean_paired_day_fraction"]}
                      for s, x in report["splits"].items()}))


if __name__ == "__main__":
    main()
