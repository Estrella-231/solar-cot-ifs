"""Audit source availability for CPP gaps in fixed Hunan train/val manifests.

Only reads manifest metadata and file names. Test data are never opened.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cpp-root", type=Path, required=True)
    p.add_argument("--hdf-root", type=Path, required=True)
    p.add_argument("--era-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    frames = defaultdict(set)
    with a.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] not in ("train", "val"):
                continue
            for rel in row["data_relpaths"].split("|"):
                t = Path(rel).stem
                if len(t) != 12 or not t.isdigit():
                    raise ValueError(t)
                frames[row["split"]].add(t)
    all_missing = []
    counts = defaultdict(Counter)
    for split in ("train", "val"):
        for t in sorted(frames[split]):
            cpp = a.cpp_root / t[:4] / t[:8] / f"FY4B_AGRI_{t}00.nc"
            if cpp.is_file():
                continue
            day = t[:8]
            raw_dir = a.hdf_root / t[:4] / day
            raw = list(raw_dir.glob(f"*_{t}00_*_4000M_*.HDF"))
            era = {
                "temperature": a.era_root / "profile/T" / t[:4] / f"era5-temperature-{day}.nc",
                "humidity": a.era_root / "profile/R" / t[:4] / f"era5-relative_humidity-{day}.nc",
                "skin_temperature": a.era_root / "single/SKT" / t[:4] / f"era5-skin_temperature-{day}.nc",
            }
            source_ok = len(raw) == 1 and raw[0].stat().st_size > 0
            era_ok = all(x.is_file() and x.stat().st_size > 0 for x in era.values())
            status = "ready" if source_ok and era_ok else "missing_raw" if not source_ok else "missing_era"
            counts[split][status] += 1
            all_missing.append({"split": split, "utc": t, "status": status,
                                "raw_hdf": str(raw[0]) if len(raw) == 1 else None,
                                "era_ok": era_ok})
    report = {"schema": "hunan_cpp_v2_missing_input_audit_v1", "test_used": False,
              "manifest_sha256": sha(a.manifest), "counts": {k: dict(v) for k, v in counts.items()},
              "items": all_missing}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "total": len(all_missing)}))


if __name__ == "__main__":
    main()
