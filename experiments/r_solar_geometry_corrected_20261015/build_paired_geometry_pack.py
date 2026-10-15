#!/usr/bin/env python3
"""Create old-vs-corrected R inputs on identical CPP pixels, train/val only."""
import argparse, csv, hashlib, json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_geometry_corrected import compute_solar_angles

EXPECTED_CANDIDATE_SHA = "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807"
COS80 = float(np.cos(np.deg2rad(80.0)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ("pack", "candidates", "grid-static", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.candidates) != EXPECTED_CANDIDATE_SHA:
        raise ValueError("candidate manifest is not the frozen monthly train/val split")
    status = json.loads((a.pack / "pack_status.json").read_text())
    if status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or status.get("test_payloads_read") != 0:
        raise ValueError("source R pack is incomplete or has unexpected test access")
    if status.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA:
        raise ValueError("source pack split hash mismatch")

    candidates = {}
    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] not in ("train", "val"):
                raise ValueError("candidate manifest contains a non train/val row")
            key = (r["target_time_utc"], r["station"])
            candidates[key] = tuple(map(int, r["station_grid_rc"].split(",")))
    with np.load(a.grid_static, allow_pickle=False) as z:
        lat, lon, saa = (np.asarray(z[k], np.float32) for k in ("lat", "lon", "saa"))
    if any(x.shape != (256, 256) for x in (lat, lon, saa)):
        raise ValueError("expected Hunan 256x256 grid geometry")

    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8")))
    sup = [r for r in rows if r["state"] == "SUPERVISED"]
    x0 = np.load(a.pack / "x_raw.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    m0 = np.load(a.pack / "mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    if x0.shape != (len(sup), 16, 16, 16) or cot.shape != (len(sup), 1, 16, 16) or m0.shape != cot.shape:
        raise ValueError("source array/rows shape mismatch")

    # Construct all corrected angles from the frozen station grid and nominal UTC.
    # Keep only pixels eligible under BOTH old and corrected day/zenith masks, so
    # the old-formula and corrected-formula retrains use exactly the same labels.
    x_old = np.asarray(x0, np.float32).copy()
    x_new = x_old.copy()
    common = np.zeros_like(m0)
    geometry_rows = []
    for i, r in enumerate(sup):
        key = (r["target_time_utc"], r["station"])
        if key not in candidates:
            raise ValueError(f"supervised pack row absent from frozen manifest: {key}")
        cr, cc = candidates[key]
        sl = np.s_[cr - 8:cr + 8, cc - 8:cc + 8]
        dt = datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S")
        soz, soa = compute_solar_angles(lat[sl], lon[sl], dt)
        physical_night = soz >= 90.0
        soa = np.where(physical_night, 0.0, soa)
        cs = np.cos(np.deg2rad(soz)).astype(np.float32)
        cr = np.cos(np.deg2rad(soa - saa[sl])).astype(np.float32)
        day = (soz < 85.0).astype(np.float32)
        x_new[i, 13] = cs
        x_new[i, 14] = cr
        x_new[i, 15] = day
        corrected_eligible = (day > 0.5) & (cs > COS80)
        common[i, 0] = m0[i, 0] & corrected_eligible
        geometry_rows.append(dict(array_index=i, split=r["split"], station=r["station"],
                                  target_time_utc=r["target_time_utc"],
                                  old_pixels=int(m0[i].sum()), corrected_eligible_pixels=int(corrected_eligible.sum()),
                                  common_pixels=int(common[i].sum())))
    if int(common.sum()) == 0:
        raise ValueError("empty common CPP support")

    a.output.mkdir(parents=True)
    np.save(a.output / "x_old_geometry.npy", x_old, allow_pickle=False)
    np.save(a.output / "x_corrected_geometry.npy", x_new, allow_pickle=False)
    np.save(a.output / "cot.npy", np.asarray(cot, np.float32), allow_pickle=False)
    np.save(a.output / "common_mask.npy", common, allow_pickle=False)
    with (a.output / "rows_supervised.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sup[0])); w.writeheader(); w.writerows(sup)
    with (a.output / "geometry_support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(geometry_rows[0])); w.writeheader(); w.writerows(geometry_rows)

    counts = {}
    for split in ("train", "val"):
        idx = [i for i, r in enumerate(sup) if r["split"] == split]
        counts[split] = dict(rows=len(idx), old_pixels=int(m0[idx].sum()), common_pixels=int(common[idx].sum()),
                             excluded_by_common_geometry=int(m0[idx].sum() - common[idx].sum()))
        for station in ("sili", "zhujia"):
            si = [i for i in idx if sup[i]["station"] == station]
            counts[split][station] = dict(rows=len(si), old_pixels=int(m0[si].sum()), common_pixels=int(common[si].sum()))
    hashes = {n: sha(a.output / n) for n in ("x_old_geometry.npy", "x_corrected_geometry.npy", "cot.npy",
                                               "common_mask.npy", "rows_supervised.csv", "geometry_support.csv")}
    report = dict(state="PASS_PAIRED_GEOMETRY_COMMON_SUPPORT", split_id=status["split_id"],
                  source_pack_status_sha256=sha(a.pack / "pack_status.json"),
                  source_pack_arrays={n: sha(a.pack / n) for n in ("x_raw.npy", "cot.npy", "mask.npy", "rows.csv")},
                  candidate_sha256=sha(a.candidates), grid_static_sha256=sha(a.grid_static),
                  corrected_formula="equation_of_time=4*rad2deg(trigonometric_expression)",
                  corrected_geometry_source_sha256=sha(Path(__file__).with_name("solar_geometry_corrected.py")),
                  common_support="source CPP/input-valid mask AND old-R mask AND corrected day_mask>0.5 AND corrected cosSOZ>cos80",
                  counts=counts, trainval_rows=len(sup), test_payloads_read=0, hashes=hashes)
    (a.output / "pack_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
