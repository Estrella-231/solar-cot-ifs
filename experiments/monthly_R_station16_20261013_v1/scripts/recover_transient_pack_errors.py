#!/usr/bin/env python3
"""Recover transient CPP read failures into a new, auditable train/val pack."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np

import pack_station16_cpp_agri as packer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(nc, raw, bits, geom, station_rc, glat, glon):
    cr, cc = station_rc
    sl = np.s_[cr - 8:cr + 8, cc - 8:cc + 8]
    lat_axis, lon_axis = packer.axes(nc)
    rr, ccols, _ = packer.cpp_index(lat_axis, lon_axis, glat[sl], glon[sl])
    rlo, rhi = int(rr.min()), int(rr.max()) + 1
    clo, chi = int(ccols.min()), int(ccols.max()) + 1
    cot = packer.arr(nc.variables["COT"], (slice(rlo, rhi), slice(clo, chi)))
    clpvar = nc.variables.get("CLP")
    if clpvar is None:
        raise KeyError("CPP product has no CLP")
    clp = packer.arr(clpvar, (slice(rlo, rhi), slice(clo, chi)))
    cot = cot[np.ix_(rr - rlo, ccols - clo)]
    clp = clp[np.ix_(rr - rlo, ccols - clo)]
    if np.diff(rr).mean() < 0:
        cot, clp = cot[::-1], clp[::-1]
    if np.diff(ccols).mean() < 0:
        cot, clp = cot[:, ::-1], clp[:, ::-1]

    agri = np.asarray(raw[np.asarray(packer.CHANNELS), sl[0], sl[1]], np.float32)
    valid_ch = np.stack([((bits[sl] & (1 << c)) != 0) for c in packer.CHANNELS]) & np.isfinite(agri)
    g = np.stack([x[sl] for x in geom]).astype(np.float32)
    input_ok = valid_ch.all(0) & np.isfinite(g).all(0)
    mask = (np.isfinite(cot) & (cot >= 0) & (cot <= 100) & np.isfinite(clp) &
            np.isin(clp, (0, 1, 2)) & input_ok & (g[2] > 0.5) & (g[0] > packer.COS80))
    if not mask.any():
        return None
    x = np.concatenate((agri, g), axis=0)
    x[:13, ~input_ok] = np.nan
    return x, cot, mask


def main():
    p = argparse.ArgumentParser()
    for name in ("base-pack", "recheck", "data-root", "grid-static", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    base, out = a.base_pack, a.output
    status = json.loads((base / "pack_status.json").read_text(encoding="utf-8"))
    recheck = json.loads(a.recheck.read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or status.get("test_payloads_read") != 0:
        raise RuntimeError("base train/validation pack is not complete or test gate failed")
    if recheck.get("state") != "CPP_SOURCE_ERROR_RECHECK_COMPLETE" or recheck.get("test_payloads_read") != 0:
        raise RuntimeError("CPP retry audit is incomplete or test gate failed")
    if recheck.get("candidate_sha256") != status.get("candidate_sha256"):
        raise RuntimeError("retry audit and base pack candidate hashes differ")
    recovered = [r for r in recheck.get("results", [])
                 if r.get("final_state") == "TRANSIENT_OR_INTERMITTENT_READ"
                 and r.get("split") == "train"
                 and all(x.get("state") == "READ_OK" for x in r.get("retries", []))]
    if len(recovered) != recheck.get("failed_frames") or not recovered:
        raise RuntimeError("not all first-pass failures were cleanly recovered")

    candidate_by_utc = {}
    candidate_path = base.parent.parent / "input" / "target_candidates_train_val.csv"
    with candidate_path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] not in ("train", "val"):
                raise RuntimeError("recovery refuses candidate rows outside train/validation")
            candidate_by_utc.setdefault(r["target_time_utc"], {})[r["station"]] = tuple(
                map(int, r["station_grid_rc"].split(",")))

    with (base / "rows.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0])
    by_time = {}
    for r in rows:
        if r["state"] == "SOURCE_READ_ERROR":
            by_time.setdefault(r["target_time_utc"], []).append(r)
    recover_times = {r["target_time_utc"]: r for r in recovered}
    if set(by_time) != set(recover_times):
        raise RuntimeError("source error rows do not match retry audit timestamps")
    if any(len(v) != 2 or {x["station"] for x in v} != set(packer.STATIONS) for v in by_time.values()):
        raise RuntimeError("each failed frame must have exactly two station error rows")
    out.mkdir(parents=True, exist_ok=False)
    with np.load(a.grid_static, allow_pickle=False) as z:
        glat, glon = np.asarray(z["lat"], np.float32), np.asarray(z["lon"], np.float32)
    staged_recovery = {}
    for utc, err in sorted(recover_times.items()):
        bjt = packer.bjt_stamp(utc)
        agri_path = Path(next(r["agri_path"] for r in by_time[utc]))
        aux_matches = sorted((a.data_root / "aux_256" / bjt[:6]).glob(bjt + "_*_aux.npz"))
        if len(aux_matches) != 1 or not agri_path.is_file():
            raise RuntimeError(f"missing AGRI/aux source during recovery: {utc}")
        raw = np.load(agri_path, mmap_mode="r", allow_pickle=False)
        if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
            raise RuntimeError(f"unexpected AGRI source shape/dtype: {utc} {raw.shape}/{raw.dtype}")
        with np.load(aux_matches[0], allow_pickle=False) as z:
            bits = z["agri_valid_bits"]
            geom = tuple(np.asarray(z[n], np.float32) for n in ("cosSOZ", "cosRAA", "day_mask"))
        station_rc = candidate_by_utc.get(utc, {})
        if set(station_rc) != set(packer.STATIONS):
            raise RuntimeError(f"candidate station mapping missing for {utc}")
        extracted = None
        retry_errors = []
        for attempt in range(1, 4):
            try:
                with netCDF4.Dataset(err["cpp_path"]) as nc:
                    extracted = {st: extract(nc, raw, bits, geom, station_rc[st], glat, glon)
                                 for st in packer.STATIONS}
                break
            except Exception as exc:
                retry_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if extracted is None:
            raise RuntimeError(f"CPP retry failed during recovery for {utc}: {retry_errors}")
        staged_recovery[utc] = (extracted, str(aux_matches[0]))
        del raw

    additional_examples = sum(result is not None
                              for extracted, _ in staged_recovery.values()
                              for result in extracted.values())
    old_x = np.load(base / "x_raw.npy", mmap_mode="r", allow_pickle=False)
    old_y = np.load(base / "cot.npy", mmap_mode="r", allow_pickle=False)
    old_m = np.load(base / "mask.npy", mmap_mode="r", allow_pickle=False)
    try:
        n0 = len(old_x)
        xmap = np.lib.format.open_memmap(out / "x_raw.npy", mode="w+", dtype=np.float32,
                                          shape=(n0 + additional_examples, 16, 16, 16))
        ymap = np.lib.format.open_memmap(out / "cot.npy", mode="w+", dtype=np.float32,
                                          shape=(n0 + additional_examples, 1, 16, 16))
        mmap = np.lib.format.open_memmap(out / "mask.npy", mode="w+", dtype=np.bool_,
                                          shape=(n0 + additional_examples, 1, 16, 16))
        chunk = 256
        for lo in range(0, n0, chunk):
            hi = min(n0, lo + chunk)
            xmap[lo:hi] = old_x[lo:hi]
            ymap[lo:hi] = old_y[lo:hi]
            mmap[lo:hi] = old_m[lo:hi]
        current_index = n0
        no_valid_station_rows = 0
        for utc, (extracted, aux_path) in sorted(staged_recovery.items()):
            for station in packer.STATIONS:
                result = extracted[station]
                if result is None:
                    rec = next(r for r in by_time[utc] if r["station"] == station)
                    rec.update(state="NO_VALID_CPP_PIXELS", array_index="", supervised_pixels="0",
                               valid_zero_pixels="0", thick_ge30_pixels="0", aux_path=aux_path)
                    no_valid_station_rows += 1
                    continue
                x, cot, mask = result
                xmap[current_index] = x
                ymap[current_index, 0] = cot
                mmap[current_index, 0] = mask
                rec = next(r for r in by_time[utc] if r["station"] == station)
                rec.update(state="SUPERVISED", array_index=str(current_index),
                           supervised_pixels=str(int(mask.sum())),
                           valid_zero_pixels=str(int(((cot == 0) & mask).sum())),
                           thick_ge30_pixels=str(int(((cot >= 30) & mask).sum())),
                           aux_path=aux_path)
                current_index += 1
        xmap.flush(); ymap.flush(); mmap.flush()
        del xmap, ymap, mmap
    finally:
        del old_x, old_y, old_m

    # Copy audit metadata while preserving the original failed-attempt evidence separately.
    for name in ("source_coverage.csv",):
        shutil.copy2(base / name, out / name)
    shutil.copy2(base / "pack_audit_external_preflight.json", out / "base_pack_audit.json")
    shutil.copy2(base / "source_errors.csv", out / "source_errors_first_pass.csv")
    shutil.copy2(a.recheck, out / "source_error_recheck.json")
    with (out / "source_errors.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=("split", "target_time_utc", "bjt", "cpp_path", "agri_path", "error")).writeheader()
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    final_status = dict(status)
    final_status["packed_examples"] = current_index
    final_status["by_split"] = {s: sum(r["state"] == "SUPERVISED" and r["split"] == s for r in rows)
                                 for s in ("train", "val")}
    state_counts = {}
    for split in ("train", "val"):
        state_counts[split] = {}
        for r in rows:
            if r["split"] == split:
                state_counts[split][r["state"]] = state_counts[split].get(r["state"], 0) + 1
    final_status["sample_states_by_split"] = state_counts
    final_status["source_read_error_frames"] = 0
    final_status["first_pass_source_read_error_frames"] = len(recovered)
    final_status["source_retry_recovered_frames"] = len(recovered)
    final_status["recovered_from_pack_status_sha256"] = sha(base / "pack_status.json")
    final_status["source_error_recheck_sha256"] = sha(a.recheck)
    final_status["state"] = "COMPLETE_MONTHLY_R_STATION16_PACK"
    (out / "pack_status.json").write_text(json.dumps(final_status, indent=2) + "\n", encoding="utf-8")
    report = {"state": "RECOVERED_TRANSIENT_CPP_ERRORS",
              "test_payloads_read": 0, "recovered_frames": len(recovered),
              "appended_station_examples": current_index - n0,
              "recovered_no_valid_station_rows": no_valid_station_rows,
              "base_examples": n0, "final_examples": current_index,
              "retry_json_sha256": sha(a.recheck)}
    (out / "recovery_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
