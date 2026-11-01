#!/usr/bin/env python3
"""Build train/val-only 16x16 AGRI/CPP pairs for a fresh monthly R."""
import argparse, csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("sili", "zhujia")
COS80 = float(np.cos(np.deg2rad(80.0)))
EXPECTED_CANDIDATE_SHA = "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807"
EXPECTED_CPP_INVENTORY_SHA = "0a303ec6965be360689f3891e96b5e4085ad3e6550b2db6504a8b2ac35565dd2"
ROW_FIELDS = ("split", "station", "target_time_utc", "bjt", "state", "array_index",
              "supervised_pixels", "valid_zero_pixels", "thick_ge30_pixels",
              "agri_path", "cpp_path", "aux_path")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arr(var, key=Ellipsis):
    return np.asarray(np.ma.filled(var[key], np.nan), dtype=np.float32)


def axes(nc):
    lat, lon = nc.variables["LAT"], nc.variables["LON"]
    la = arr(lat, (slice(None), 0)) if len(lat.shape) == 2 else arr(lat)
    lo = arr(lon, (0, slice(None))) if len(lon.shape) == 2 else arr(lon)
    if la.ndim != 1 or lo.ndim != 1 or not (np.all(np.diff(la) > 0) or np.all(np.diff(la) < 0)) or not (np.all(np.diff(lo) > 0) or np.all(np.diff(lo) < 0)):
        raise ValueError("CPP LAT/LON must provide monotonic rectilinear axes")
    return la, lo


def cpp_index(lat_axis, lon_axis, lat2d, lon2d):
    # Map on the CPP rectilinear axes and verify every curvilinear AGRI coordinate.
    lat_row = lat2d[:, lat2d.shape[1] // 2]
    lon_col = lon2d[lon2d.shape[0] // 2, :]
    rr = np.abs(lat_axis[:, None] - lat_row[None, :]).argmin(0)
    cc = np.abs(lon_axis[:, None] - lon_col[None, :]).argmin(0)
    err = max(float(np.max(np.abs(lat2d - lat_axis[rr, None]))), float(np.max(np.abs(lon2d - lon_axis[None, cc]))))
    if err > 1e-4 or len(np.unique(rr)) != 16 or len(np.unique(cc)) != 16:
        raise ValueError(f"CPP grid mapping failed: max coordinate error={err}")
    return rr, cc, err


def bjt_stamp(utc):
    return (datetime.strptime(utc, "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y%m%d%H%M")


def main():
    p = argparse.ArgumentParser()
    for name in ("candidates", "cpp-inventory", "data-root", "grid-static", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke limit over train CPP frame keys; 0 means all train/val")
    p.add_argument("--smoke-utc", action="append", default=[], help="explicit train/val UTC timestamp for a small real-payload smoke")
    p.add_argument("--smoke-utc-file", type=Path, help="text file with one explicit train/val UTC timestamp per line")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")
    if sha(a.candidates) != EXPECTED_CANDIDATE_SHA:
        raise ValueError("candidate manifest hash differs from frozen audited train/val table")
    if sha(a.cpp_inventory) != EXPECTED_CPP_INVENTORY_SHA:
        raise ValueError("CPP inventory hash differs from audited train/val support table")

    with np.load(a.grid_static, allow_pickle=False) as z:
        glat, glon = np.asarray(z["lat"], np.float32), np.asarray(z["lon"], np.float32)
    if glat.shape != (256, 256) or glon.shape != (256, 256):
        raise ValueError(f"expected Hunan 256x256 grid, got {glat.shape}/{glon.shape}")

    candidate_by_utc = {}
    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] not in ("train", "val"):
                raise ValueError("candidate file must already be train/val only")
            utc = row["target_time_utc"]
            item = candidate_by_utc.setdefault(utc, {"split": row["split"], "rel": row["agri_frame_relpath"], "stations": {}})
            if item["split"] != row["split"] or item["rel"] != row["agri_frame_relpath"]:
                raise ValueError(f"split/path mismatch at {utc}")
            item["stations"][row["station"]] = tuple(map(int, row["station_grid_rc"].split(",")))
    if any(set(v["stations"]) != set(STATIONS) for v in candidate_by_utc.values()):
        raise ValueError("every frame must have exactly both Hunan stations")

    cpp_by_utc = {}
    with a.cpp_inventory.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] in ("train", "val") and row["cpp_present"] in ("1", "True", "true"):
                utc = row["target_time_utc"]
                prior = cpp_by_utc.setdefault(utc, row["cpp_path"])
                if prior != row["cpp_path"]:
                    raise ValueError(f"duplicate CPP timestamp maps to multiple files: {utc}")
    selected = [u for u, r in sorted(candidate_by_utc.items()) if u in cpp_by_utc]
    smoke_times = list(a.smoke_utc)
    if a.smoke_utc_file:
        smoke_times.extend(x.strip() for x in a.smoke_utc_file.read_text(encoding="utf-8").splitlines() if x.strip())
    if smoke_times:
        if a.limit: raise ValueError("use either --smoke-utc or --limit")
        if len(smoke_times) != len(set(smoke_times)): raise ValueError("duplicate smoke UTC timestamps")
        if any(u not in selected for u in smoke_times): raise ValueError("smoke UTC must have both candidate and CPP support")
        if {candidate_by_utc[u]["split"] for u in smoke_times} != {"train", "val"}:
            raise ValueError("smoke UTC set must include at least one train and one val time")
        selected = sorted(smoke_times)
    elif a.limit:
        train = [u for u in selected if candidate_by_utc[u]["split"] == "train"]
        val = [u for u in selected if candidate_by_utc[u]["split"] == "val"]
        selected = train[:a.limit] + val[:min(a.limit, len(val))]
    if not selected:
        raise ValueError("no train/val timestamps have CPP support")

    a.output.mkdir(parents=True)
    coverage_rows = [dict(split=v["split"], target_time_utc=u, bjt=bjt_stamp(u), cpp_present=int(u in cpp_by_utc), cpp_path=cpp_by_utc.get(u, ""))
                     for u, v in sorted(candidate_by_utc.items())]
    with (a.output / "source_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0])); w.writeheader(); w.writerows(coverage_rows)
    # Upper-bound allocation: at most two station examples per CPP timestamp.
    nmax = 2 * len(selected)
    xmap = np.lib.format.open_memmap(a.output / "x_raw.npy", mode="w+", dtype=np.float32, shape=(nmax, 16, 16, 16))
    ymap = np.lib.format.open_memmap(a.output / "cot.npy", mode="w+", dtype=np.float32, shape=(nmax, 1, 16, 16))
    mmap = np.lib.format.open_memmap(a.output / "mask.npy", mode="w+", dtype=np.bool_, shape=(nmax, 1, 16, 16))
    rows, source_errors, count = [], [], 0
    mapping_error_max = 0.0
    for k, utc in enumerate(selected, 1):
        frame = candidate_by_utc[utc]
        bjt = bjt_stamp(utc)
        raw_path = a.data_root / "data" / frame["rel"]
        if utc not in cpp_by_utc:
            for station in STATIONS:
                rows.append(dict(split=frame["split"], station=station, target_time_utc=utc,
                                 bjt=bjt, state="CPP_UNAVAILABLE", supervised_pixels=0,
                                 agri_path=str(raw_path), cpp_path=""))
            continue
        aux_matches = sorted((a.data_root / "aux_256" / bjt[:6]).glob(bjt + "_*_aux.npz"))
        if len(aux_matches) != 1 or not raw_path.is_file():
            for station in STATIONS:
                rows.append(dict(split=frame["split"], station=station, target_time_utc=utc,
                                 bjt=bjt, state="INPUT_UNAVAILABLE", supervised_pixels=0,
                                 agri_path=str(raw_path), cpp_path=cpp_by_utc[utc]))
            continue
        staged = []
        try:
            raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
            if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
                raise ValueError(f"unexpected AGRI cache {raw_path}: {raw.shape}/{raw.dtype}")
            with np.load(aux_matches[0], allow_pickle=False) as z:
                bits = z["agri_valid_bits"]
                cossoz, cosraa, day = (np.asarray(z[n], np.float32) for n in ("cosSOZ", "cosRAA", "day_mask"))
            with netCDF4.Dataset(cpp_by_utc[utc]) as nc:
                lat_axis, lon_axis = axes(nc)
                for station in STATIONS:
                    cr, cc = frame["stations"][station]
                    r0, c0 = cr - 8, cc - 8
                    if r0 < 0 or c0 < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                        raise ValueError(f"station crop outside grid: {utc} {station} {(cr, cc)}")
                    sl = np.s_[r0:r0 + 16, c0:c0 + 16]
                    rr, ccols, err = cpp_index(lat_axis, lon_axis, glat[sl], glon[sl])
                    mapping_error_max = max(mapping_error_max, err)
                    rlo, rhi, clo, chi = int(rr.min()), int(rr.max()) + 1, int(ccols.min()), int(ccols.max()) + 1
                    cot = arr(nc.variables["COT"], (slice(rlo, rhi), slice(clo, chi)))[np.ix_(rr - rlo, ccols - clo)]
                    clpvar = nc.variables.get("CLP")
                    if clpvar is None:
                        raise ValueError(f"CPP product has no CLP: {cpp_by_utc[utc]}")
                    clp = arr(clpvar, (slice(rlo, rhi), slice(clo, chi)))[np.ix_(rr - rlo, ccols - clo)]
                    if np.diff(rr).mean() < 0:
                        cot, clp = cot[::-1], clp[::-1]
                    if np.diff(ccols).mean() < 0:
                        cot, clp = cot[:, ::-1], clp[:, ::-1]

                    agri = np.asarray(raw[np.asarray(CHANNELS), sl[0], sl[1]], np.float32)
                    valid_ch = np.stack([((bits[sl] & (1 << c)) != 0) for c in CHANNELS]) & np.isfinite(agri)
                    geom = np.stack((cossoz[sl], cosraa[sl], day[sl]))
                    input_ok = valid_ch.all(0) & np.isfinite(geom).all(0)
                    mask = (np.isfinite(cot) & (cot >= 0) & (cot <= 100) & np.isfinite(clp) &
                            np.isin(clp, (0, 1, 2)) & input_ok & (geom[2] > 0.5) & (geom[0] > COS80))
                    if mask.any():
                        x = np.concatenate((agri, geom), axis=0)
                        x[:13, ~input_ok] = np.nan
                        staged.append((station, x, cot, mask, int(((cot == 0) & mask).sum()),
                                       int(((cot >= 30) & mask).sum())))
                    else:
                        staged.append((station, None, cot, mask, 0, 0))
            # Commit both stations only after the full timestamp has been read and mapped.
            for station, x, cot, mask, zero_count, thick_count in staged:
                if x is not None:
                    xmap[count] = x
                    ymap[count, 0] = cot
                    mmap[count, 0] = mask
                    state, idx = "SUPERVISED", count
                    count += 1
                else:
                    state, idx = "NO_VALID_CPP_PIXELS", ""
                rows.append(dict(split=frame["split"], station=station, target_time_utc=utc,
                                 bjt=bjt, state=state, array_index=idx,
                                 supervised_pixels=int(mask.sum()), valid_zero_pixels=zero_count,
                                 thick_ge30_pixels=thick_count,
                                 agri_path=str(raw_path), cpp_path=cpp_by_utc[utc], aux_path=str(aux_matches[0])))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            source_errors.append(dict(split=frame["split"], target_time_utc=utc, bjt=bjt,
                                      cpp_path=cpp_by_utc[utc], agri_path=str(raw_path), error=detail))
            for station in STATIONS:
                rows.append(dict(split=frame["split"], station=station, target_time_utc=utc,
                                 bjt=bjt, state="SOURCE_READ_ERROR", supervised_pixels=0,
                                 agri_path=str(raw_path), cpp_path=cpp_by_utc[utc], aux_path=str(aux_matches[0])))
        if k % 500 == 0:
            xmap.flush(); ymap.flush(); mmap.flush()
            print(f"PROGRESS {k}/{len(selected)} samples={count}", flush=True)

    # Record CPP gaps explicitly in the full package without allocating or
    # opening any CPP/test payload for those keys.
    if not smoke_times and not a.limit:
        for utc, frame in sorted(candidate_by_utc.items()):
            if utc in cpp_by_utc:
                continue
            for station in STATIONS:
                rows.append(dict(split=frame["split"], station=station, target_time_utc=utc,
                                 bjt=bjt_stamp(utc), state="CPP_UNAVAILABLE", supervised_pixels=0,
                                 agri_path=str(a.data_root / "data" / frame["rel"]), cpp_path="", aux_path=""))

    xmap.flush(); ymap.flush(); mmap.flush()
    # Trim upper-bound memmaps to exact row count without touching test data.
    for name, arr0 in (("x_raw.npy", xmap), ("cot.npy", ymap), ("mask.npy", mmap)):
        tmp = a.output / (name + ".trim.npy")
        np.save(tmp, np.asarray(arr0[:count]).copy(), allow_pickle=False)
        arr0.flush()
    del xmap, ymap, mmap
    for name in ("x_raw.npy", "cot.npy", "mask.npy"):
        (a.output / name).unlink()
        (a.output / (name + ".trim.npy")).replace(a.output / name)
    with (a.output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS); w.writeheader(); w.writerows(rows)
    if source_errors:
        with (a.output / "source_errors.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(source_errors[0])); w.writeheader(); w.writerows(source_errors)
    state_counts = {}
    for split in ("train", "val"):
        state_counts[split] = {state: sum(r.get("split") == split and r.get("state") == state for r in rows)
                               for state in ("SUPERVISED", "NO_VALID_CPP_PIXELS", "INPUT_UNAVAILABLE",
                                             "CPP_UNAVAILABLE", "SOURCE_READ_ERROR")}
    report = dict(state="COMPLETE_MONTHLY_R_STATION16_PACK", split_id="hunan_monthly_block_2d_buffer1d_seed42_v1",
                  test_payloads_read=0, candidate_sha256=sha(a.candidates), cpp_inventory_sha256=sha(a.cpp_inventory),
                  grid_static_sha256=sha(a.grid_static), cpp_grid_mapping_max_error_deg=mapping_error_max,
                  candidate_unique_frames=len(candidate_by_utc), selected_cpp_frames=len(selected),
                  cpp_missing_frames=sum(utc not in cpp_by_utc for utc in candidate_by_utc),
                  packed_examples=int(count), by_split={s: int(sum(r.get("split") == s and r.get("state") == "SUPERVISED" for r in rows)) for s in ("train", "val")},
                  sample_states_by_split=state_counts,
                  source_read_error_frames=len(source_errors),
                  mask="finite COT 0..100 AND CLP in {0,1,2} AND all13 AGRI channels valid AND day_mask>0.5 AND cosSOZ>cos80",
                  valid_zero_cot_retained=True, missing_cpp_is_not_clear_sky=True)
    (a.output / "pack_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
