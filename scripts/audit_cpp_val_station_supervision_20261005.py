"""Audit validation CPP supervision at both station 5x5 neighborhoods only.

Reads only val CPP payload windows and target-time auxiliary masks/geometry.
No test split or full AGRI image is read.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = {"sili": (163, 148), "zhujia": (73, 150)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--aux-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--known-read-stall", action="append", default=[],
                    help="UTC timestamp whose prior NetCDF read stalled; record invalid and skip")
    a = ap.parse_args()
    with np.load(a.grid, allow_pickle=False) as g:
        glat, glon = np.asarray(g["lat"]), np.asarray(g["lon"])
    rows_all = cols_all = None
    output_rows = []
    with a.inventory.open(newline="", encoding="utf-8") as f:
        products = [r for r in csv.DictReader(f) if r["split"] == "val" and r["cpp_present"] == "1"]
    # The CPP grid is fixed for this product bank. Read coordinate axes once,
    # then reuse the verified mapping for all timestamps instead of rereading
    # multi-megabyte coordinate fields for every NetCDF file.
    with netCDF4.Dataset(products[0]["cpp_path"]) as nc:
        lat = np.asarray(np.ma.filled(nc.variables["LAT"][:], np.nan))
        lon = np.asarray(np.ma.filled(nc.variables["LON"][:], np.nan))
        rows_all = np.abs(lat[:, None] - glat[:, 0][None, :]).argmin(0)
        cols_all = np.abs(lon[:, None] - glon[0, :][None, :]).argmin(0)
        err = max(float(np.max(np.abs(glat - lat[rows_all, None]))),
                  float(np.max(np.abs(glon - lon[None, cols_all]))))
        if err > 1e-4:
            raise RuntimeError(f"grid mismatch: {err}")
    stalled = set(a.known_read_stall)
    for k, rec in enumerate(products, 1):
        bjt = rec["bjt"]
        aux_dir = a.aux_root / bjt[:6]
        matches = list(aux_dir.glob(bjt + "_*_aux.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one target auxiliary for {bjt}, found {matches}")
        with np.load(matches[0], allow_pickle=False) as aux:
            bits = aux["agri_valid_bits"]
            channel_valid = np.stack([(bits & (1 << c)) != 0 for c in CHANNELS], axis=0)
            geom = {name: np.asarray(aux[name]) for name in ("cosSOZ", "day_mask")}
        station_aux = {}
        for name, (r, c) in STATIONS.items():
            sl = np.s_[r-2:r+3, c-2:c+3]
            agri_ok = channel_valid[:, sl[0], sl[1]].all(0)
            solar_ok = (geom["day_mask"][sl] > .5) & (geom["cosSOZ"][sl] > np.cos(np.deg2rad(80)))
            station_aux[name] = (sl, agri_ok & solar_ok)
        if rec["utc"] in stalled:
            for name in STATIONS:
                output_rows.append({"station": name, "utc": rec["utc"], "bjt": bjt,
                                    "month_bjt": rec["month_bjt"], "season_bjt": rec["season_bjt"],
                                    "cpp_numeric_pixels": 0, "day_agri_phase_supervised_pixels": 0,
                                    "supervision_any": 0, "zero_supervised": 0,
                                    "thick_ge30_supervised": 0,
                                    "quality_error": "prior scan stalled in NFS NetCDF read; excluded conservatively",
                                    "aux_path": str(matches[0]), "cpp_path": rec["cpp_path"]})
            print(f"CPP_QUALITY_READ_STALL utc={rec['utc']} excluded=1", flush=True)
            continue
        with netCDF4.Dataset(rec["cpp_path"]) as nc:
            # Read only the bounding box of the two local station neighborhoods.
            requested_r, requested_c = [], []
            for r, c in STATIONS.values():
                requested_r.extend(rows_all[r-2:r+3].tolist())
                requested_c.extend(cols_all[c-2:c+3].tolist())
            r0, r1 = min(requested_r), max(requested_r)+1
            c0, c1 = min(requested_c), max(requested_c)+1
            clp_var = nc.variables.get("CLP")
            if clp_var is None:
                raise RuntimeError("CPP has no CLP cloud-phase variable")
            try:
                cot_raw = nc.variables["COT"][r0:r1, c0:c1]
                cot = np.asarray(np.ma.filled(cot_raw, np.nan), np.float32)
                clp_raw = clp_var[r0:r1, c0:c1]
                clp = np.asarray(np.ma.filled(clp_raw, np.nan), np.float32)
            except Exception as exc:
                for name in STATIONS:
                    output_rows.append({"station": name, "utc": rec["utc"], "bjt": bjt,
                                        "month_bjt": rec["month_bjt"], "season_bjt": rec["season_bjt"],
                                        "cpp_numeric_pixels": 0, "day_agri_phase_supervised_pixels": 0,
                                        "supervision_any": 0, "zero_supervised": 0,
                                        "thick_ge30_supervised": 0, "quality_error": str(exc),
                                        "aux_path": str(matches[0]), "cpp_path": rec["cpp_path"]})
                print(f"CPP_QUALITY_READ_ERROR utc={rec['utc']} path={rec['cpp_path']} error={exc}", flush=True)
                continue
            # Source latitude/longitude axes are increasing in this product;
            # normalize each small patch by the mapped row/column direction.
            for name, (r, c) in STATIONS.items():
                rr = rows_all[r-2:r+3] - r0
                cc = cols_all[c-2:c+3] - c0
                patch = cot[np.ix_(rr, cc)]
                phase = clp[np.ix_(rr, cc)]
                if np.diff(rows_all[r-2:r+3]).mean() < 0:
                    patch, phase = patch[::-1], phase[::-1]
                if np.diff(cols_all[c-2:c+3]).mean() < 0:
                    patch, phase = patch[:, ::-1], phase[:, ::-1]
                valid = np.isfinite(patch) & (patch >= 0) & (patch <= 100)
                phase_ok = np.isfinite(phase) & np.isin(phase, (0, 1, 2))
                eligible = valid & phase_ok & station_aux[name][1]
                output_rows.append({"station": name, "utc": rec["utc"], "bjt": bjt,
                                    "month_bjt": rec["month_bjt"], "season_bjt": rec["season_bjt"],
                                    "cpp_numeric_pixels": int(valid.sum()),
                                    "day_agri_phase_supervised_pixels": int(eligible.sum()),
                                    "supervision_any": int(eligible.any()),
                                    "zero_supervised": int(((patch == 0) & eligible).sum()),
                                    "thick_ge30_supervised": int(((patch >= 30) & eligible).sum()),
                                    "quality_error": "",
                                    "aux_path": str(matches[0]), "cpp_path": rec["cpp_path"]})
        if k % 100 == 0:
            print(f"processed_val_products={k}/{len(products)}", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        w.writeheader()
        w.writerows(output_rows)
    summary = defaultdict(lambda: {"supported_products": 0, "products": 0,
                                   "effective_pixels": 0, "dates_with_any": set(),
                                   "zero_pixels": 0, "thick_ge30_pixels": 0})
    for row in output_rows:
        for stratum in (row["season_bjt"], row["month_bjt"]):
            s = summary[(row["station"], stratum)]
            s["products"] += 1
            s["supported_products"] += row["supervision_any"]
            s["effective_pixels"] += row["day_agri_phase_supervised_pixels"]
            s["zero_pixels"] += row["zero_supervised"]
            s["thick_ge30_pixels"] += row["thick_ge30_supervised"]
            if row["supervision_any"]:
                s["dates_with_any"].add(row["bjt"][:8])
    summary_json = {}
    for key, v in summary.items():
        summary_json["/".join(key)] = {**{k: x for k, x in v.items() if k != "dates_with_any"},
                                        "unique_BJT_days": len(v["dates_with_any"])}
    summary_path = a.output.with_name("cpp_val_station_supervision_summary.json")
    read_failures = sorted({(x["utc"], x["quality_error"], x["cpp_path"])
                            for x in output_rows if x.get("quality_error")})
    summary_path.write_text(json.dumps({"state": "VAL_STATION_5X5_MASK_AUDIT",
                                       "test_used": False,
                                       "method": "finite COT 0..100 AND CLP in {0,1,2} AND all13 AGRI valid bits AND day_mask>0.5 AND cosSOZ>cos80",
                                       "products": len(products), "quality_read_failure_count": len(read_failures),
                                       "quality_read_failures": [{"utc": x[0], "error": x[1], "path": x[2]}
                                                                  for x in read_failures],
                                       "station_month_season": summary_json},
                                      indent=2) + "\n", encoding="utf-8")
    print(f"products={len(products)} rows={len(output_rows)} output={a.output}")
    print(f"summary={summary_path}")
    for key, val in sorted(summary_json.items()):
        if key.endswith("/DJF") or key.endswith("/MAM") or key.endswith("/JJA") or key.endswith("/SON"):
            print(f"{key}: {val}", flush=True)


if __name__ == "__main__":
    main()
