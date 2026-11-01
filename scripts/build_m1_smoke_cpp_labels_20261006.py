"""Export only eight train-window future CPP labels and strict masks for M1."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--cpp-root", type=Path, required=True)
    ap.add_argument("--aux-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.grid, allow_pickle=False) as grid:
        glat, glon = np.asarray(grid["lat"]), np.asarray(grid["lon"])
    with a.windows.open(newline="", encoding="utf-8-sig") as f:
        windows = list(csv.DictReader(f))
    cot_rows, mask_rows, records = [], [], []
    rows_all = cols_all = None
    cot_varnames = None
    for win in windows:
        win_cot, win_mask, win_rec = [], [], []
        for stamp in win["future_utc"].split("|"):
            cpp_path = a.cpp_root / stamp[:4] / stamp[:8] / f"FY4B_AGRI_{stamp}00.nc"
            if not cpp_path.is_file():
                raise FileNotFoundError(cpp_path)
            t_bjt = datetime.strptime(stamp, "%Y%m%d%H%M") + timedelta(hours=8)
            aux_dir = a.aux_root / t_bjt.strftime("%Y%m")
            matches = list(aux_dir.glob(t_bjt.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
            if len(matches) != 1:
                raise RuntimeError(f"expected one target aux file at {t_bjt}: {matches}")
            with np.load(matches[0], allow_pickle=False) as aux:
                bits = aux["agri_valid_bits"]
                agri_valid = np.stack([(bits & (1 << c)) != 0 for c in CHANNELS], axis=0).all(0)
                cossoz, day = np.asarray(aux["cosSOZ"]), np.asarray(aux["day_mask"])
            read_error = ""
            try:
                with netCDF4.Dataset(cpp_path) as nc:
                    if rows_all is None:
                        lat = np.asarray(np.ma.filled(nc.variables["LAT"][:], np.nan))
                        lon = np.asarray(np.ma.filled(nc.variables["LON"][:], np.nan))
                        rows_all = np.abs(lat[:, None] - glat[:, 0][None, :]).argmin(0)
                        cols_all = np.abs(lon[:, None] - glon[0, :][None, :]).argmin(0)
                        lat_err = float(np.max(np.abs(glat - lat[rows_all, None])))
                        lon_err = float(np.max(np.abs(glon - lon[None, cols_all])))
                        if max(lat_err, lon_err) > 1e-4:
                            raise RuntimeError(f"CPP/grid coordinate mismatch: {lat_err},{lon_err}")
                    if nc.variables.get("CLP") is None:
                        raise RuntimeError(f"no CLP phase variable: {cpp_path}")
                    requested_r, requested_c = rows_all.tolist(), cols_all.tolist()
                    r0, r1, c0, c1 = min(requested_r), max(requested_r)+1, min(requested_c), max(requested_c)+1
                    cot = np.asarray(np.ma.filled(nc.variables["COT"][r0:r1, c0:c1], np.nan), np.float32)
                    clp = np.asarray(np.ma.filled(nc.variables["CLP"][r0:r1, c0:c1], np.nan), np.float32)
                    if rows_all[0] > rows_all[-1]:
                        cot, clp = cot[::-1], clp[::-1]
                    if cols_all[0] > cols_all[-1]:
                        cot, clp = cot[:, ::-1], clp[:, ::-1]
            except Exception as exc:
                read_error = f"{type(exc).__name__}: {exc}"
                cot = np.zeros((256, 256), np.float32)
                clp = np.full((256, 256), np.nan, np.float32)
                print(f"CPP_READ_ERROR utc={stamp} path={cpp_path} mask_all_false=true error={read_error}", flush=True)
            valid = np.isfinite(cot) & (cot >= 0) & (cot <= 100) & (not read_error)
            phase_valid = np.isfinite(clp) & np.isin(clp, (0, 1, 2)) & (not read_error)
            mask = valid & phase_valid & agri_valid & (day > .5) & (cossoz > np.cos(np.deg2rad(80)))
            if cot.shape != (256, 256) or mask.shape != cot.shape:
                raise RuntimeError(f"unexpected CPP crop shape {cot.shape}")
            stat = cpp_path.stat()
            win_cot.append(cot)
            win_mask.append(mask)
            win_rec.append({"utc": stamp, "target_bjt": t_bjt.strftime("%Y-%m-%d %H:%M"),
                            "cpp_path": str(cpp_path), "cpp_size": stat.st_size,
                            "cpp_mtime_ns": stat.st_mtime_ns,
                            "aux_path": str(matches[0]),
                            "raw_valid_fraction": float(valid.mean()),
                            "masked_fraction": float(mask.mean()),
                            "cpp_read_error": read_error,
                            "zero_masked_fraction": float(((cot == 0) & mask).sum() / max(1, mask.sum())),
                            "thick_ge30_masked_fraction": float(((cot >= 30) & mask).sum() / max(1, mask.sum())),
                            "phase_values": [int(x) for x in np.unique(clp[np.isfinite(clp)])]})
        cot_rows.append(np.stack(win_cot))
        mask_rows.append(np.stack(win_mask))
        records.append({"seq_id": win["seq_id"], "bjt_start": win["bjt_start"], "month": win["month"],
                        "targets": win_rec})
        print(f"packed={win['seq_id']} valid_pixels={sum(x['masked_fraction']*256*256 for x in win_rec):.0f}", flush=True)
    cots = np.stack(cot_rows).astype(np.float32)[:, :, None]
    masks = np.stack(mask_rows).astype(bool)[:, :, None]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, cot=cots, mask=masks,
                        seq_ids=np.asarray([x["seq_id"] for x in windows], dtype="S32"))
    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    sidecar = {"state": "M1_TRAIN_ONLY_CPP_LABELS_PACKED", "test_used": False,
               "windows": records, "arrays": {"cot": list(cots.shape), "mask": list(masks.shape)},
               "archive_sha256": digest,
               "mask": "finite COT 0..100 AND CLP in {0,1,2} AND all13 AGRI valid bits AND day_mask>0.5 AND cosSOZ>cos80",
               "source_limit": "CPP retrieval reference; source producer and independent physical QA remain unresolved"}
    a.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": sidecar["state"], "cot": list(cots.shape), "mask": list(masks.shape),
                      "valid_fraction": float(masks.mean()), "sha256": digest, "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
