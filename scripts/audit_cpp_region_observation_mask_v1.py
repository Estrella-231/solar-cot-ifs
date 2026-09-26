"""Explain regional invalid pixels using each independently calibrated HDF channel."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import netCDF4
import numpy as np
from produce_hunan_cpp_region_v1 import atomic_json, grid_mapping, sha

p = argparse.ArgumentParser()
for name in ("base", "grid", "region", "output"):
    p.add_argument("--"+name, type=Path, required=True)
p.add_argument("--time", required=True)
a = p.parse_args()
sys.path[:0] = [str(a.base), str(a.base / "Code")]
import Retrieval_FY4B_105E as adapter
from Satellite_Data_Loaders import Calibrate_FY_Channel
rows, cols, *_ = grid_mapping(a.grid)
sl = (slice(rows[0], rows[-1]+1), slice(cols[0], cols[-1]+1))
hr = adapter.AGRI_FUNC_LAT_INDEX[sl]
hc = adapter.AGRI_FUNC_LON_INDEX[sl]
ry0, ry1, cx0, cx1 = int(hr.min()), int(hr.max()+1), int(hc.min()), int(hc.max()+1)
target = datetime.fromisoformat(a.time)
tag = target.strftime("%Y%m%d%H%M%S")
files = list((Path(adapter.AGRI_DIR) / tag[:4] / tag[:8]).glob(f"*_{tag}_*_4000M_*.HDF"))
if len(files) != 1:
    raise RuntimeError("nonunique HDF")
masks, channels = [], []
with netCDF4.Dataset(files[0]) as ds:
    for c in (11, 12, 13, 14, 15):
        raw = ds["Data"][f"NOMChannel{c}"][ry0:ry1, cx0:cx1]
        lut = ds["Calibration"][f"CALChannel{c}"][:]
        value = Calibrate_FY_Channel(raw, lut)[hr-ry0, hc-cx0]
        finite = np.isfinite(value)
        masks.append(finite)
        channels.append({"channel": c, "finite_fraction": float(finite.mean()),
                         "native_masked_fraction": float(np.ma.getmaskarray(raw).mean())})
saz = adapter._Get_AGRI_SAZ()[sl]
masks.append(np.isfinite(saz))
independent = np.all(masks, axis=0)
with np.load(a.region, allow_pickle=False) as z:
    observed, valid, cot = z["observation_valid"], z["valid"], z["cot"]
    matches = bool(np.array_equal(independent, observed))
    report = {"utc": a.time, "hdf": str(files[0]), "hdf_size": files[0].stat().st_size,
              "grid_sha256": sha(a.grid), "region_sha256": sha(a.region), "channels": channels,
              "saz_finite_fraction": float(np.isfinite(saz).mean()),
              "independent_observed_fraction": float(independent.mean()),
              "stored_observed_fraction": float(observed.mean()), "mask_same": matches,
              "model_invalid_given_observed": int(np.sum(observed & ~valid)),
              "invalid_observations_filled_finite_cot": int(np.sum(~observed & np.isfinite(cot))),
              "test_used": False}
atomic_json(a.output, report)
print(json.dumps(report), flush=True)
if not matches:
    raise RuntimeError("stored observation mask differs from independently read raw channels")
