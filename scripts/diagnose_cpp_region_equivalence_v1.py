"""Separate continuous COT differences from cloud-phase classification changes."""
import argparse
import json
import netCDF4
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--region", required=True)
p.add_argument("--reference", required=True)
a = p.parse_args()
with np.load(a.region) as z:
    cot, clp = z["cot"], z["clp"]
with netCDF4.Dataset(a.reference) as z:
    rcot = np.asarray(z["COT"][1213:1469, 2061:2317])
    rclp = np.asarray(z["CLP"][1213:1469, 2061:2317])
different = clp != rclp
common = np.isfinite(cot) & np.isfinite(rcot)
same = common & ~different
delta = np.abs(cot-rcot)
coords = np.argwhere(different)
print(json.dumps({"phase_mismatches": int(different.sum()),
                  "sample_mismatch_coordinates": coords[:30].tolist(),
                  "cot_same_phase_mae": float(delta[same].mean()),
                  "cot_same_phase_max_abs": float(delta[same].max()),
                  "all_cot_mae": float(delta[common].mean())}))
