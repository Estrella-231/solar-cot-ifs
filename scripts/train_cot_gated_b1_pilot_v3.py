"""Seed-42 A/AC pilot with pointwise COT-conditioned AGRI modulation."""
import argparse
import hashlib
import json
from pathlib import Path

import train_station_aware_b1_pilot_v2 as base
from cot_gated_station_head_v3 import COTGatedStationHead

base.StationAwareSpatialHead = COTGatedStationHead
base.GROUPS = ("A_gate", "AC_gate")
base.FUSION_DESCRIPTION = "pointwise COT affine gate modulates same-pixel AGRI features"
base.AGGREGATION_DESCRIPTION = "station [8,8] feature + 5x5 station neighbourhood + full patch"

def predict(model, data, ids, group):
    cot = data["cot_map"][ids] if group == "AC_gate" else data["cot_map"][ids].new_zeros((len(ids), 1, 16, 16))
    with base.torch.autocast("cuda", dtype=base.torch.bfloat16):
        return model(data["x"][ids], cot, base.torch.zeros((len(ids), 4), device="cuda"),
                     data["g"][ids], data["station"][ids], data["lead"][ids]).float()

base.predict = predict

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--allow-unverified-cpp", action="store_true")
    a = p.parse_args()
    if not a.allow_unverified_cpp:
        raise RuntimeError("requires explicit exploratory CPP authorization")
    base.main(a.root, a.out, a.batch)
    complete = a.out / "complete.json"
    report = json.loads(complete.read_text())
    report["entrypoint_sha256"] = sha(__file__)
    report["gated_head_sha256"] = sha(Path(__file__).with_name("cot_gated_station_head_v3.py"))
    tmp = complete.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(complete)
