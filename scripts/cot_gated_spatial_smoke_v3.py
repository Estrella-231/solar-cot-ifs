"""B0 wrapper recording the COT-gated station-aware head contract."""
import argparse
import hashlib
import json
from pathlib import Path

import station_aware_spatial_smoke_v2 as base
from cot_gated_station_head_v3 import COTGatedStationHead, STATION_ROW_COL

base.StationAwareSpatialHead = COTGatedStationHead
base.STATION_ROW_COL = STATION_ROW_COL
base.FUSION_DESCRIPTION = "pointwise COT affine gate modulates same-pixel AGRI features"
base.AGGREGATION_DESCRIPTION = "exact station feature + 5x5 station neighbourhood + full-patch mean"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main(root, output, sidecar):
    base.main(root, output, sidecar)
    path = output / "station_aware_spatial_smoke.json"
    report = json.loads(path.read_text())
    report["entrypoint_sha256"] = sha(__file__)
    report["gated_head_sha256"] = sha(Path(__file__).with_name("cot_gated_station_head_v3.py"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    a = p.parse_args(); main(a.root, a.output, a.sidecar)
