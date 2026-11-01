#!/usr/bin/env python3
"""Read one official U frame as station patches and run frozen R_U on CPU."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for n in ("support", "r-pack", "r-run", "data-root", "grid-static", "geometry-source", "source", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.support / "support_audit.json").read_text())
    assert audit["state"] == "PASS_24_FRAME_SUPPORT_DISJOINT_U_H_V"
    row = next(csv.DictReader((a.support / "u_sequences.csv").open(newline="", encoding="utf-8")))
    assert row["split"] == "train"
    rel = row["data_relpaths"].split("|")[0]

    sys.path.insert(0, str(a.source))
    from hunan_data import CHANNELS, resolve_aux
    sys.path.insert(0, str(Path(__file__).parent))
    from train_r_u import COTUNet
    sys.path.insert(0, str(a.geometry_source.parent))
    from solar_geometry_corrected import compute_solar_angles

    centers = {}
    for r in csv.DictReader((a.r_pack / "rows_u_supervised.csv").open(newline="", encoding="utf-8")):
        rc = tuple(map(int, r["station_grid_rc"].split(",")))
        if r["station"] in centers and centers[r["station"]] != rc:
            raise ValueError("station grid center changes in U R pack")
        centers[r["station"]] = rc
    assert set(centers) == {"sili", "zhujia"}
    t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
    raw = np.load(a.data_root / "data" / rel, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (20, 256, 256) and raw.dtype == np.float32
    aux = resolve_aux(a.data_root, t + timedelta(hours=8))
    with np.load(aux, allow_pickle=False) as z:
        bits = np.asarray(z["agri_valid_bits"], np.uint16)
    with np.load(a.grid_static, allow_pickle=False) as z:
        lat = np.asarray(z["lat"], np.float32)
        lon = np.asarray(z["lon"], np.float32)
        saa = np.asarray(z["saa"], np.float32)
    norm = json.loads((a.r_run / "normalization.json").read_text())
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]
    ck = torch.load(a.r_run / "epoch_050.pt", map_location="cpu", weights_only=False)
    model = COTUNet().eval()
    model.load_state_dict(ck["model"], strict=True)

    records = {}
    for station in ("sili", "zhujia"):
        rr, cc = centers[station]
        sl = np.s_[rr - 8:rr + 8, cc - 8:cc + 8]
        aa = np.asarray(raw[np.asarray(CHANNELS), sl[0], sl[1]], np.float32)
        vv = np.stack([(bits[sl] & (1 << k)) != 0 for k in CHANNELS]) & np.isfinite(aa)
        soz, soa = compute_solar_angles(lat[sl], lon[sl], t)
        soa = np.where(soz >= 90.0, 0.0, soa)
        gg = np.stack((np.cos(np.deg2rad(soz)), np.cos(np.deg2rad(soa - saa[sl])),
                       (soz < 85.0).astype(np.float32))).astype(np.float32)
        x = (np.concatenate((aa, gg), axis=0) - mean) / std
        x[~np.isfinite(x)] = 0.0
        x[:13, ~vv.all(axis=0)] = 0.0
        with torch.inference_mode():
            cot = model(torch.from_numpy(x[None]))
        assert cot.shape == (1, 1, 16, 16) and torch.isfinite(cot).all() and (cot >= 0).all()
        records[station] = {"valid_agri_channel_pixels": int(vv.sum()),
                            "valid_cot_geom_pixels": int(((gg[2] > .5) & (gg[0] > np.cos(np.deg2rad(80)))).sum()),
                            "cot_min": float(cot.min()), "cot_max": float(cot.max())}
    result = {"state": "PASS_U_RAW_PATCH_QA_CORRECTED_GEOMETRY_RU_CPU_SMOKE",
              "u_sequence_id": row["seq_id"], "utc_frame": Path(rel).stem,
              "station_patch_shape": [13, 16, 16], "stations": records,
              "r_u_checkpoint_sha256": sha(a.r_run / "epoch_050.pt"),
              "test_payloads_read": 0, "buffer_payloads_read": 0,
              "performance_score": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
