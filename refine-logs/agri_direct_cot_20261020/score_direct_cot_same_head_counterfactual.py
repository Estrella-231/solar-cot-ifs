from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = Path(r"C:\Users\29728\AppData\Local\Temp\direct_cot_rowbank_recovery_20261021_0530")
PACK = TMP / "results" / "direct_cot_single_frame_ghi_pack_v1"
EVENT = ROOT / "results" / "direct_cot_ghi_validation_score_v1_local4060" / "event_diagnostics"
DAY = TMP / "day_assignment.csv"
OUT = EVENT / "same_head_zero_cot_bootstrap.json"
R = 5000
SEED = 20261020
STATIONS = ("sili", "zhujia")

by_month = defaultdict(list)
with DAY.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["split"] == "val":
            d = datetime.strptime(r["date_BJT"], "%Y-%m-%d").date()
            by_month[d.strftime("%Y-%m")].append(d)
blocks = {}
for month, ds in by_month.items():
    ds = sorted(set(ds))
    if len(ds) != 2 or (ds[1] - ds[0]).days != 1:
        raise ValueError(f"bad frozen val block {month}: {ds}")
    for d in ds:
        blocks[d.isoformat()] = month
if len(by_month) != 26:
    raise ValueError(f"expected 26 frozen blocks, got {len(by_month)}")

def load_station(station: str):
    d = PACK / station / "val"
    keys = [json.loads(x) for x in (d / "keys.jsonl").read_text(encoding="utf-8").splitlines()]
    cf = np.load(EVENT / f"{station}_seed42_same_head_zero_cot.npz", allow_pickle=False)
    assert len(keys) == len(cf["truth_ghi_wm2"])
    target_date = [(datetime.strptime(k["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat() for k in keys]
    b = np.array([blocks[x] for x in target_date])
    lead = np.array([int(k["lead_minutes"]) for k in keys])
    return {"truth": np.asarray(cf["truth_ghi_wm2"], np.float64),
            "with": np.asarray(cf["prediction_with_pxc_cot"], np.float64),
            "neutral": np.asarray(cf["prediction_same_head_cot_zero"], np.float64),
            "block": b, "short": (lead >= 15) & (lead <= 60),
            "keys_sha": hashlib.sha256((d / "keys.jsonl").read_bytes()).hexdigest()}

data = {s: load_station(s) for s in STATIONS}
block_ids = sorted(by_month)
rng = np.random.default_rng(SEED)
draws = rng.integers(0, len(block_ids), size=(R, len(block_ids)))

def components(d, arm, mask):
    e = d[arm] - d["truth"]
    num = np.zeros(len(block_ids), np.float64)
    den = np.zeros(len(block_ids), np.float64)
    for i, b in enumerate(block_ids):
        m = (d["block"] == b) & mask
        num[i] = np.sum(e[m] ** 2)
        den[i] = np.count_nonzero(m)
    return num, den

all_samples = {s: np.ones(len(data[s]["truth"]), dtype=bool) for s in STATIONS}
summary = {}
for band, masks in (("all_16_leads", all_samples),
                    ("lead_15_60", {s: data[s]["short"] for s in STATIONS})):
    draws_delta = np.zeros(R, np.float64)
    point_site = {}
    comps = {}
    for s in STATIONS:
        d = data[s]
        with_num, den = components(d, "with", masks[s])
        neutral_num, den0 = components(d, "neutral", masks[s])
        if not np.array_equal(den, den0):
            raise ValueError("paired arms have different row counts")
        comps[s] = (with_num, neutral_num, den)
        point_with = np.sqrt(with_num.sum() / den.sum())
        point_neutral = np.sqrt(neutral_num.sum() / den.sum())
        point_site[s] = {"with_cot_rmse": float(point_with), "neutral_cot_rmse": float(point_neutral),
                         "delta_rmse_with_minus_neutral": float(point_with - point_neutral)}
    for j in range(R):
        w = np.bincount(draws[j], minlength=len(block_ids)).astype(np.float64)
        ds = []
        for s in STATIONS:
            a, b, n = comps[s]
            ds.append(np.sqrt((a @ w) / (n @ w)) - np.sqrt((b @ w) / (n @ w)))
        draws_delta[j] = np.mean(ds)
    summary[band] = {"site_equal_point_delta_rmse": float(np.mean([v["delta_rmse_with_minus_neutral"] for v in point_site.values()])),
                     "site_equal_block_bootstrap_95ci": [float(x) for x in np.quantile(draws_delta, [.025, .975])],
                     "station_point_metrics": point_site}

manifest = {
    "state": "COMPLETE_DIRECT_COT_SAME_HEAD_COUNTERFACTUAL_BLOCK_BOOTSTRAP",
    "scope": "validation-only seed42 frozen P-XC heads; identical AGRI/geometry; with predicted COT vs neutral zero normalized COT; diagnostic not new trained candidate",
    "bootstrap": {"repetitions": R, "seed": SEED, "blocks": len(block_ids), "unit": "frozen consecutive two-day monthly validation blocks, with replacement and multiplicity retained"},
    "bands": summary,
    "test_payload_opened": False,
    "buffer_payload_opened": False,
    "future_observed_agri_used_as_input": False,
    "cpp_used_as_input": False,
    "validation_keys_sha256": {s: data[s]["keys_sha"] for s in STATIONS},
}
OUT.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, allow_nan=False))
