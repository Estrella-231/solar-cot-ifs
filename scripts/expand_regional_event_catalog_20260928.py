"""Expand validation cloud-event examples using existing frozen pilot arrays.

No model is trained. Selection is based only on CPP-referenced station COT
trajectory signatures and a precomputed historical motion proxy, not GHI or
SimVP error. CPP remains a retrieval reference, not independent physical truth.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def metrics_for(series: np.ndarray, valid: np.ndarray, motion: list[float]) -> dict:
    y = np.where(valid, series, np.nan)
    ok = np.flatnonzero(np.isfinite(y))
    if len(ok) == 0:
        return {"valid_leads": 0}
    early = y[:4]
    late = y[-4:]
    early_mean = float(np.nanmean(early)) if np.isfinite(early).any() else None
    late_mean = float(np.nanmean(late)) if np.isfinite(late).any() else None
    peak_i = int(np.nanargmax(y))
    peak = float(y[peak_i])
    early_peak = float(np.nanmax(y[:5])) if np.isfinite(y[:5]).any() else None
    late_peak_delta = None
    if np.isfinite(y[4:]).any() and early_mean is not None:
        late_peak_delta = float(np.nanmax(y[4:]) - early_mean)
    decay_drop = (float(early_peak - late_mean) if early_peak is not None and late_mean is not None else None)
    mnorm = float(np.linalg.norm(motion))
    return {
        "valid_leads": int(len(ok)), "peak_cot": peak, "peak_lead_min": 15 * (peak_i + 1),
        "early_mean_first4_cot": early_mean, "late_mean_last4_cot": late_mean,
        "late_peak_minus_early_mean_cot": late_peak_delta, "early_peak_first5_cot": early_peak,
        "early_peak_minus_late_mean_cot": decay_drop,
        "trajectory_range_cot": float(np.nanmax(y) - np.nanmin(y)),
        "motion_row_col_per15": [float(x) for x in motion], "motion_norm_pixels_per15": mnorm,
    }


def category_candidates(row: dict) -> list[str]:
    m = row["trajectory_metrics"]
    out = []
    if (m["valid_leads"] >= 10 and m["peak_cot"] >= 30 and m["peak_lead_min"] >= 75
            and m["late_peak_minus_early_mean_cot"] is not None
            and m["late_peak_minus_early_mean_cot"] >= 15):
        out.append("growth")
    if (m["valid_leads"] >= 10 and m["early_peak_first5_cot"] is not None
            and m["early_peak_first5_cot"] >= 30
            and m["early_peak_minus_late_mean_cot"] is not None
            and m["early_peak_minus_late_mean_cot"] >= 15):
        out.append("decay")
    if (m["valid_leads"] >= 10 and m["peak_cot"] >= 15
            and m["trajectory_range_cot"] >= 15 and m["motion_norm_pixels_per15"] >= 1.0):
        out.append("motion_proxy")
    if (m["valid_leads"] >= 12 and m["peak_cot"] < 20
            and m["trajectory_range_cot"] < 10):
        out.append("low_cot_control")
    return out


def spread_pick(rows: list[dict], k: int, used: set[int]) -> list[dict]:
    rows = [r for r in rows if r["sequence"] not in used]
    if not rows:
        return []
    # At most one example per BJT date within a category; sample evenly in time.
    by_day = {}
    for r in sorted(rows, key=lambda z: z["init_bjt"]):
        by_day.setdefault(r["init_bjt"][:10], r)
    days = list(by_day.values())
    indices = np.unique(np.linspace(0, len(days) - 1, min(k, len(days))).round().astype(int))
    return [days[int(i)] for i in indices]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-per-category", type=int, default=2)
    a = p.parse_args()
    exp = a.root / "experiments/regional_station_cot_20260925"
    bank = exp / "regional_ghi_pilot_bank_20260927_v1/val"
    cache = exp / "agri_cpp_pair_bank_v1"
    out = a.output
    out.mkdir(parents=True, exist_ok=False)

    seq_ids = np.load(bank / "sequence_indices.npy", allow_pickle=False).astype(int)
    forecast = np.load(bank / "cot_local_forecast.npy", mmap_mode="r")
    oracle = np.load(bank / "cot_local_oracle.npy", mmap_mode="r")
    history = np.load(bank / "cot_local_history.npy", mmap_mode="r")
    assert forecast.shape[0] == oracle.shape[0] == history.shape[0] == len(seq_ids)
    catalog_path = exp / "sili_thick_motion_20260927_v1/sequence_catalog.json"
    catalog = {int(x["sequence"]): x for x in json.loads(catalog_path.read_text())}
    cfg_path = a.root / "configs/s_frozen_hunan_seed42.json"
    cfg = json.loads(cfg_path.read_text())
    manifest_rows = list(__import__("csv").DictReader(open(cfg["manifest"], encoding="utf-8")))
    val_rows = [r for r in manifest_rows if r["split"] == "val"]
    # Sequence indices in the bank index the validation-only manifest view.
    rows_by_seq = {int(i): r for i, r in enumerate(val_rows)}

    records = []
    dirs = {}
    for sh in json.loads((cache / "index.json").read_text())["shards"]:
        d = cache / sh["directory"]
        dirs[sh["directory"]] = d
        for r in json.loads((d / "records.json").read_text()):
            if r["split"] == "val" and r["included_for_R"]:
                records.append({**r, "directory": d})
    by_stamp = {r["utc"]: r for r in records}
    ref_array = {}
    mask_array = {}
    for r in records:
        shdir = str(r["directory"])
        if shdir not in ref_array:
            ref_array[shdir] = np.load(r["directory"] / "cot.npy", mmap_mode="r")
            mask_array[shdir] = np.load(r["directory"] / "mask.npy", mmap_mode="r")

    rows = []
    for j, seq in enumerate(seq_ids):
        seq = int(seq)
        if seq not in catalog or seq not in rows_by_seq:
            continue
        rels = rows_by_seq[seq]["data_relpaths"].split("|")
        # CPP arrays are 256x256; local source and GHI pilot bank use a 64x64 Sili crop.
        ref = np.full(16, np.nan, np.float32)
        valid = np.zeros(16, bool)
        for ti, rel in enumerate(rels[8:24]):
            stamp = Path(rel).stem
            rec = by_stamp.get(stamp)
            if rec is None:
                continue
            shdir = str(rec["directory"])
            ar = int(rec["array_row"])
            cot = ref_array[shdir][ar]
            mask = mask_array[shdir][ar].astype(bool)
            r0, c0 = 163, 148
            local = cot[r0-32:r0+32, c0-32:c0+32]
            vm = mask[r0-32:r0+32, c0-32:c0+32] & np.isfinite(local)
            if vm[32, 32]:
                ref[ti] = local[32, 32]
                valid[ti] = True
        if not valid.any():
            continue
        motion = catalog[seq]["shift_row_col_per_15min"]
        met = metrics_for(ref, valid, motion)
        init_bjt = catalog[seq]["init"]
        rows.append({"bank_index": int(j), "sequence": seq, "init_bjt": init_bjt,
                     "trajectory_metrics": met, "categories": category_candidates({"trajectory_metrics": met})})

    category_names = ["growth", "decay", "motion_proxy", "low_cot_control"]
    selected = []
    # Existing plotted cases were selected in earlier outcome-aware diagnostics;
    # exclude them from this fresh event-signature sample.
    used: set[int] = {534, 2311, 2869}
    for cat in category_names:
        eligible = [r for r in rows if cat in r["categories"]]
        picks = spread_pick(eligible, a.max_per_category, used)
        for r in picks:
            used.add(r["sequence"])
            selected.append({**r, "selected_category": cat})

    # Save full trajectory catalog and compact case fields. Selection does not inspect GHI or model errors.
    (out / "event_candidate_catalog.json").write_text(json.dumps({
        "test_used": False, "training_performed": False,
        "cohort": "128 fixed Sili validation sequences; CPP matched subset; same frozen pilot arrays",
        "reference": "CPP satellite retrieval reference; not independent physical truth",
        "selection_rule": {
            "growth": ">=10 valid leads; CPP peak>=30; peak at >=75 min; later peak exceeds first-four mean by >=15 COT",
            "decay": ">=10 valid leads; first-five peak>=30; first-five peak exceeds last-four mean by >=15 COT",
            "motion_proxy": ">=10 valid leads; peak>=15; trajectory range>=15; precomputed history motion norm>=1 pixel/15min; a motion proxy, not verified cloud-object tracking",
            "low_cot_control": ">=12 valid leads; peak<20; trajectory range<10 COT",
            "sampling": "up to two distinct BJT dates per category, evenly spaced across eligible dates; no duplicate sequence across categories; selection does not inspect GHI or forecast error",
        },
        "candidate_counts": {c: sum(c in r["categories"] for r in rows) for c in category_names},
        "selected": selected, "all_candidates": rows,
    }, indent=2), encoding="utf-8")

    import torch
    from scipy.ndimage import shift as ndi_shift
    import sys
    sys.path.insert(0, "/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPFull_20260906")
    from hunan_data import load_frame
    data_root = Path("/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed")
    norm = json.loads(Path(cfg["normalization"]).read_text())
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]
    rgb_cache = {}
    for rec in records:
        rgb_cache.setdefault(rec["utc"], rec)
    for item in selected:
        j, seq = item["bank_index"], item["sequence"]
        rels = rows_by_seq[seq]["data_relpaths"].split("|")
        ref = np.full((16, 64, 64), np.nan, np.float32)
        valid = np.zeros((16, 64, 64), bool)
        rgb = np.full((16, 64, 64, 3), np.nan, np.float32)
        for ti, rel in enumerate(rels[8:24]):
            rec = by_stamp.get(Path(rel).stem)
            if rec is None:
                continue
            shdir = str(rec["directory"])
            ar = int(rec["array_row"])
            cot = ref_array[shdir][ar]
            mask = mask_array[shdir][ar].astype(bool)
            sl = (slice(131, 195), slice(116, 180))
            ref[ti] = cot[sl]
            valid[ti] = mask[sl] & np.isfinite(ref[ti])
            x = np.load(rec["directory"] / "x_raw.npy", mmap_mode="r")[ar]
            rgb[ti] = np.moveaxis(x[[2, 1, 0], 131:195, 116:180], 0, -1)
        simvp = np.expm1(np.asarray(forecast[j, 0, :, 0], np.float32))
        oracle_cot = np.expm1(np.asarray(oracle[j, 0, :, 0], np.float32))
        h = np.expm1(np.asarray(history[j, 0, -1, 0], np.float32))
        dy, dx = item["trajectory_metrics"]["motion_row_col_per15"]
        transport = np.stack([ndi_shift(h, shift=((ti+1)*dy, (ti+1)*dx), order=1,
                                        mode="constant", cval=np.nan, prefilter=False)
                              for ti in range(16)]).astype(np.float32)
        np.savez_compressed(out / f"event_seq{seq:04d}.npz", reference=ref, valid=valid,
                            rgb=rgb, simvp=simvp, oracle=oracle_cot, transport=transport,
                            history=h)
    print(json.dumps({"state": "COMPLETE_NO_TRAINING", "candidate_counts": {c: sum(c in r["categories"] for r in rows) for c in category_names},
                      "selected": [{"category": x["selected_category"], "sequence": x["sequence"], "init_bjt": x["init_bjt"]} for x in selected],
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
