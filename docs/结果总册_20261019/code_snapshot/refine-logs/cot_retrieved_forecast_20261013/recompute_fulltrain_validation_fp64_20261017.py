#!/usr/bin/env python3
"""Independent FP64 rescoring for the 2026-10-17 paired COT forecast screen.

Scores saved predictions and an offline observed-future-AGRI->R proxy on the
CPP-common mask. The paper's deployable SimVP-forecast-AGRI->R reference is
reported separately from its existing audited result. CPP is a retrieval
reference, not independent physical truth. No test payload is read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def summarize(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, cpp: np.ndarray):
    # Shapes: sample, lead, row, col. Use float64 accumulators, same station/lead macro.
    bins = [("all", -np.inf, np.inf), ("near_zero_lt1", -np.inf, 1.0),
            ("thin_1_5", 1.0, 5.0), ("moderate_5_30", 5.0, 30.0),
            ("thick_ge30", 30.0, np.inf)]
    result = {}
    for name, lo, hi in bins:
        sse = np.zeros((2, 16), np.float64)
        count = np.zeros((2, 16), np.int64)
        for i in range(pred.shape[0]):
            station = i % 2  # CSV/DataLoader order: Sili, then Zhujia per sequence.
            m = mask[i] & (cpp[i] >= lo) & (cpp[i] < hi)
            e = np.where(m, pred[i].astype(np.float64) - target[i].astype(np.float64), 0.0)
            sse[station] += np.square(e).sum(axis=(-2, -1))
            count[station] += m.sum(axis=(-2, -1))
        rmse = np.sqrt(np.divide(sse, count, out=np.full_like(sse, np.nan), where=count > 0))
        supported = np.isfinite(rmse)
        result[name] = {
            "station_lead_macro_rmse": float(np.nanmean(rmse)),
            "station_rmse": [float(np.nanmean(rmse[s])) for s in range(2)],
            "lead_rmse_station_mean": [float(np.nanmean(rmse[:, k])) for k in range(16)],
            "pixels": [int(count[s].sum()) for s in range(2)],
            "supported_station_lead_cells": int(supported.sum()),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--sidecars", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ids = np.load(a.pack / "sequence_indices_val.npy", mmap_mode="r", allow_pickle=False)
    bank = np.load(a.bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(a.pack / "cpp_cot_physical.npy", mmap_mode="r", allow_pickle=False)
    cm = np.load(a.pack / "cpp_common_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    valid = np.load(a.sidecars / "frame_valid_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    nseq = len(ids)
    target = np.empty((nseq * 2, 16, 16, 16), np.float32)
    ref = np.empty_like(target)
    mask = np.empty_like(target, dtype=bool)
    rpred = np.empty_like(target)
    for q in range(nseq):
        fi = np.asarray(ids[q, 8:], dtype=np.int64)
        for s in range(2):
            i = q * 2 + s
            target[i] = np.expm1(np.asarray(bank[fi, s, 0], dtype=np.float32))
            rpred[i] = target[i]
            ref[i] = np.asarray(cpp[fi, s, 0], dtype=np.float32)
            mask[i] = np.asarray(cm[fi, s, 0], dtype=bool) & np.asarray(valid[fi, s, 0], dtype=bool)
    # This is an offline reference using observed future AGRI, not the
    # deployable SimVP-forecast-AGRI baseline reported in the paper.
    full_ref, full_mask = ref, mask
    offline = "offline_observed_future_AGRI_to_R"
    full = {offline: summarize(rpred, full_ref, full_mask, full_ref)}
    arms = {offline: summarize(rpred[:, :, 7:10, 7:10],
                               ref[:, :, 7:10, 7:10], mask[:, :, 7:10, 7:10],
                               ref[:, :, 7:10, 7:10])}
    selections = {}
    for arm, ep in (("P_R", 3), ("P_Hybrid", 9)):
        path = a.run / f"{arm}_val_pred_epoch{ep:02d}.npy"
        pred = np.load(path, mmap_mode="r", allow_pickle=False)
        if pred.shape[0] != nseq * 2:
            raise ValueError(f"unexpected prediction shape {pred.shape}; expected first dimension {nseq*2}")
        # Saved model output is FP16; independently score its stored values in FP64.
        full[arm] = summarize(pred[:, :, 0], full_ref, full_mask, full_ref)
        arms[arm] = summarize(pred[:, :, 0, 7:10, 7:10], ref[:, :, 7:10, 7:10],
                              mask[:, :, 7:10, 7:10], ref[:, :, 7:10, 7:10])
        selections[arm] = {"epoch": ep, "prediction_sha256": sha256(path), "prediction_shape": list(pred.shape)}
    result = {"split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
              "sequences": nseq, "station_samples": nseq * 2, "leads": 16,
              "score_basis": "CPP-common pixel mask AND forecast-frame valid mask; FP64 SSE; station-by-lead macro",
              "cpp_is_independent_truth": False, "test_payload_opened": False,
              "selections": selections,
              "scores": {"full_patch_16x16": full, "station_local_3x3": arms}}
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({scale: {k: round(v["all"]["station_lead_macro_rmse"], 4)
                              for k, v in vals.items()}
                      for scale, vals in result["scores"].items()}, indent=2))
    print(a.out)


if __name__ == "__main__":
    main()
