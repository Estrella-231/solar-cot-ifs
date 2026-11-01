#!/usr/bin/env python3
"""Extract validation-only fields for the four frozen CPP-selected events.

Run on the server in the existing ``swc`` environment. The extraction is
read-only: it indexes saved predictions, the compact CPP sequence pack, the
new-R bank and its validity sidecar; it does not open any test payload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_utc(value: str) -> datetime:
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    pack = a.branch / "data/sequence_cpp_pack_20261017"
    bank = a.branch.parent.parent / "experiments/monthly_R_station16_20261013_v1/results/new_monthly_R_seed42_cot_bank_v1"
    sidecars = a.branch / "data/frame_sidecars_20261017_v1"
    base_run = a.branch / "results/fulltrain_20261017_seed42_b1024_e12"
    amp_run = a.branch / "results/amplitude_loss_20261017_seed42_b1024_e12"
    event_manifest = json.loads(a.events.read_text(encoding="utf-8"))
    cases = event_manifest["selected_cases"]
    if len(cases) != 4 or any(c["frame_count"] != 17 for c in cases):
        raise ValueError("expected exactly four frozen 17-frame validation cases")

    seq_rows = [r for r in csv.DictReader((pack / "sequence_keys.csv").open(encoding="utf-8"))
                if r["split"] == "val"]
    seq_indices = np.load(pack / "sequence_indices_val.npy", mmap_mode="r", allow_pickle=False)
    pred_rows = list(csv.DictReader((base_run / "validation_prediction_keys.csv").open(encoding="utf-8")))
    if len(seq_rows) != len(seq_indices) or len(pred_rows) != 2 * len(seq_rows):
        raise ValueError("validation sequence/key/prediction counts disagree")

    cpp = np.load(pack / "cpp_cot_physical.npy", mmap_mode="r", allow_pickle=False)
    cpp_mask = np.load(pack / "cpp_common_mask.npy", mmap_mode="r", allow_pickle=False)
    r_log = np.load(bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    r_valid = np.load(sidecars / "frame_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    p_base = np.load(base_run / "P_Hybrid_val_pred_epoch09.npy", mmap_mode="r", allow_pickle=False)
    p_amp = np.load(amp_run / "P_Hybrid_val_pred_epoch11.npy", mmap_mode="r", allow_pickle=False)
    if p_base.shape != (2 * len(seq_rows), 16, 1, 16, 16) or p_amp.shape != p_base.shape:
        raise ValueError("direct model prediction tensor shape mismatch")

    out = {k: [] for k in ("cpp_cot", "common_valid", "future_agri_R_cot", "future_agri_R_valid",
                            "simvp_history_cot", "p_hybrid_base_cot", "p_hybrid_amp_cot")}
    records = []
    for case in cases:
        t_utc = parse_utc(case["init_time_bjt"])
        matches = [i for i, row in enumerate(seq_rows)
                   if parse_utc(row["init_time_utc"]) == t_utc]
        if len(matches) != 1:
            raise ValueError(f"{case['station']} {case['init_time_bjt']}: expected one val sequence, found {matches}")
        q = matches[0]
        seq = seq_rows[q]
        s = 0 if case["station"] == "sili" else 1 if case["station"] == "zhujia" else None
        if s is None:
            raise ValueError(f"unknown station {case['station']}")
        pred_row = pred_rows[q * 2 + s]
        if pred_row["station"] != case["station"] or parse_utc(pred_row["init_time_utc"]) != t_utc:
            raise ValueError("direct prediction key is not the expected sequence/station")
        frame_ix = np.asarray(seq_indices[q, 8:], dtype=np.int64)
        if frame_ix.shape != (16,):
            raise ValueError("expected 16 future frame indices")
        cm = np.asarray(cpp_mask[frame_ix, s, 0], dtype=bool)
        rv = np.asarray(r_valid[frame_ix, s, 0], dtype=bool)
        common = cm & rv
        c = np.asarray(cpp[frame_ix, s, 0], dtype=np.float32)
        rr = np.expm1(np.asarray(r_log[frame_ix, s, 0], dtype=np.float32)).clip(0.0)
        p0 = np.asarray(p_base[q * 2 + s, :, 0], dtype=np.float32)
        p1 = np.asarray(p_amp[q * 2 + s, :, 0], dtype=np.float32)
        if not all(x.shape == (16, 16, 16) for x in (c, common, rr, rv, p0, p1)):
            raise ValueError("unexpected extracted event tensor shape")
        for k, v in (("cpp_cot", c), ("common_valid", common), ("future_agri_R_cot", rr),
                     ("future_agri_R_valid", rv), ("simvp_history_cot", np.expm1(
                         np.asarray(r_log[int(seq_indices[q, 7]), s, 0], dtype=np.float32)).clip(0.0)),
                     ("p_hybrid_base_cot", p0), ("p_hybrid_amp_cot", p1)):
            out[k].append(v)
        records.append({
            "selection_type": case["selection_type"], "station": case["station"], "season": case["season"],
            "init_time_bjt": case["init_time_bjt"], "init_time_utc": seq["init_time_utc"],
            "seq_id": seq["seq_id"], "validation_sequence_index": q, "station_index": s,
            "direct_prediction_row": q * 2 + s, "future_target_leads_minutes": list(range(15, 241, 15)),
            "cpp_change_selection_value": case["D_mean_abs_adjacent_cot"],
            "initial_event_common_valid_pixels": case["common_valid_pixels_16x16"],
            "future_common_valid_pixels_by_lead": common.reshape(16, -1).sum(axis=1).astype(int).tolist(),
            "future_frame_indices": frame_ix.tolist(),
        })

    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **{k: np.stack(v) for k, v in out.items()})
    manifest = {
        "state": "PASS_FIXED_EVENT_KEYS_EXACT",
        "event_selection_manifest_sha256": sha256(a.events),
        "sequence_pack_manifest_sha256": sha256(pack / "pack_manifest.json"),
        "direct_validation_prediction_keys_sha256": sha256(base_run / "validation_prediction_keys.csv"),
        "base_P_Hybrid_prediction_sha256": sha256(base_run / "P_Hybrid_val_pred_epoch09.npy"),
        "amplitude_P_Hybrid_prediction_sha256": sha256(amp_run / "P_Hybrid_val_pred_epoch11.npy"),
        "R_bank_manifest_sha256": sha256(bank / "bank_manifest.json"),
        "R_frame_sidecar_manifest_sha256": sha256(sidecars / "pack_manifest.json"),
        "case_records": records,
        "n_events": len(records), "n_leads_per_event": 16,
        "cpp_is_independent_truth": False, "test_payload_opened": False,
    }
    manifest["output_sha256"] = sha256(a.output)
    a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": manifest["state"], "n_events": len(records),
                      "event_keys": [r["seq_id"] for r in records],
                      "output": str(a.output), "output_sha256": manifest["output_sha256"],
                      "test_payload_opened": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
