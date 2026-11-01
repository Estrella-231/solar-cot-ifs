#!/usr/bin/env python3
"""Metadata-only joins for train/validation retrieved-COT forecast assets.

This deliberately does not open any bank COT/AGRI frame payloads. It verifies
the exact identity of CPP supervised-row indices and the 24-frame temporal
windows against the new monthly COT-bank frame index.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def parse_utc(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
        tzinfo=None
    ).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r-rows", type=Path, required=True)
    p.add_argument("--clp", type=Path, required=True)
    p.add_argument("--frame-keys", type=Path, required=True)
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    bank = json.loads(a.bank_manifest.read_text(encoding="utf-8"))
    if bank.get("state") != "COMPLETE" or bank.get("test_payload_opened") is not False:
        raise ValueError("bank completion/test gate failed")
    if bank.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split contract")

    rows = pd.read_csv(a.r_rows, low_memory=False)
    supervised = rows.loc[rows.state.eq("SUPERVISED")].reset_index(drop=True)
    with np.load(a.clp, allow_pickle=False) as z:
        n = len(supervised)
        if not (len(z["sample_indices"]) == len(z["split"]) == len(z["station"])
                == len(z["target_time_utc"]) == n):
            raise ValueError("CPP sidecar and SUPERVISED row count differ")
        if not np.array_equal(z["sample_indices"], np.arange(n)):
            raise ValueError("CPP compact indices are not dense SUPERVISED-row indices")
        for col in ("split", "station", "target_time_utc"):
            if not np.array_equal(z[col].astype(str), supervised[col].astype(str).to_numpy()):
                raise ValueError(f"CPP sidecar key mismatch: {col}")
        clp_values = z["clp"]
        clp_mask = z["common_mask"].astype(bool)
        if clp_values.shape != clp_mask.shape or clp_values.shape[1:] != (1, 16, 16):
            raise ValueError("CPP field/mask shape mismatch")
        if np.any(clp_mask & ~np.isin(clp_values, (0, 1, 2))):
            raise ValueError("invalid CLP class under common mask")

    fk = pd.read_csv(a.frame_keys, dtype=str, keep_default_na=False)
    fk["index"] = fk["index"].astype(int)
    if fk["index"].tolist() != list(range(len(fk))):
        raise ValueError("bank frame indices are not dense/order-preserving")
    if fk["target_time_utc"].duplicated().any():
        raise ValueError("duplicate frame timestamp in bank")
    frame = fk.set_index("target_time_utc").to_dict("index")

    seq = pd.read_csv(a.sequences, dtype=str, keep_default_na=False)
    if not set(seq["split"].unique()).issubset({"train", "val"}):
        raise ValueError("sequence manifest contains a non-train/val split")
    checked = {"train": 0, "val": 0}
    for r in seq.itertuples(index=False):
        hist = [parse_utc(x) for x in r.history_frame_utc.split("|")]
        fut = [parse_utc(x) for x in r.target_frame_utc.split("|")]
        times = hist + fut
        if len(hist) != 8 or len(fut) != 16:
            raise ValueError(f"bad 8-to-16 sequence lengths: {r.seq_id}")
        if any(t not in frame for t in times):
            raise ValueError(f"sequence timestamp missing from COT bank: {r.seq_id}")
        if any(frame[t]["split"] != r.split for t in times):
            raise ValueError(f"cross-split frame in sequence: {r.seq_id}")
        expected = [r.history_relpaths.split("|") + r.target_relpaths.split("|")][0]
        if any(frame[t]["agri_frame_relpath"] != rel for t, rel in zip(times, expected)):
            raise ValueError(f"sequence path/frame-key mismatch: {r.seq_id}")
        checked[r.split] += 1

    if checked != {"train": 35320, "val": 3140}:
        raise ValueError(f"unexpected sequence counts: {checked}")
    if set(supervised.split.unique()) != {"train", "val"}:
        raise ValueError("CPP rows do not contain exactly train/val")
    result = {
        "state": "PASS_METADATA_KEY_JOIN",
        "split_id": bank["split_id"],
        "bank_candidate_sha256": bank["candidate_sha256"],
        "bank_R_checkpoint_sha256": bank["R_checkpoint_sha256"],
        "bank_frames": len(fk),
        "sequences_checked": checked,
        "cpp_supervised_rows": len(supervised),
        "cpp_train_rows": int(supervised.split.eq("train").sum()),
        "cpp_val_rows": int(supervised.split.eq("val").sum()),
        "cpp_common_valid_pixels": int(clp_mask.sum()),
        "clp_counts_common_mask": {str(k): int(v) for k, v in zip(*np.unique(clp_values[clp_mask], return_counts=True))},
        "test_payload_opened": False,
        "payloads_opened": ["CPP COT/class/mask arrays only; no AGRI/COT-bank arrays"],
        "sha256": {p.name: sha256(p) for p in (a.r_rows, a.clp, a.frame_keys, a.sequences, a.bank_manifest)},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
