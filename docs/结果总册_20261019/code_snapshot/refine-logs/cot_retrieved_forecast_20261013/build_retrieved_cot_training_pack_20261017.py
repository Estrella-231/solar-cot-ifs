#!/usr/bin/env python3
"""Build train/val sequence indices and CPP target sidecars by explicit keys."""
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


def utc_key(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
        tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-keys", type=Path, required=True)
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--r-rows", type=Path, required=True)
    p.add_argument("--cpp-cot", type=Path, required=True)
    p.add_argument("--cpp-mask", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")
    frames = pd.read_csv(a.frame_keys, dtype=str, keep_default_na=False)
    if set(frames.split.unique()) - {"train", "val"}:
        raise ValueError("bank frame keys include a non-train/val split")
    if frames.target_time_utc.duplicated().any():
        raise ValueError("duplicate bank timestamp")
    if not np.array_equal(frames["index"].astype(int).to_numpy(), np.arange(len(frames))):
        raise ValueError("bank frame index column is not dense row order")
    frame_idx = {t: int(i) for i, t in zip(frames["index"], frames.target_time_utc)}
    frame_split = dict(zip(frames.target_time_utc, frames.split))
    frame_rel = dict(zip(frames.target_time_utc, frames.agri_frame_relpath))

    seq = pd.read_csv(a.sequences, dtype=str, keep_default_na=False)
    if set(seq.split.unique()) - {"train", "val"}:
        raise ValueError("sequence manifest contains a non-train/val split")
    packed = {"train": [], "val": []}
    key_rows = []
    for r in seq.itertuples(index=False):
        hist = [utc_key(x) for x in r.history_frame_utc.split("|")]
        fut = [utc_key(x) for x in r.target_frame_utc.split("|")]
        rel = r.history_relpaths.split("|") + r.target_relpaths.split("|")
        times = hist + fut
        if len(hist) != 8 or len(fut) != 16 or len(rel) != 24:
            raise ValueError(f"invalid sequence lengths {r.seq_id}")
        if any(t not in frame_idx for t in times):
            raise ValueError(f"sequence misses bank frame {r.seq_id}")
        if any(frame_split[t] != r.split for t in times):
            raise ValueError(f"cross-split sequence {r.seq_id}")
        if any(frame_rel[t] != path for t, path in zip(times, rel)):
            raise ValueError(f"sequence path mismatch {r.seq_id}")
        ids = [frame_idx[t] for t in times]
        seq_row = len(packed[r.split])
        packed[r.split].append(ids)
        key_rows.append([r.split, seq_row, r.seq_id, r.init_time_utc, r.init_time_bjt])

    rrows = pd.read_csv(a.r_rows, low_memory=False)
    sup = rrows.loc[rrows.state.eq("SUPERVISED")].reset_index(drop=True)
    if len(sup) != 16812:
        raise ValueError(f"unexpected CPP supervised row count {len(sup)}")
    if not np.array_equal(sup["array_index"].astype(int).to_numpy(), np.arange(len(sup))):
        raise ValueError("CPP pack array_index does not match filtered SUPERVISED row order")
    cpp = np.load(a.cpp_cot, mmap_mode="r", allow_pickle=False)
    cpp_mask = np.load(a.cpp_mask, mmap_mode="r", allow_pickle=False)
    if cpp.shape != (len(sup), 1, 16, 16) or cpp_mask.shape != cpp.shape:
        raise ValueError("CPP target/mask shape differs from SUPERVISED rows")

    a.output.mkdir(parents=True)
    n, stations = len(frames), ("sili", "zhujia")
    out_cot = np.lib.format.open_memmap(a.output/"cpp_cot_physical.npy", mode="w+",
                                       dtype=np.float32, shape=(n,2,1,16,16))
    out_mask = np.lib.format.open_memmap(a.output/"cpp_common_mask.npy", mode="w+",
                                        dtype=np.uint8, shape=(n,2,1,16,16))
    out_cot[:] = 0; out_mask[:] = 0
    seen = set()
    for i, r in sup.iterrows():
        key = str(r.target_time_utc)
        if key not in frame_idx or frame_split[key] != r.split:
            raise ValueError(f"CPP/bank split or timestamp mismatch {(r.station,key)}")
        if r.station not in stations:
            raise ValueError(f"unexpected station {r.station}")
        sample_key = (r.station, key)
        if sample_key in seen:
            raise ValueError(f"duplicate CPP row {sample_key}")
        seen.add(sample_key)
        idx, si = frame_idx[key], stations.index(r.station)
        m = cpp_mask[i].astype(bool)
        vals = np.asarray(cpp[i], dtype=np.float32)
        if not np.isfinite(vals[m]).all() or np.any(vals[m] < 0):
            raise ValueError(f"invalid CPP physical COT on common mask: {sample_key}")
        out_cot[idx,si] = np.where(m, vals, 0.0)
        out_mask[idx,si] = m.astype(np.uint8)
    out_cot.flush(); out_mask.flush()
    for split in ("train", "val"):
        np.save(a.output/f"sequence_indices_{split}.npy", np.asarray(packed[split], dtype=np.int32), allow_pickle=False)
    with (a.output/"sequence_keys.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["split","split_index","seq_id","init_time_utc","init_time_bjt"]); w.writerows(key_rows)
    meta = {
        "state": "PASS_TRAIN_VAL_SEQUENCE_CPP_PACK",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "stations": list(stations), "frames": n,
        "sequences": {k: len(v) for k,v in packed.items()},
        "cpp_supervised_rows": len(seen),
        "cpp_common_valid_pixels": int(out_mask.sum()),
        "cpp_values_are_physical_dimensionless_cot": True,
        "sequence_index_layout": "[sequence, 24 frames: 8 history then 16 targets]",
        "frame_layout": "[bank_frame_index, station(sili,zhujia), 1,16,16]",
        "test_payload_opened": False,
        "source_sha256": {p.name: sha256(p) for p in (a.frame_keys,a.sequences,a.r_rows,a.cpp_cot,a.cpp_mask)},
        "output_sha256": {p.name: sha256(p) for p in a.output.iterdir() if p.is_file() and p.name != "pack_manifest.json"},
    }
    (a.output/"pack_manifest.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))


if __name__ == "__main__":
    main()
