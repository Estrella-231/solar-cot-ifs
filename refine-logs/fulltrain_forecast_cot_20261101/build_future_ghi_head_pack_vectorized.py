"""Vectorized equivalent of build_future_ghi_head_pack.py.

This keeps the original sample contract and row order while gathering the
three causal frames in chunks. It is a separate producer so an already-running
versioned build retains an honest producer hash and output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_future_ghi_head_pack import (
    NLEADS, PATCH, SPLITS, STATIONS, context_source, csv_rows, sha,
    train_forecast_provenance,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", type=Path, required=True)
    p.add_argument("--crosswalk", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--ghi-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-rows", type=int, default=4096)
    p.add_argument("--limit-rows-per-split", type=int, default=None,
                   help="benchmark only; writes an explicitly incomplete pack, never a production pack")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.chunk_rows < 1:
        raise ValueError("chunk-rows must be positive")

    bank_meta = json.loads((a.forecast_bank / "bank_manifest.json").read_text(encoding="utf-8"))
    label_meta = json.loads((a.ghi_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    cross_meta = json.loads(a.crosswalk.read_text(encoding="utf-8"))
    if bank_meta.get("state") != "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK":
        raise ValueError("forecast bank is not complete")
    if bank_meta.get("test_payload_opened") is not False or bank_meta.get("future_observed_agri_read") is not False:
        raise ValueError("forecast-bank provenance does not certify future-label isolation")
    if label_meta.get("state") != "COMPLETE" or label_meta.get("test_payload_opened") is not False:
        raise ValueError("GHI label pack gate failed")
    if cross_meta.get("split_mismatch_rows") != 0:
        raise ValueError("crosswalk has train/validation split mismatches")

    with np.load(a.keys, allow_pickle=False) as z:
        split_code = z["split_code"].astype(np.uint8, copy=False)
        seq_index = z["sequence_index_within_split"].astype(np.int32, copy=False)
        station_index = z["station_index"].astype(np.uint8, copy=False)
        lead_index = z["lead_index"].astype(np.uint8, copy=False)
        ghi_index = z["ghi_row_index"].astype(np.int32, copy=False)
    n = len(split_code)
    if n != int(bank_meta.get("crosswalk_row_count_expected", n)):
        raise ValueError("crosswalk row count differs from forecast-bank contract")
    if any(len(x) != n for x in (seq_index, station_index, lead_index, ghi_index)):
        raise ValueError("crosswalk key arrays differ in length")
    if not (np.isin(split_code, [0, 1]).all() and (station_index < len(STATIONS)).all()
            and (lead_index < NLEADS).all()):
        raise ValueError("crosswalk contains invalid split/station/lead index")

    # The bank CSV order is authoritative. Dense split-local lookups replace
    # per-sample Python dict queries and preserve arbitrary sequence indices.
    bank_rows = list(csv_rows(a.forecast_bank / "index.csv"))
    if any(r["split"] not in SPLITS for r in bank_rows):
        raise ValueError("forecast bank contains a non-train/validation index")
    lookup = {}
    for code, split_name in enumerate(SPLITS):
        rows = [(i, int(r["sequence_index_within_split"])) for i, r in enumerate(bank_rows)
                if r["split"] == split_name]
        if len(rows) != sum(1 for r in bank_rows if r["split"] == split_name):
            raise ValueError("forecast bank contains a non-train/validation split")
        max_seq = max((seq for _, seq in rows), default=-1)
        arr = np.full(max_seq + 1, -1, np.int64)
        for bank_i, seq in rows:
            if arr[seq] >= 0:
                raise ValueError(f"duplicate forecast bank sequence {(split_name, seq)}")
            arr[seq] = bank_i
        lookup[code] = arr
    provenance = train_forecast_provenance(
        bank_meta, sum(1 for r in bank_rows if r["split"] == "train"))
    bank_i = np.full(n, -1, np.int64)
    for code in (0, 1):
        mask = split_code == code
        if mask.any():
            seq = seq_index[mask]
            if (seq < 0).any() or (seq >= len(lookup[code])).any():
                raise ValueError("crosswalk sequence index out of range")
            bank_i[mask] = lookup[code][seq]
    if (bank_i < 0).any():
        raise ValueError("crosswalk sequence missing from forecast bank")

    if provenance["train_forecast_oof"]:
        expected_arrays = {"history_agri_raw.npy", "history_geometry.npy", "history_cot_log1p.npy",
                           "forecast_agri_raw.npy", "forecast_geometry.npy", "forecast_cot_log1p.npy"}
        declared = bank_meta.get("array_sha256")
        if not isinstance(declared, dict) or set(declared) != expected_arrays:
            raise ValueError("OOF bank array hashes are missing or incomplete")
        for name in expected_arrays:
            if declared[name] != sha(a.forecast_bank / name):
                raise ValueError(f"OOF bank array hash mismatch: {name}")
        if bank_meta.get("index_sha256") != sha(a.forecast_bank / "index.csv"):
            raise ValueError("OOF bank index hash mismatch")

    label_rows = list(csv_rows(a.ghi_pack / "rows.csv"))
    if (ghi_index < 0).any() or ghi_index.max(initial=-1) >= len(label_rows):
        raise ValueError("GHI row index is out of range")
    label_split = np.fromiter((0 if r["split"] == "train" else 1 if r["split"] == "val" else 255
                               for r in label_rows), dtype=np.uint8, count=len(label_rows))
    label_station = np.fromiter((STATIONS.index(r["station"].lower()) if r["station"].lower() in STATIONS else 255
                                 for r in label_rows), dtype=np.uint8, count=len(label_rows))
    label_ghi = np.fromiter((float(r["ghi"]) for r in label_rows), dtype=np.float32, count=len(label_rows))
    label_clear = np.fromiter((float(r["clear"]) for r in label_rows), dtype=np.float32, count=len(label_rows))
    label_target = np.asarray([r["target_time_utc"] for r in label_rows], dtype="datetime64[s]")
    if (label_split[ghi_index] != split_code).any() or (label_station[ghi_index] != station_index).any():
        bad = int(np.flatnonzero((label_split[ghi_index] != split_code)
                                 | (label_station[ghi_index] != station_index))[0])
        raise ValueError(f"label split/station mismatch at crosswalk row {bad}")

    specs = {
        "history_agri_raw.npy": (2, 13), "history_geometry.npy": (2, 2),
        "history_cot_log1p.npy": (2, 1), "forecast_agri_raw.npy": (NLEADS, 13),
        "forecast_geometry.npy": (NLEADS, 2), "forecast_cot_log1p.npy": (NLEADS, 1),
    }
    arrays = {}
    for name, (time_n, channels) in specs.items():
        x = np.load(a.forecast_bank / name, mmap_mode="r", allow_pickle=False)
        expected = (len(bank_rows), 2, time_n, channels, PATCH, PATCH)
        if x.shape != expected or x.dtype != np.float32:
            raise ValueError(f"{name}: expected float32 {expected}, got {x.dtype} {x.shape}")
        arrays[name] = x

    a.output.mkdir(parents=True)
    split_rows = {"train": np.flatnonzero(split_code == 0), "val": np.flatnonzero(split_code == 1)}
    for sn, ids in split_rows.items():
        dates=(label_target[ghi_index[ids]]+np.timedelta64(8,"h")).astype("datetime64[D]")
        split_rows[sn]=ids[~((station_index[ids]==0)&(dates==np.datetime64("2024-09-05")))]
    if a.limit_rows_per_split is not None:
        if a.limit_rows_per_split < 1:
            raise ValueError("limit-rows-per-split must be positive")
        split_rows = {name: ids[:a.limit_rows_per_split] for name, ids in split_rows.items()}
    row_files, counts = {}, {}
    for split_code_value, split_name in enumerate(SPLITS):
        ids = split_rows[split_name]
        d = a.output / split_name
        d.mkdir()
        count = len(ids)
        outputs = {
            "agri_raw.npy": np.lib.format.open_memmap(d / "agri_raw.npy", mode="w+", dtype=np.float32,
                                                       shape=(count, 3, 13, 3, 3)),
            "geometry.npy": np.lib.format.open_memmap(d / "geometry.npy", mode="w+", dtype=np.float32,
                                                       shape=(count, 3, 2, 3, 3)),
            "cot_log1p.npy": np.lib.format.open_memmap(d / "cot_log1p.npy", mode="w+", dtype=np.float32,
                                                        shape=(count, 3, 1, 3, 3)),
            "ghi.npy": np.lib.format.open_memmap(d / "ghi.npy", mode="w+", dtype=np.float32, shape=(count,)),
            "clear.npy": np.lib.format.open_memmap(d / "clear.npy", mode="w+", dtype=np.float32, shape=(count,)),
        }
        keys_path = d / "keys.csv"
        with keys_path.open("w", newline="", encoding="utf-8") as key_f:
            key_f.write("split,station,sequence_index_within_split,target_time_utc,init_time_utc,lead_minutes,ghi_row_index,crosswalk_row_index\r\n")
            for begin in range(0, count, a.chunk_rows):
                end = min(begin + a.chunk_rows, count)
                cross = ids[begin:end]
                bi, st, li, gi, sq = bank_i[cross], station_index[cross], lead_index[cross], ghi_index[cross], seq_index[cross]
                lead_minutes = 15 * (li.astype(np.int64) + 1)
                for output_name, hist_name, future_name in (
                    ("agri_raw.npy", "history_agri_raw.npy", "forecast_agri_raw.npy"),
                    ("geometry.npy", "history_geometry.npy", "forecast_geometry.npy"),
                    ("cot_log1p.npy", "history_cot_log1p.npy", "forecast_cot_log1p.npy"),
                ):
                    channels = outputs[output_name].shape[2]
                    for pos, offset in enumerate((-2, -1, 0)):
                        is_hist, src = context_source(li, offset)
                        block = np.empty((len(cross), channels, PATCH, PATCH), dtype=np.float32)
                        hist_rows = np.flatnonzero(is_hist)
                        future_rows = np.flatnonzero(~is_hist)
                        if len(hist_rows):
                            block[hist_rows] = arrays[hist_name][bi[hist_rows], st[hist_rows], src[hist_rows]]
                        if len(future_rows):
                            block[future_rows] = arrays[future_name][bi[future_rows], st[future_rows], src[future_rows]]
                        if not np.isfinite(block).all():
                            raise ValueError(f"non-finite {output_name} at crosswalk row {int(cross[0])}")
                        outputs[output_name][begin:end, pos] = block
                outputs["ghi.npy"][begin:end] = label_ghi[gi]
                outputs["clear.npy"][begin:end] = label_clear[gi]

                targets = label_target[gi]
                init_times = targets - lead_minutes.astype("timedelta64[m]")
                target_strings = np.char.replace(targets.astype(str), "T", " ")
                init_strings = np.char.replace(init_times.astype(str), "T", " ")
                stations = np.asarray(STATIONS, dtype="U16")[st]
                columns = [np.full(len(cross), split_name, dtype="U5"), stations, sq.astype(str),
                           target_strings, init_strings, lead_minutes.astype(str), gi.astype(str), cross.astype(str)]
                lines = columns[0]
                for col in columns[1:]:
                    lines = np.char.add(np.char.add(lines, ","), col)
                np.savetxt(key_f, lines, fmt="%s", newline="\r\n")
        for value in outputs.values():
            value.flush()
        row_files[split_name] = {name: sha(d / name) for name in
                                 ("agri_raw.npy", "geometry.npy", "cot_log1p.npy", "ghi.npy", "clear.npy", "keys.csv")}
        counts[split_name] = count

    train_agri = np.load(a.output / "train/agri_raw.npy", mmap_mode="r", allow_pickle=False)
    train_cot = np.load(a.output / "train/cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    ag_mean = np.empty(13, np.float64); ag_std = np.empty(13, np.float64)
    for c in range(13):
        v = np.asarray(train_agri[:, :, c], dtype=np.float64)
        ag_mean[c] = v.mean(); ag_std[c] = max(v.std(), 1e-6)
    c = np.asarray(train_cot, dtype=np.float64)
    cot_mean = float(c.mean()); cot_std = max(float(c.std()), 1e-6)
    meta = {
        "state": ("INCOMPLETE_BENCHMARK_ONLY" if a.limit_rows_per_split is not None
                  else "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK"), **provenance,
        "split_id": label_meta["split_id"], "sample_key": "(station,init_time_utc,target_time_utc,lead)",
        "rows": counts, "history_offsets_minutes": [-15, 0], "lead_minutes": list(range(15, 241, 15)),
        "context_rule": "three causal frames [target-30,target-15,target]; history slots are init-15 and init, so +15 uses both history slots then forecast lead +15",
        "input_sources": {"history_agri": "observed AGRI before init only", "history_cot": "frozen R on observed history AGRI only",
                          "future_agri": "frozen SimVP prediction only", "future_cot": "frozen R applied to predicted AGRI only",
                          "geometry": "deterministic target-frame cosSOZ/cosRAA"},
        "normalization_fit": "per-channel AGRI and log1p(COT) mean/std from the materialized training forecast instances only; shared by A and AC",
        "AGRI_mean": ag_mean.tolist(), "AGRI_std": ag_std.tolist(),
        "COT_log1p_mean": cot_mean, "COT_log1p_std": cot_std,
        "producer_sha256": sha(Path(__file__)),
        "forecast_bank_manifest_sha256": sha(a.forecast_bank / "bank_manifest.json"),
        "crosswalk_sha256": sha(a.crosswalk), "crosswalk_keys_sha256": sha(a.keys),
        "ghi_label_pack_manifest_sha256": sha(a.ghi_pack / "pack_manifest.json"), "array_sha256": row_files,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False, "cpp_used_as_input": False,
        "builder": "vectorized_chunk_gather", "chunk_rows": a.chunk_rows,
        "benchmark_limit_rows_per_split": a.limit_rows_per_split,
    }
    (a.output / "pack_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": meta["state"], "rows": counts, "test_payload_opened": False,
                      "AGRI_norm_mean": meta["AGRI_mean"], "COT_norm_mean": cot_mean}, indent=2))


if __name__ == "__main__":
    main()
