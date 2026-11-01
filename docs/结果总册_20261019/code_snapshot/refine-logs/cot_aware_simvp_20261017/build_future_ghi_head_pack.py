"""Materialize causal 3-frame, 3x3 future-GHI inputs from a frozen forecast bank.

This script consumes only train/validation rows. Forecast arrays must contain
SimVP-predicted AGRI and frozen-R COT; history arrays contain only the last two
observed AGRI/R-COT frames. No observed future AGRI or CPP array is accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
SPLITS = ("train", "val")
NLEADS = 16
PATCH = 3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def context_source(lead_index: np.ndarray, offset: int):
    """Return (is_history, source_index) for T-30/T-15/T at each lead.

    ``lead_index`` is zero-based for +15..+240 min. Negative source indices
    address the two observed AGRI frames at init-15 min and init.
    """
    source = lead_index.astype(np.int16) + int(offset)
    is_history = source < 0
    # -2 maps to history slot 0 (init-15); -1 maps to slot 1 (init).
    index = np.where(is_history, source + 2, source)
    if (index < 0).any() or (index >= NLEADS).any():
        raise ValueError("context source index out of range")
    return is_history, index.astype(np.int16)


def history_context_mask(lead_minutes: np.ndarray) -> np.ndarray:
    """Mask the [target-30,target-15,target] positions available at init.

    The mask marks context timestamps at or before init_time. For +15 min the
    history slots are init-15 and init; for +30 min only init is historical.
    """
    lead = np.asarray(lead_minutes, dtype=np.int16)
    return np.stack((lead <= 30, lead <= 15, lead <= 0), axis=-1).astype(np.float32)


def train_forecast_provenance(bank_meta: dict, train_sequences: int) -> dict:
    in_sample = bank_meta.get("in_sample_train_forecasts")
    if not isinstance(in_sample, bool):
        raise ValueError("forecast bank must explicitly declare whether train forecasts are in-sample")
    if not in_sample:
        if bank_meta.get("oof_fold_count") != 5:
            raise ValueError("OOF forecast bank must certify all five date-heldout folds")
        if bank_meta.get("oof_train_sequences") != train_sequences:
            raise ValueError("OOF forecast bank does not certify complete train-key coverage")
        fold_hashes = bank_meta.get("oof_fold_model_hashes")
        if not isinstance(fold_hashes, dict) or set(fold_hashes) != {str(i) for i in range(5)}:
            raise ValueError("OOF forecast bank must identify all five fold checkpoints")
        try:
            hashes_valid = all(isinstance(v, str) and len(v) == 64 and int(v, 16) >= 0
                               for v in fold_hashes.values())
        except ValueError:
            hashes_valid = False
        if not hashes_valid:
            raise ValueError("OOF fold checkpoint hashes are incomplete")
        source = bank_meta.get("forecast_source")
        if not isinstance(source, str) or "oof" not in source.lower():
            raise ValueError("OOF forecast bank source description is missing")
    return {"train_forecast_oof": not in_sample,
            "train_forecast_source": bank_meta.get("forecast_source"),
            "forecast_bank_oof_fold_count": bank_meta.get("oof_fold_count", 0),
            "warning": ("Train rows use date-heldout five-fold SimVP OOF predictions; validation rows use the fixed full-train SimVP. This is validation screening, not independent-test evidence."
                        if not in_sample else
                        "Training forecasts are in-sample; this pack is a screening pilot only and cannot establish forecast-input generalization.")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", type=Path, required=True)
    p.add_argument("--crosswalk", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--ghi-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

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

    bank_index = {}
    for i, row in enumerate(csv_rows(a.forecast_bank / "index.csv")):
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("forecast bank contains a non-train/validation index")
        key = (0 if split == "train" else 1, int(row["sequence_index_within_split"]))
        if key in bank_index:
            raise ValueError(f"duplicate sequence bank key {key}")
        bank_index[key] = i
    provenance = train_forecast_provenance(
        bank_meta, sum(1 for split, _ in bank_index if split == 0))
    if provenance["train_forecast_oof"]:
        expected_arrays = {"history_agri_raw.npy", "history_geometry.npy", "history_cot_log1p.npy",
                           "forecast_agri_raw.npy", "forecast_geometry.npy", "forecast_cot_log1p.npy"}
        declared_arrays = bank_meta.get("array_sha256")
        if not isinstance(declared_arrays, dict) or set(declared_arrays) != expected_arrays:
            raise ValueError("OOF bank array hashes are missing or incomplete")
        for name in expected_arrays:
            if declared_arrays[name] != sha(a.forecast_bank / name):
                raise ValueError(f"OOF bank array hash mismatch: {name}")
        if bank_meta.get("index_sha256") != sha(a.forecast_bank / "index.csv"):
            raise ValueError("OOF bank index hash mismatch")

    label_rows = list(csv_rows(a.ghi_pack / "rows.csv"))
    if max(ghi_index, default=-1) >= len(label_rows):
        raise ValueError("GHI row index is out of range")
    expected_keys = np.load(a.keys, allow_pickle=False)

    # Bank layouts: [sequence, station, recent_history(2), channel, 3, 3]
    # and [sequence, station, forecast_lead(16), channel, 3, 3]. COT has one
    # channel; geometry has two. All arrays hold physical AGRI, raw log1p(COT),
    # and physical cosSOZ/cosRAA respectively.
    arrays = {}
    specs = {
        "history_agri_raw.npy": (2, 13),
        "history_geometry.npy": (2, 2),
        "history_cot_log1p.npy": (2, 1),
        "forecast_agri_raw.npy": (NLEADS, 13),
        "forecast_geometry.npy": (NLEADS, 2),
        "forecast_cot_log1p.npy": (NLEADS, 1),
    }
    for name, (time_n, channels) in specs.items():
        x = np.load(a.forecast_bank / name, mmap_mode="r", allow_pickle=False)
        expected = (len(bank_index), 2, time_n, channels, PATCH, PATCH)
        if x.shape != expected or x.dtype != np.float32:
            raise ValueError(f"{name}: expected float32 {expected}, got {x.dtype} {x.shape}")
        arrays[name] = x

    # Create split-local outputs. Training statistics are computed only after
    # all training inputs are materialized, then shared by A and AC.
    a.output.mkdir(parents=True)
    split_rows = {"train": np.flatnonzero(split_code == 0), "val": np.flatnonzero(split_code == 1)}
    row_files = {}
    counts = {}
    for split in SPLITS:
        ids = split_rows[split]
        d = a.output / split
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
        key_fields = ("split", "station", "sequence_index_within_split", "target_time_utc",
                      "init_time_utc", "lead_minutes", "ghi_row_index", "crosswalk_row_index")
        with keys_path.open("w", newline="", encoding="utf-8") as key_f:
            key_writer = csv.DictWriter(key_f, fieldnames=key_fields)
            key_writer.writeheader()
            for out_i, cross_i in enumerate(ids):
                sc, sq, st, li, gi = (int(expected_keys[name][cross_i]) for name in
                                      ("split_code", "sequence_index_within_split", "station_index", "lead_index", "ghi_row_index"))
                bi = bank_index.get((sc, sq))
                if bi is None:
                    raise ValueError(f"missing forecast sequence bank row {(sc, sq)}")
                lr = label_rows[gi]
                split_name = "train" if sc == 0 else "val"
                if lr["split"] != split_name or lr["station"].lower() != STATIONS[st]:
                    raise ValueError(f"label split/station mismatch at crosswalk row {cross_i}")
                lead_minutes = 15 * (li + 1)
                target_utc = lr["target_time_utc"]
                init_utc = (datetime.strptime(target_utc, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=lead_minutes)).strftime("%Y-%m-%d %H:%M:%S")
                row_features = []
                for offset in (-2, -1, 0):
                    is_hist, source = context_source(np.asarray([li]), offset)
                    hist = bool(is_hist[0]); ti = int(source[0])
                    if hist:
                        ag_name, ge_name, co_name = "history_agri_raw.npy", "history_geometry.npy", "history_cot_log1p.npy"
                    else:
                        ag_name, ge_name, co_name = "forecast_agri_raw.npy", "forecast_geometry.npy", "forecast_cot_log1p.npy"
                    ag = arrays[ag_name][bi, st, ti]
                    ge = arrays[ge_name][bi, st, ti]
                    co = arrays[co_name][bi, st, ti]
                    if not (np.isfinite(ag).all() and np.isfinite(ge).all() and np.isfinite(co).all()):
                        raise ValueError(f"non-finite model input at row {cross_i}, offset {offset}")
                    row_features.append((ag, ge, co))
                outputs["agri_raw.npy"][out_i] = np.stack([v[0] for v in row_features])
                outputs["geometry.npy"][out_i] = np.stack([v[1] for v in row_features])
                outputs["cot_log1p.npy"][out_i] = np.stack([v[2] for v in row_features])
                outputs["ghi.npy"][out_i] = float(lr["ghi"])
                outputs["clear.npy"][out_i] = float(lr["clear"])
                key_writer.writerow({"split": split_name, "station": STATIONS[st],
                                     "sequence_index_within_split": sq, "target_time_utc": target_utc,
                                     "init_time_utc": init_utc, "lead_minutes": lead_minutes,
                                     "ghi_row_index": gi, "crosswalk_row_index": int(cross_i)})
        for value in outputs.values():
            value.flush()
        row_files[split] = {name: sha(d / name) for name in
                            ("agri_raw.npy", "geometry.npy", "cot_log1p.npy", "ghi.npy", "clear.npy", "keys.csv")}
        counts[split] = count

    # Fit common input normalization on forecast instances in train only.
    tr = a.output / "train"
    train_agri = np.load(tr / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    train_cot = np.load(tr / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    ag_mean = np.empty(13, np.float64); ag_std = np.empty(13, np.float64)
    for c in range(13):
        v = np.asarray(train_agri[:, :, c], dtype=np.float64)
        ag_mean[c] = v.mean(); ag_std[c] = max(v.std(), 1e-6)
    c = np.asarray(train_cot, dtype=np.float64)
    cot_mean = float(c.mean()); cot_std = max(float(c.std()), 1e-6)
    meta = {
        "state": "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK",
        **provenance,
        "split_id": label_meta["split_id"], "sample_key": "(station,init_time_utc,target_time_utc,lead)",
        "rows": counts, "history_offsets_minutes": [-15, 0],
        "lead_minutes": list(range(15, 241, 15)),
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
        "ghi_label_pack_manifest_sha256": sha(a.ghi_pack / "pack_manifest.json"),
        "array_sha256": row_files,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False, "cpp_used_as_input": False,
    }
    (a.output / "pack_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": meta["state"], "rows": counts, "test_payload_opened": False,
                      "AGRI_norm_mean": meta["AGRI_mean"], "COT_norm_mean": cot_mean}, indent=2))


if __name__ == "__main__":
    main()
