#!/usr/bin/env python3
"""Join causal H/V COT forecasts to the frozen official GHI train/validation rows.

This builder reads only H/V prediction arrays, their sequence indices, and the
existing train/validation GHI labels. It has no test/buffer payload path.
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
SPLITS = (("H", "train", "h"), ("V", "val", "v"))
ARMS = ("none", "simvp_r", "p_c", "p_xc")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_official_sequence_index(path: Path) -> dict[str, dict[int, str]]:
    rows = read_csv(path)
    out: dict[str, dict[int, str]] = {"train": {}, "val": {}}
    for row in rows:
        split = row["split"]
        if split not in out:
            continue
        idx, seq_id = int(row["sequence_index_within_split"]), row["seq_id"]
        if idx in out[split] or seq_id in out[split].values():
            raise ValueError(f"duplicate sequence index or ID in official {split} index: {idx}/{seq_id}")
        out[split][idx] = seq_id
    for split in out:
        if not out[split]:
            raise ValueError(f"empty official {split} sequence index")
    return out


def load_prediction_index(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    out = {}
    for row in rows:
        i = int(row["sequence_index_within_split"])
        sid = row["seq_id"]
        if sid in out or i != len(out):
            raise ValueError(f"prediction index is duplicate or non-contiguous: {path}")
        out[sid] = i
    return out


def utc_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def payload_for_key(key: dict[str, str], seq: dict[str, str], lead_idx: int) -> dict:
    paths = seq["data_relpaths"].split("|")
    if len(paths) != 24:
        raise ValueError(f"sequence lacks 24 frames: {seq['seq_id']}")
    init = utc_time(key["init_time_utc"])
    target = utc_time(key["target_time_utc"])
    if datetime.strptime(Path(paths[7]).stem, "%Y%m%d%H%M") != init:
        raise ValueError(f"history-end/init mismatch for {seq['seq_id']}")
    if datetime.strptime(Path(paths[8 + lead_idx]).stem, "%Y%m%d%H%M") != target:
        raise ValueError(f"forecast target/lead path mismatch for {seq['seq_id']} lead={lead_idx + 1}")
    if int((target - init).total_seconds() // 60) != int(key["lead_minutes"]):
        raise ValueError(f"lead/time mismatch in label key: {key}")
    return {"paths": paths, "init": init, "target": target}


def build(args) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    inf = json.loads((args.inference / "inference_manifest.json").read_text(encoding="utf-8"))
    if inf.get("state") != "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE":
        raise ValueError("causal H/V inference is incomplete")
    for name in ("future_observed_agri_read", "cpp_read", "ghi_labels_read", "test_payload_opened", "buffer_payload_opened"):
        if inf.get(name) is not False:
            raise ValueError(f"inference receipt violates {name}=false")

    exclusion_obj = json.loads(args.exclusions.read_text(encoding="utf-8"))
    if exclusion_obj.get("date_timezone") != "Asia/Shanghai":
        raise ValueError("station-day exclusion timezone mismatch")
    exclusions = {(r["station"].lower(), r["bjt_date"]) for r in exclusion_obj["exclusions"]}

    # GHI label indices are defined by the frozen SimVP→R bank's global
    # split index, not by fold-1's U-only/H-heldout manifest. Crosswalk first,
    # then retain only the predeclared H or V support. In particular, outer-
    # train rows mapped to U are expected and must be skipped for H training.
    official_index = load_official_sequence_index(args.sequence_index)
    support_rows: dict[str, dict[str, dict[str, str]]] = {"h": {}, "v": {}}
    pred_indices: dict[str, dict[str, int]] = {}
    for split in ("h", "v"):
        support_rows[split] = {r["seq_id"]: r for r in read_csv(args.support / f"{split}_sequences.csv")}
        if len(support_rows[split]) != len(read_csv(args.support / f"{split}_sequences.csv")):
            raise ValueError(f"duplicate sequence IDs in {split.upper()} support")
        pred_indices[split] = load_prediction_index(args.inference / split.upper() / "index.csv")
        if set(pred_indices[split]) != set(support_rows[split]):
            raise ValueError(f"{split.upper()} prediction index does not cover exact support")

    # The historical GHI keys refer to the frozen fold-1 sequence manifest.
    # Crosswalk to seq_id first, then to this run's complete H/V support rows.
    exclusions_by_split = {"train": set(), "val": set()}
    result_counts = {}
    source_hashes: dict[str, str] = {}
    args.output.mkdir(parents=True)
    prediction_arrays = {}
    for split in ("H", "V"):
        d = args.inference / split
        prediction_arrays[split] = {
            "agri": np.load(d / "agri_raw.npy", mmap_mode="r", allow_pickle=False),
            "geometry": np.load(d / "geometry_raw.npy", mmap_mode="r", allow_pickle=False),
            "simvp_r": np.load(d / "cot_simvp_r.npy", mmap_mode="r", allow_pickle=False),
            "p_c": np.load(d / "cot_p_c.npy", mmap_mode="r", allow_pickle=False),
            "p_xc": np.load(d / "cot_p_xc.npy", mmap_mode="r", allow_pickle=False),
        }
    for region, label_split, pred_split in SPLITS:
        label_dir = args.labels / label_split
        keys_path, ghi_path, clear_path = label_dir / "keys.csv", label_dir / "ghi.npy", label_dir / "clear.npy"
        keys = read_csv(keys_path)
        ghi = np.load(ghi_path, mmap_mode="r", allow_pickle=False)
        clear = np.load(clear_path, mmap_mode="r", allow_pickle=False)
        if ghi.dtype != np.float32 or clear.dtype != np.float32 or ghi.shape != (len(keys),) or clear.shape != ghi.shape:
            raise ValueError(f"invalid official {label_split} GHI/clear label arrays")
        if not np.isfinite(ghi).all() or not np.isfinite(clear).all() or (ghi < 0).any() or (clear <= 0).any():
            raise ValueError(f"invalid official {label_split} labels")

        all_sequence_ids = official_index[label_split]
        current_support = support_rows[pred_split]
        current_indices = pred_indices[pred_split]
        row_sel = []
        for ri, key in enumerate(keys):
            idx = int(key["sequence_index_within_split"])
            seq_id = all_sequence_ids.get(idx)
            if seq_id is None:
                raise ValueError(f"official {label_split} label sequence index absent from frozen index: {idx}")
            if seq_id not in current_support:
                continue
            station = key["station"].lower()
            if station not in STATIONS:
                raise ValueError(f"unexpected station in official key: {station}")
            lead_minutes = int(key["lead_minutes"])
            if lead_minutes not in range(15, 241, 15):
                raise ValueError(f"invalid lead: {lead_minutes}")
            lead_idx = lead_minutes // 15 - 1
            payload_for_key(key, current_support[seq_id], lead_idx)
            target_bjt = (utc_time(key["target_time_utc"]) + timedelta(hours=8)).date().isoformat()
            if (station, target_bjt) in exclusions:
                exclusions_by_split[label_split].add(ri)
                continue
            row_sel.append((ri, key, seq_id, station, lead_idx, current_indices[seq_id]))

        # Coverage audit is metadata-only and freezes exact scored row counts.
        expected = json.loads(args.coverage_audit.read_text(encoding="utf-8"))["splits"][region]
        expected_rows = int(expected["common_scoreable_rows_after_station_day_exclusions"])
        if len(row_sel) != expected_rows:
            raise ValueError(f"{region} scoreable row count {len(row_sel)} != frozen audit {expected_rows}")

        per_station = {}
        for station in STATIONS:
            sel = [r for r in row_sel if r[3] == station]
            if not sel:
                raise ValueError(f"no {station}/{label_split} rows")
            d = args.output / station / label_split
            (d / "cot_raw").mkdir(parents=True)
            n = len(sel)
            agri = np.empty((n, 13, 3, 3), np.float32)
            geometry = np.empty((n, 2, 3, 3), np.float32)
            cots = {arm: np.zeros((n, 1, 3, 3), np.float32) for arm in ARMS}
            label_ghi = np.empty(n, np.float32)
            label_clear = np.empty(n, np.float32)
            out_keys = []
            for oi, (ri, key, seq_id, _, lead_idx, si) in enumerate(sel):
                pred = prediction_arrays[pred_split.upper()]
                if si < 0 or si >= len(pred_indices[pred_split]):
                    raise ValueError("bad prediction sequence index")
                station_idx = STATIONS.index(station)
                agri[oi] = pred["agri"][si, station_idx, lead_idx]
                geometry[oi] = pred["geometry"][si, station_idx, lead_idx]
                for arm in ("simvp_r", "p_c", "p_xc"):
                    cots[arm][oi] = pred[arm][si, station_idx, lead_idx, :, 7:10, 7:10]
                label_ghi[oi] = ghi[ri]
                label_clear[oi] = clear[ri]
                out_keys.append(dict(key))
            np.save(d / "agri_raw.npy", agri, allow_pickle=False)
            np.save(d / "geometry_raw.npy", geometry, allow_pickle=False)
            np.save(d / "ghi.npy", label_ghi, allow_pickle=False)
            np.save(d / "clear.npy", label_clear, allow_pickle=False)
            for arm in ARMS:
                np.save(d / "cot_raw" / f"{arm}.npy", cots[arm], allow_pickle=False)
            with (d / "keys.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                for key in out_keys:
                    f.write(json.dumps(key, sort_keys=True, ensure_ascii=False) + "\n")
            per_station[f"{station}/{label_split}"] = n
            for filename in ("agri_raw.npy", "geometry_raw.npy", "ghi.npy", "clear.npy", "keys.jsonl",
                             *(f"cot_raw/{arm}.npy" for arm in ARMS)):
                rel = f"{station}/{label_split}/{filename}"
                source_hashes[rel] = sha(d / filename)
        result_counts.update(per_station)

    if not (args.output / "sili" / "train" / "keys.jsonl").exists():
        raise RuntimeError("H bank not written")
    bank = {
        "state": "COMPLETE_DIRECT_COT_FORECAST_ROW_BANK", "arms": list(ARMS), "stations": list(STATIONS),
        "counts": result_counts, "train_partition": "H", "validation_partition": "V",
        "normalization_fit_partition": "H", "agri_source": "fold1-SimVP-U-only-forecast",
        "split_support_audit_sha256": inf["split_support_audit_sha256"],
        "source_inference_manifest_sha256": sha(args.inference / "inference_manifest.json"),
        "official_label_pack_manifest_sha256": sha(args.labels / "pack_manifest.json"),
        "official_sequence_index_sha256": sha(args.sequence_index),
        "coverage_audit_sha256": sha(args.coverage_audit),
        "station_day_exclusions_sha256": sha(args.exclusions), "artifact_sha256": source_hashes,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False, "cpp_used_as_input": False,
        "ghi_labels_read_after_inference": True,
        "label_policy": "official frozen train/val label rows crosswalked by frozen sequence index, intersected with predeclared H/V support, common four-arm rows, then shared station-day exclusion",
        "test_and_buffer_payloads_opened": False,
    }
    atomic_json(args.output / "bank_manifest.json", bank)
    print(json.dumps({"state": bank["state"], "counts": result_counts,
                      "test_payload_opened": False, "buffer_payload_opened": False}, indent=2))
    return bank


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("inference", "support", "sequence-index", "labels", "coverage-audit", "exclusions", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
