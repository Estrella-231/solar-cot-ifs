#!/usr/bin/env python3
"""Join forecast-AGRI -> frozen-R COT with train/val 15-min GHI labels.

Each row is a unique (station, init_time, target_time) forecast example. The
pack is distinct from the same-time real-AGRI pack and never reads test/buffer
payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
EXPECTED_SIMVP_INDEX_SHA = "4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62"
EXPECTED_SEQUENCES = {"train": 4086, "val": 407}
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stage_lineage(stage_report_path: Path, forecast_bank_report: dict) -> dict:
    stage = json.loads(stage_report_path.read_text(encoding="utf-8"))
    if stage.get("status") != "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED":
        raise ValueError("Stage report is not train/validation-only")
    if stage.get("source_index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Stage report is not bound to the frozen Hubei sequence index")
    # The forecast bank preserves the canonical index. The stage index is a
    # path-rewritten derivative used only to access local payloads.
    if forecast_bank_report.get("index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Forecast bank is not bound to the frozen canonical index")
    if forecast_bank_report.get("canonical_index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Forecast bank omits or mismatches the canonical index lineage")
    if forecast_bank_report.get("source_stage_report_sha256") != sha(stage_report_path):
        raise ValueError("Forecast bank was produced from a different stage report")
    if forecast_bank_report.get("source_stage_index_sha256") != stage.get("stage_index_sha256"):
        raise ValueError("Forecast bank path-remapped stage index lineage mismatch")
    if forecast_bank_report.get("source_stage_manifest_sha256") != stage.get("stage_manifest_sha256"):
        raise ValueError("Forecast bank stage-manifest lineage mismatch")
    if forecast_bank_report.get("stage_root") != stage.get("stage_root"):
        raise ValueError("Forecast bank stage root mismatch")
    if stage.get("sequences") != EXPECTED_SEQUENCES or stage.get("test_sequences") != 0:
        raise ValueError("Unexpected stage train/validation sequence counts")
    if stage.get("test_or_buffer_payloads_staged") != 0:
        raise ValueError("Test/buffer payloads were staged")
    if sha(Path(stage["source_index"])) != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Frozen canonical sequence-index checksum mismatch")
    if sha(Path(stage["stage_index"])) != stage.get("stage_index_sha256"):
        raise ValueError("Path-rewritten stage index checksum mismatch")
    return stage


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_time(text: str) -> str:
    value = text.strip().replace("-", "").replace(":", "").replace(" ", "")
    if len(value) == 14:
        value = value[:12]
    if len(value) != 12 or not value.isdigit():
        raise ValueError(f"Invalid BJT timestamp {text!r}")
    return value


def parse_time(text: str) -> datetime:
    return datetime.strptime(normalize_time(text), "%Y%m%d%H%M")


def build_sample_index(forecast_rows: list[dict[str, str]], split: str,
                       split_by_date: dict[str, str], labels: dict[tuple[str, str], tuple[str, float, float]]) -> list[tuple[int, int, str, str, int, float, float]]:
    """Return (sequence_i, lead_i, station, target, lead_min, GHI, clear)."""
    out = []
    seen = set()
    for sequence_i, row in enumerate(forecast_rows):
        if row.get("split") != split or int(row.get("sample_index", "-1")) != sequence_i:
            raise ValueError(f"Noncanonical sequence key at {split}/{sequence_i}")
        init_text = normalize_time(row["init_time_bjt"])
        init = parse_time(init_text)
        targets = row.get("target_times_bjt", "").split("|")
        if len(targets) != 16:
            raise ValueError(f"Expected 16 target times at {split}/{sequence_i}")
        for lead_i, target_raw in enumerate(targets):
            target_text = normalize_time(target_raw)
            lead_min = 15 * (lead_i + 1)
            target = parse_time(target_text)
            if target != init + timedelta(minutes=lead_min):
                raise ValueError(f"Lead/time mismatch at {split}/{sequence_i}/{lead_min}: {init_text}->{target_text}")
            # Midnight crossing is valid when every forecast target remains in
            # the same frozen split; date assignment below is checked per lead.
            if split_by_date.get(target.strftime("%Y-%m-%d")) != split:
                raise ValueError(f"Target date does not match frozen split at {target_text}")
            for station in STATIONS:
                label = labels.get((station, target_text))
                if label is None:
                    continue
                label_split, ghi, clear = label
                if label_split != split:
                    raise ValueError(f"Label/forecast split mismatch at {station}/{target_text}")
                key = (station, init_text, target_text)
                if key in seen:
                    raise ValueError(f"Duplicate forecast sample key {key}")
                seen.add(key)
                out.append((sequence_i, lead_i, station, target_text, lead_min, ghi, clear))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", type=Path, required=True)
    p.add_argument("--stage-report", type=Path, required=True,
                   help="Audited path-rewrite lineage for the frozen SimVP sequence index")
    p.add_argument("--forecast-cot-bank", type=Path, required=True)
    p.add_argument("--oof-train-bank", type=Path,
                   help="Audited five-fold train-only OOF S->R bank; when supplied, only its train tensors are used")
    p.add_argument("--labels", type=Path, required=True,
                   help="ghi15_train_val_daylight_clear.csv; train/val only")
    p.add_argument("--split", type=Path, required=True,
                   help="frozen day_assignment.csv")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.output}")
    simvp_report_path = a.forecast_bank / "BANK_REPORT.json"
    cot_report_path = a.forecast_cot_bank / "BANK_REPORT.json"
    simvp_report = json.loads(simvp_report_path.read_text(encoding="utf-8"))
    cot_report = json.loads(cot_report_path.read_text(encoding="utf-8"))
    if simvp_report.get("state") != "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK" or simvp_report.get("test_payloads_read") != 0:
        raise ValueError("Forecast AGRI source must be merged train/validation bank")
    stage = validate_stage_lineage(a.stage_report, simvp_report)
    for split, expected_n in EXPECTED_SEQUENCES.items():
        if int(simvp_report.get("splits", {}).get(split, {}).get("sequences", -1)) != expected_n:
            raise ValueError(f"Unexpected forecast-bank sequence count for {split}")
    if cot_report.get("state") != "MERGED_SIMVP_FORECAST_COT_TRAINVAL_BANK_TEST_CLOSED" or cot_report.get("test_payloads_read") != 0:
        raise ValueError("Forecast COT source must be merged train/validation bank")
    if cot_report.get("ghi_labels_read") is not False:
        raise ValueError("Forecast COT bank must not have read GHI labels")
    if cot_report.get("canonical_source_index_sha256") != EXPECTED_SIMVP_INDEX_SHA:
        raise ValueError("Forecast COT canonical index lineage mismatch")
    if cot_report.get("source_stage_index_sha256") != stage.get("stage_index_sha256"):
        raise ValueError("Forecast COT stage index lineage mismatch")
    if cot_report.get("source_stage_report_sha256") != sha(a.stage_report):
        raise ValueError("Forecast COT came from a different stage report")
    if cot_report.get("source_forecast_bank_report_sha256") != sha(simvp_report_path):
        raise ValueError("Forecast-COT does not derive from this forecast-AGRI bank")
    oof_report_path = None
    oof_report = None
    if a.oof_train_bank is not None:
        oof_report_path = a.oof_train_bank / "BANK_REPORT.json"
        oof_report = json.loads(oof_report_path.read_text(encoding="utf-8"))
        if (oof_report.get("state") != "MERGED_HUBEI_SIMVP_FORECAST_GHI_OOF_TRAIN_BANK_TEST_CLOSED"
                or oof_report.get("train_forecast_predictions_are_oof") is not True
                or oof_report.get("oof_fold_count") != 5
                or oof_report.get("canonical_index_sha256") != EXPECTED_SIMVP_INDEX_SHA
                or oof_report.get("test_payloads_read") != 0
                or oof_report.get("test_or_buffer_rows") != 0
                or oof_report.get("ghi_labels_read") is not False
                or oof_report.get("train_sequences") != EXPECTED_SEQUENCES["train"]):
            raise ValueError("OOF train bank does not pass the frozen five-fold train-only contract")
        fold_counts = oof_report.get("fold_sequence_counts", {})
        if set(fold_counts) != {"0", "1", "2", "3", "4"} or sum(int(v) for v in fold_counts.values()) != EXPECTED_SEQUENCES["train"]:
            raise ValueError("OOF train bank fold counts are incomplete or inconsistent")
        oof_train_dir = a.oof_train_bank / "train"
        for name in ("pred_agri_patch.npy", "future_geometry_patch.npy", "forecast_cot.npy"):
            if oof_report.get("arrays_sha256", {}).get(name) != sha(oof_train_dir / name):
                raise ValueError(f"OOF train bank array checksum mismatch for {name}")
        if (oof_report.get("keys_sha256") != sha(oof_train_dir / "keys.csv")
                or oof_report.get("heldout_fold_sha256") != sha(oof_train_dir / "heldout_fold.csv")):
            raise ValueError("OOF train bank key/fold checksum mismatch")
    if sha(a.split) != EXPECTED_SPLIT_SHA:
        raise ValueError("Frozen Hubei split checksum mismatch")

    split_rows = read_csv(a.split)
    split_by_date = {r["date_BJT"].replace("-", ""): r["split"] for r in split_rows}
    # Convert to YYYY-MM-DD keys for date lookup.
    split_by_date = {datetime.strptime(k, "%Y%m%d").strftime("%Y-%m-%d"): v for k, v in split_by_date.items()}
    labels: dict[tuple[str, str], tuple[str, float, float]] = {}
    for row in read_csv(a.labels):
        split = row.get("split", "")
        if split not in ("train", "val"):
            raise ValueError("Label file must contain only train/validation rows")
        if row.get("valid_daylight_clear_sky_label") not in ("1", "1.0", "True", "true"):
            continue
        station = row["station"]
        if station not in STATIONS:
            raise ValueError(f"Unknown station {station}")
        target = normalize_time(row["target_time_BJT"])
        if split_by_date.get(datetime.strptime(target, "%Y%m%d%H%M").strftime("%Y-%m-%d")) != split:
            raise ValueError(f"Label date/split mismatch {station}/{target}")
        if not row.get("ghi15_raw") or not row.get("clear_sky_ghi_15min_wm2"):
            raise ValueError(f"Valid label row lacks GHI/clear-sky value {station}/{target}")
        ghi, clear = float(row["ghi15_raw"]), float(row["clear_sky_ghi_15min_wm2"])
        if not np.isfinite(ghi) or ghi < 0 or not np.isfinite(clear) or clear <= 0:
            raise ValueError(f"Invalid GHI/clear-sky value {station}/{target}")
        key = (station, target)
        if key in labels:
            raise ValueError(f"Duplicate station/target label {key}")
        labels[key] = (split, ghi, clear)

    # Validate canonical bank keys and construct matching label keys.
    all_samples = []
    source_arrays = {}
    for split in ("train", "val"):
        if split == "train" and a.oof_train_bank is not None:
            sdir = cdir = a.oof_train_bank / "train"
            simvp_keys = read_csv(sdir / "keys.csv")
            cot_keys = simvp_keys
            expected = EXPECTED_SEQUENCES["train"]
            agri = np.load(sdir / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
            geometry = np.load(sdir / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
            cot = np.load(sdir / "forecast_cot.npy", mmap_mode="r", allow_pickle=False)
            fold_rows = read_csv(sdir / "heldout_fold.csv")
            if len(fold_rows) != expected or any(int(r["sample_index"]) != i or not 0 <= int(r["heldout_fold"]) < 5
                                                  for i, r in enumerate(fold_rows)):
                raise ValueError("OOF train bank fold assignments are missing/noncanonical")
        else:
            sdir, cdir = a.forecast_bank / split, a.forecast_cot_bank / split
            simvp_keys, cot_keys = read_csv(sdir / "keys.csv"), read_csv(cdir / "keys.csv")
            if simvp_keys != cot_keys:
                raise ValueError(f"SimVP/COT sequence keys differ for {split}")
            expected = int(simvp_report["splits"][split]["sequences"])
            if len(simvp_keys) != expected or int(cot_report["splits"][split]["sequences"]) != expected:
                raise ValueError(f"Sequence count mismatch for {split}")
            agri = np.load(sdir / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
            geometry = np.load(sdir / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
            cot = np.load(cdir / "forecast_cot.npy", mmap_mode="r", allow_pickle=False)
        if agri.shape != (expected, 16, 13, 16, 16) or geometry.shape != (expected, 16, 3, 16, 16) or cot.shape != (expected, 16, 1, 16, 16):
            raise ValueError(f"Unexpected source array shape for {split}")
        samples = build_sample_index(simvp_keys, split, split_by_date, labels)
        all_samples.extend((split, *sample) for sample in samples)
        source_arrays[split] = (agri, geometry, cot, simvp_keys)
        print(f"INDEXED {split} forecast examples={len(samples)}", flush=True)
    if not all_samples:
        raise RuntimeError("No common forecast/GHI train-val samples")

    a.output.mkdir(parents=True, exist_ok=False)
    n = len(all_samples)
    agri_out = np.lib.format.open_memmap(a.output / "agri_raw.npy", mode="w+", dtype=np.float32, shape=(n, 13, 3, 3))
    geo_out = np.lib.format.open_memmap(a.output / "geometry.npy", mode="w+", dtype=np.float32, shape=(n, 2, 3, 3))
    cot_out = np.lib.format.open_memmap(a.output / "cot.npy", mode="w+", dtype=np.float32, shape=(n, 1, 3, 3))
    ghi_out = np.lib.format.open_memmap(a.output / "ghi_wm2.npy", mode="w+", dtype=np.float32, shape=(n,))
    clear_out = np.lib.format.open_memmap(a.output / "clear_sky_wm2.npy", mode="w+", dtype=np.float32, shape=(n,))
    columns = ("index", "split", "station", "init_time_bjt", "target_time_bjt", "lead_minutes", "ghi_wm2", "clear_sky_wm2", "source_sequence_index")
    with (a.output / "rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for out_i, (split, seq_i, lead_i, station, target, lead_min, ghi, clear) in enumerate(all_samples):
            agri, geometry, cot, keys = source_arrays[split]
            srow = keys[seq_i]
            init = normalize_time(srow["init_time_bjt"])
            r, c = CENTERS[station]
            sl = np.s_[r - 1:r + 2, c - 1:c + 2]
            # Slice the spatial patch after taking the channel-first view. Mixing
            # integer channel indexing with two slice indices in one NumPy
            # expression can move the advanced-indexed dimensions to the front.
            ag = np.asarray(agri[seq_i, lead_i][:, sl[0], sl[1]], dtype=np.float32)
            ge = np.asarray(geometry[seq_i, lead_i][:2, sl[0], sl[1]], dtype=np.float32)
            co = np.asarray(cot[seq_i, lead_i][0, sl[0], sl[1]], dtype=np.float32)[None]
            if ag.shape != (13, 3, 3) or ge.shape != (2, 3, 3) or co.shape != (1, 3, 3):
                raise ValueError(f"Station patch has wrong shape at {station}/{init}/{target}")
            if not (np.isfinite(ag).all() and np.isfinite(ge).all() and np.isfinite(co).all() and (co >= 0).all()):
                raise ValueError(f"Invalid forecast inputs at {station}/{init}/{target}")
            agri_out[out_i] = ag; geo_out[out_i] = ge; cot_out[out_i] = co
            ghi_out[out_i] = ghi; clear_out[out_i] = clear
            writer.writerow({"index": out_i, "split": split, "station": station,
                             "init_time_bjt": init, "target_time_bjt": target,
                             "lead_minutes": lead_min, "ghi_wm2": ghi,
                             "clear_sky_wm2": clear, "source_sequence_index": seq_i})
            if (out_i + 1) % 20000 == 0 or out_i + 1 == n:
                print(f"PACKED {out_i+1}/{n}", flush=True)
    for arr in (agri_out, geo_out, cot_out, ghi_out, clear_out):
        arr.flush()
    del agri_out, geo_out, cot_out, ghi_out, clear_out

    rows = read_csv(a.output / "rows.csv")
    by = Counter((r["split"], r["station"], int(r["lead_minutes"])) for r in rows)
    arrays = {name: a.output / name for name in ("agri_raw.npy", "geometry.npy", "cot.npy", "ghi_wm2.npy", "clear_sky_wm2.npy")}
    report = {
        "state": "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED",
        "canonical_source_index_sha256": EXPECTED_SIMVP_INDEX_SHA,
        "source_stage_index_sha256": stage["stage_index_sha256"],
        "source_stage_report_sha256": sha(a.stage_report),
        "samples": n,
        "samples_by_split_station": {sp: {st: sum(r["split"] == sp and r["station"] == st for r in rows) for st in STATIONS} for sp in ("train", "val")},
        "samples_by_split_station_lead": {f"{sp}/{st}/{lead}": count for (sp, st, lead), count in sorted(by.items())},
        "same_keys_all_arms": True,
        "input_mode": "forecast",
        "feature_source": "SimVP forecast AGRI -> frozen R forecast COT",
        "forecast_leads_min": list(range(15, 241, 15)),
        "target_time_basis": "BJT interval end; target=init+lead",
        "label": "15-min right-closed GHI window; original engineering W/m2 operating convention, sensor/export calibration metadata unverified",
        "split_sha256": EXPECTED_SPLIT_SHA,
        "simvp_index_sha256": simvp_report["index_sha256"],
        "simvp_checkpoint_sha256": simvp_report["checkpoint_sha256"],
        "forecast_cot_bank_report_sha256": sha(cot_report_path),
        "frozen_R_checkpoint_sha256": cot_report["frozen_R_checkpoint_sha256"],
        "train_forecast_inputs_are_oof": a.oof_train_bank is not None,
        "oof_fold_count": 5 if a.oof_train_bank is not None else 0,
        "oof_train_bank_report_sha256": sha(oof_report_path) if oof_report_path is not None else "",
        "oof_fold_source_lineage": oof_report.get("fold_sources") if oof_report is not None else None,
        "validation_forecast_source": "frozen_full_train_S_R",
        "normalization": "not applied here; trainer fits station-specific AGRI and log1p(COT) statistics using train rows only; geometry physical",
        "test_and_buffer_values_read": 0,
        "source_array_shapes": {name: list(np.load(path, mmap_mode="r", allow_pickle=False).shape) for name, path in arrays.items()},
        "array_sha256": {name: sha(path) for name, path in arrays.items()},
        "rows_sha256": sha(a.output / "rows.csv"),
        "samples_distinctness": "unique (station, init_time_bjt, target_time_bjt); repeated targets at different init/lead remain distinct forecast cases",
    }
    (a.output / "pack_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "samples": n,
                      "samples_by_split_station": report["samples_by_split_station"],
                      "test_and_buffer_values_read": 0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
