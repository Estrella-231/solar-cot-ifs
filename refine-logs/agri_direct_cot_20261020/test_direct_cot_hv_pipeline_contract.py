#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def test_layout_helpers():
    pred = module("predict_direct_cot_hv", "predict_direct_cot_hv.py")
    agri = np.array([[[1.0, np.nan], [np.inf, 3.0]]], np.float32)
    normalized = pred.normalize_history_agri(agri, np.zeros((1, 1, 1), np.float32),
                                             np.ones((1, 1, 1), np.float32))
    assert np.array_equal(normalized, np.array([[[1.0, 0.0], [0.0, 3.0]]], np.float32))
    geo = np.empty((2, 2, 16, 3, 16, 16), np.float32)
    for bi in range(2):
        for si in range(2):
            for ti in range(16): geo[bi, si, ti] = 100 * bi + 10 * si + ti
    flat = pred.station_major_geometry(geo, 2)
    assert flat.shape == (4, 16, 3, 16, 16)
    assert flat[2, 7, 0, 0, 0] == 107 and flat[1, 3, 0, 0, 0] == 13
    raw = np.arange(4 * 1 * 16 * 16 * 16, dtype=np.float32).reshape(4, 1, 16, 16, 16)
    bank = pred.p_output_to_sequence_station(raw, 2)
    assert bank.shape == (2, 2, 16, 1, 16, 16)
    assert bank[1, 1, 5, 0, 0, 0] == raw[3, 0, 5, 0, 0]


def test_r_geometry_provenance_uses_pack_audit():
    pred = module("predict_direct_cot_hv", "predict_direct_cot_hv.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); pack = root / "r_pack"; pack.mkdir()
        geometry = root / "solar_geometry_corrected.py"
        geometry.write_text("# frozen corrected geometry source\n", encoding="utf-8")
        digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
        (pack / "pack_audit.json").write_text(json.dumps({
            "state": "PASS_U_ONLY_R_PACK", "train_rows_only": True,
            "test_payloads_read": 0, "buffer_payloads_read": 0,
            "geometry_source_sha256": digest,
        }), encoding="utf-8")
        audit = pred.validate_r_geometry_provenance(pack, geometry)
        assert audit["geometry_source_sha256"] == digest
        geometry.write_text("# changed\n", encoding="utf-8")
        try:
            pred.validate_r_geometry_provenance(pack, geometry)
        except ValueError as e:
            assert "geometry source hash mismatch" in str(e)
        else:
            raise AssertionError("geometry source change should fail provenance gate")


def test_hv_row_bank_join_contract():
    builder = module("build_direct_cot_forecast_row_bank", "build_direct_cot_forecast_row_bank.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inference, support, labels = root / "inf", root / "support", root / "labels"
        inference.mkdir(); support.mkdir(); labels.mkdir()
        seqs = {}
        for split, tag in (("h", "train"), ("v", "val")):
            sid = "hunan_202404011600" if split == "h" else "hunan_202404021600"
            stamps = [f"20240401{1600 + 15*i:04d}" for i in range(24)] if split == "h" else [f"20240402{1600 + 15*i:04d}" for i in range(24)]
            # Correct minute arithmetic for an hourly wrap is clearer through datetime.
            from datetime import datetime, timedelta
            t0 = datetime.strptime(stamps[0], "%Y%m%d%H%M")
            stamps = [(t0 + timedelta(minutes=15*i)).strftime("%Y%m%d%H%M") for i in range(24)]
            rels = [f"2024/202404/{s[:8]}/{s}.npy" for s in stamps]
            seqs[split] = {"seq_id": sid, "BJT_start": "2024-04-02 00:00:00", "split": tag,
                           "data_relpaths": "|".join(rels)}
            write_csv(support / f"{split}_sequences.csv", [seqs[split]], ["seq_id", "BJT_start", "split", "data_relpaths"])
            d = inference / split.upper(); d.mkdir()
            write_csv(d / "index.csv", [{"sequence_index_within_split": 0, "seq_id": sid, "BJT_start": ""}],
                      ["sequence_index_within_split", "seq_id", "BJT_start"])
            np.save(d / "agri_raw.npy", np.full((1, 2, 16, 13, 3, 3), 1, np.float32))
            np.save(d / "geometry_raw.npy", np.full((1, 2, 16, 2, 3, 3), 2, np.float32))
            np.save(d / "cot_simvp_r.npy", np.full((1, 2, 16, 1, 16, 16), 3, np.float32))
            np.save(d / "cot_p_c.npy", np.full((1, 2, 16, 1, 16, 16), 4, np.float32))
            np.save(d / "cot_p_xc.npy", np.full((1, 2, 16, 1, 16, 16), 5, np.float32))
        (inference / "inference_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE", "future_observed_agri_read": False,
            "cpp_read": False, "ghi_labels_read": False, "test_payload_opened": False,
            "buffer_payload_opened": False, "split_support_audit_sha256": "support-sha"}), encoding="utf-8")
        sequence_index = root / "sequence_index.csv"
        u_sid = "hunan_202404011545"
        write_csv(sequence_index, [
            {"split": "train", "sequence_index_within_split": 0, "seq_id": u_sid},
            {"split": "train", "sequence_index_within_split": 1, "seq_id": seqs["h"]["seq_id"]},
            {"split": "val", "sequence_index_within_split": 0, "seq_id": seqs["v"]["seq_id"]}],
            ["split", "sequence_index_within_split", "seq_id"])
        for split, sid in (("train", seqs["h"]["seq_id"]), ("val", seqs["v"]["seq_id"])):
            d = labels / split; d.mkdir(parents=True)
            keys = []
            for station in ("sili", "zhujia"):
                if split == "train":
                    # U rows appear in the official outer-train labels but are
                    # intentionally not part of H-prediction GHI-head training.
                    keys.append({"split": split, "station": station, "sequence_index_within_split": "0",
                        "target_time_utc": "2024-04-01 18:00:00", "init_time_utc": "2024-04-01 17:45:00",
                        "lead_minutes": "15", "ghi_row_index": "0", "crosswalk_row_index": "0"})
                keys.append({"split": split, "station": station,
                    "sequence_index_within_split": "1" if split == "train" else "0",
                    "target_time_utc": ("2024-04-01" if split == "train" else "2024-04-02") + " 18:00:00",
                    "init_time_utc": ("2024-04-01" if split == "train" else "2024-04-02") + " 17:45:00",
                    "lead_minutes": "15", "ghi_row_index": "0", "crosswalk_row_index": "0"})
            write_csv(d / "keys.csv", keys, list(keys[0]))
            if split == "train":
                np.save(d / "ghi.npy", np.array([10, 100, 20, 200], np.float32))
                np.save(d / "clear.npy", np.array([500, 500, 500, 500], np.float32))
            else:
                np.save(d / "ghi.npy", np.array([100, 200], np.float32))
                np.save(d / "clear.npy", np.array([500, 500], np.float32))
        (labels / "pack_manifest.json").write_text("{}", encoding="utf-8")
        coverage = root / "coverage.json"
        coverage.write_text(json.dumps({"splits": {
            "H": {"common_scoreable_rows_after_station_day_exclusions": 2},
            "V": {"common_scoreable_rows_after_station_day_exclusions": 2}}}), encoding="utf-8")
        exclusions = root / "exclusions.json"
        exclusions.write_text(json.dumps({"date_timezone": "Asia/Shanghai", "exclusions": []}), encoding="utf-8")
        out = root / "bank"
        builder.build(SimpleNamespace(inference=inference, support=support, sequence_index=sequence_index,
            labels=labels, coverage_audit=coverage, exclusions=exclusions, output=out))
        got = np.load(out / "sili/train/cot_raw/p_c.npy", allow_pickle=False)
        assert got.shape == (1, 1, 3, 3) and np.all(got == 4)
        assert np.array_equal(np.load(out / "sili/train/ghi.npy", allow_pickle=False), np.array([100], np.float32))
        assert np.array_equal(np.load(out / "zhujia/train/ghi.npy", allow_pickle=False), np.array([200], np.float32))
        assert len((out / "zhujia/val/keys.jsonl").read_text().splitlines()) == 1
        bank = json.loads((out / "bank_manifest.json").read_text())
        assert bank["state"] == "COMPLETE_DIRECT_COT_FORECAST_ROW_BANK"
        assert bank["test_payload_opened"] is False and bank["buffer_payload_opened"] is False
        pack_out = root / "ghi_pack"
        subprocess.run([sys.executable, str(HERE / "build_direct_cot_single_frame_ghi_pack.py"),
                        "--bank", str(out), "--output", str(pack_out)], check=True, capture_output=True, text=True)
        packed = json.loads((pack_out / "pack_manifest.json").read_text())
        assert packed["state"] == "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_PACK"
        assert packed["counts"] == {"sili/train": 1, "sili/val": 1, "zhujia/train": 1, "zhujia/val": 1}
        assert packed["test_payload_opened"] is False and packed["buffer_payload_opened"] is False


if __name__ == "__main__":
    test_layout_helpers()
    test_r_geometry_provenance_uses_pack_audit()
    test_hv_row_bank_join_contract()
    print("PASS direct-COT H/V pipeline contract")
