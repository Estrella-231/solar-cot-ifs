"""Synthetic causal-window and packing witness; no project payload is read."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from build_future_ghi_head_pack import context_source, history_context_mask, sha


def test_context_boundary():
    lead = np.asarray([0, 1, 15], dtype=np.int16)
    expected = [([True, True, False], [0, 1, 0]), ([True, False, False], [1, 0, 1]),
                ([False, False, False], [13, 14, 15])]
    for i, (history_flags, indices) in enumerate(expected):
        flags, ii = [], []
        for offset in (-2, -1, 0):
            hist, idx = context_source(lead[i:i+1], offset)
            flags.append(bool(hist[0]))
            ii.append(int(idx[0]))
            if not hist[0]:
                assert int(idx[0]) <= int(lead[i])
        assert flags == history_flags
        assert ii == indices
    assert np.array_equal(history_context_mask(np.asarray([15, 30, 45, 240])),
                          np.asarray([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32))


def test_synthetic_pack_end_to_end():
    script = Path(__file__).with_name("build_future_ghi_head_pack.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); bank = root / "bank"; pack = root / "ghi"; out = root / "out"
        bank.mkdir(); pack.mkdir()
        (bank / "bank_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK", "test_payload_opened": False,
            "future_observed_agri_read": False, "crosswalk_row_count_expected": 2,
            "in_sample_train_forecasts": False, "oof_fold_count": 5,
            "oof_train_sequences": 1,
            "oof_fold_model_hashes": {str(i): "0" * 64 for i in range(5)},
            "forecast_source": "five-fold OOF SimVP predictions",
        }))
        (pack / "pack_manifest.json").write_text(json.dumps({
            "state": "COMPLETE", "test_payload_opened": False,
            "split_id": "synthetic",
        }))
        (bank / "index.csv").write_text("split,sequence_index_within_split\ntrain,0\nval,0\n")
        history_agri = np.zeros((2, 2, 2, 13, 3, 3), np.float32)
        history_geometry = np.zeros((2, 2, 2, 2, 3, 3), np.float32)
        history_cot = np.zeros((2, 2, 2, 1, 3, 3), np.float32)
        for value in (history_agri, history_geometry, history_cot):
            value[:, :, 0] = 1  # init-15 min
            value[:, :, 1] = 3  # init
        np.save(bank / "history_agri_raw.npy", history_agri)
        np.save(bank / "history_geometry.npy", history_geometry)
        np.save(bank / "history_cot_log1p.npy", history_cot)
        np.save(bank / "forecast_agri_raw.npy", np.full((2, 2, 16, 13, 3, 3), 2, np.float32))
        np.save(bank / "forecast_geometry.npy", np.full((2, 2, 16, 2, 3, 3), 2, np.float32))
        np.save(bank / "forecast_cot_log1p.npy", np.full((2, 2, 16, 1, 3, 3), 2, np.float32))
        with (pack / "rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "station", "target_time_utc", "ghi", "clear", "kt"])
            w.writeheader()
            w.writerow(dict(split="train", station="sili", target_time_utc="2024-01-01 00:15:00", ghi=100, clear=500, kt=.2))
            w.writerow(dict(split="val", station="zhujia", target_time_utc="2024-02-01 00:30:00", ghi=120, clear=600, kt=.2))
        (root / "cross.json").write_text(json.dumps({"split_mismatch_rows": 0}))
        np.savez(root / "keys.npz", split_code=np.asarray([0, 1], np.uint8),
                 sequence_index_within_split=np.asarray([0, 0], np.int32),
                 station_index=np.asarray([0, 1], np.uint8), lead_index=np.asarray([0, 1], np.uint8),
                 ghi_row_index=np.asarray([0, 1], np.int32))
        manifest_path = bank / "bank_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        array_names = ("history_agri_raw.npy", "history_geometry.npy", "history_cot_log1p.npy",
                       "forecast_agri_raw.npy", "forecast_geometry.npy", "forecast_cot_log1p.npy")
        manifest["array_sha256"] = {name: sha(bank / name) for name in array_names}
        manifest["index_sha256"] = sha(bank / "index.csv")
        manifest_path.write_text(json.dumps(manifest))
        subprocess.run([sys.executable, str(script), "--forecast-bank", str(bank),
                        "--crosswalk", str(root / "cross.json"), "--keys", str(root / "keys.npz"),
                        "--ghi-pack", str(pack), "--output", str(out)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        train = np.load(out / "train/agri_raw.npy", allow_pickle=False)
        val = np.load(out / "val/agri_raw.npy", allow_pickle=False)
        # +15: init-15, init, predicted +15. +30: init, predicted +15,+30.
        assert np.all(train[0, :, 0, 0, 0] == np.asarray([1, 3, 2]))
        assert np.all(val[0, :, 0, 0, 0] == np.asarray([3, 2, 2]))
        meta = json.loads((out / "pack_manifest.json").read_text())
        assert meta["history_offsets_minutes"] == [-15, 0]
        assert len(meta["producer_sha256"]) == 64
        assert meta["test_payload_opened"] is False
        assert meta["future_observed_agri_used_as_input"] is False

        # Run the chunk-gather implementation against the same synthetic bank
        # and require byte-identical sample tensors/labels/keys. This checks
        # optimization does not alter the scientific input contract.
        vectorized = Path(__file__).with_name("build_future_ghi_head_pack_vectorized.py")
        out_v = root / "out_vectorized"
        subprocess.run([sys.executable, str(vectorized), "--forecast-bank", str(bank),
                        "--crosswalk", str(root / "cross.json"), "--keys", str(root / "keys.npz"),
                        "--ghi-pack", str(pack), "--output", str(out_v), "--chunk-rows", "1"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for split in ("train", "val"):
            for name in ("agri_raw.npy", "geometry.npy", "cot_log1p.npy", "ghi.npy", "clear.npy"):
                a = np.load(out / split / name, allow_pickle=False)
                b = np.load(out_v / split / name, allow_pickle=False)
                assert np.array_equal(a, b), (split, name)
            assert (out / split / "keys.csv").read_bytes() == (out_v / split / "keys.csv").read_bytes()
        meta_v = json.loads((out_v / "pack_manifest.json").read_text())
        assert meta_v["rows"] == meta["rows"]
        assert meta_v["test_payload_opened"] is False


if __name__ == "__main__":
    test_context_boundary()
    test_synthetic_pack_end_to_end()
    print("FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS")
