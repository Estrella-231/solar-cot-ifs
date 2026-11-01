import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from merge_future_ghi_oof_bank import merge


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with TemporaryDirectory() as td:
        root = Path(td)
        base = root / "base"; base.mkdir()
        index_rows = [
            {"bank_index": "0", "split": "train", "sequence_index_within_split": "0", "seq_id": "s0"},
            {"bank_index": "1", "split": "train", "sequence_index_within_split": "1", "seq_id": "s1"},
            {"bank_index": "2", "split": "val", "sequence_index_within_split": "0", "seq_id": "v0"},
        ]
        write_csv(base / "index.csv", index_rows, list(index_rows[0]))
        shapes = {
            "history_agri_raw.npy": [3, 2, 2, 13, 3, 3],
            "history_geometry.npy": [3, 2, 2, 2, 3, 3],
            "history_cot_log1p.npy": [3, 2, 2, 1, 3, 3],
            "forecast_agri_raw.npy": [3, 2, 16, 13, 3, 3],
            "forecast_geometry.npy": [3, 2, 16, 2, 3, 3],
            "forecast_cot_log1p.npy": [3, 2, 16, 1, 3, 3],
        }
        for name, shape in shapes.items():
            np.save(base / name, np.full(shape, -1, np.float32))
        (base / "bank_manifest.json").write_text(json.dumps({"state": "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK",
            "layouts": shapes, "test_payload_opened": False, "buffer_payload_opened": False,
            "future_observed_agri_read": False, "cpp_read": False, "crosswalk_row_count_expected": 1}), encoding="utf-8")
        manifests = root / "manifests"; manifests.mkdir()
        folds = []
        for fold, seq in enumerate(("s0", "s1")):
            d = manifests / f"fold_{fold}"; d.mkdir()
            (d / "fold_oof_ids.txt").write_text(seq + "\n")
            fb = root / f"foldbank{fold}"; fb.mkdir(); folds.append(fb)
            write_csv(fb / "index.csv", [{"bank_index": "0", "split": "train", "seq_id": seq}],
                      ["bank_index", "split", "seq_id"])
            value = .25 + .5 * fold
            np.save(fb / "forecast_agri_raw.npy", np.full((1, 2, 16, 13, 3, 3), value, np.float32))
            np.save(fb / "forecast_cot_log1p.npy", np.full((1, 2, 16, 1, 3, 3), value, np.float32))
            array_hashes = {name: sha(fb / name) for name in
                            ("forecast_agri_raw.npy", "forecast_cot_log1p.npy")}
            (fb / "bank_manifest.json").write_text(json.dumps({"state": "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS",
                "fold": fold, "fold_checkpoint_sha256": f"hash{fold}", "future_observed_agri_read": False,
                "array_sha256": array_hashes, "index_sha256": sha(fb / "index.csv"),
                "cpp_read": False, "validation_test_payload_opened": False, "buffer_payload_opened": False}), encoding="utf-8")
        out = root / "merged"
        result = merge(base, manifests, folds, out)
        assert result["merged_oof_keys"] == result["train_keys"] == 2
        ag = np.load(out / "forecast_agri_raw.npy")
        assert np.all(ag[0] == .25) and np.all(ag[1] == .75) and np.all(ag[2] == -1)
        meta = json.loads((out / "bank_manifest.json").read_text())
        assert meta["in_sample_train_forecasts"] is False
        assert meta["test_payload_opened"] is False
    print("OOF_BANK_MERGE_WITNESS_PASS")


if __name__ == "__main__":
    main()
