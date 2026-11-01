"""Witness that calibration is cross-fitted by date-heldout OOF fold."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from audit_future_cot_oof_reliability_crossfit import fit_reliability


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); forecast = root / "forecast"; train = forecast / "train"
        actual = root / "actual"; folds = root / "folds"
        train.mkdir(parents=True); actual.mkdir(); folds.mkdir()
        keys = []; actual_rows = []; forecast_cot = []
        actual_lookup = {}; sequence_rows = []
        for station_i, station in enumerate(("sili", "zhujia")):
            for lead_i, lead in enumerate(range(15, 241, 15)):
                for rep in range(5):
                    seq_index = (station_i * 16 + lead_i) * 5 + rep
                    day = rep + 1
                    target = "2024-04-%02d 12:00:00" % day
                    init = "2024-04-%02d 12:00:00" % day
                    # Different daily targets allow each held-out fold to be
                    # calibrated from the other four days only.
                    key = (station, target)
                    if key not in actual_lookup:
                        actual_lookup[key] = len(actual_rows)
                        actual_rows.append({"split": "train", "station": station,
                                            "target_time_utc": target})
                    truth = 0.4 + 0.17 * rep + 0.03 * station_i
                    pred = (truth - 0.05) / 1.1 + 0.01 * lead_i
                    keys.append({"split": "train", "station": station,
                        "sequence_index_within_split": str(seq_index),
                        "target_time_utc": target, "init_time_utc": init,
                        "lead_minutes": str(lead), "ghi_row_index": str(seq_index)})
                    forecast_cot.append(pred)
                    seqid = "hunan_test_%03d" % seq_index
                    sequence_rows.append({"bank_index": str(seq_index), "split": "train",
                        "sequence_index_within_split": str(seq_index), "seq_id": seqid,
                        "fold": rep})
        # The same actual frame has one reference COT value independent of lead.
        actual_cot = np.zeros((len(actual_rows) + 1, 1, 3, 3), np.float32)
        for key, idx in actual_lookup.items():
            station_i = 0 if key[0] == "sili" else 1
            day = int(key[1][8:10])
            actual_cot[idx, 0, 1, 1] = 0.4 + 0.17 * (day - 1) + 0.03 * station_i
        actual_rows.append({"split": "val", "station": "sili",
                            "target_time_utc": "2024-05-01 12:00:00"})
        actual_cot[-1] = np.nan  # must never be indexed by this train-only audit
        farr = np.zeros((len(keys), 3, 1, 3, 3), np.float32)
        farr[:, 2, 0, 1, 1] = np.asarray(forecast_cot, np.float32)
        with (train / "keys.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(keys[0])); w.writeheader(); w.writerows(keys)
        with (actual / "rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(actual_rows[0])); w.writeheader(); w.writerows(actual_rows)
        (root / "bank_index.csv").write_text("bank_index,split,sequence_index_within_split,seq_id\n" +
            "".join("%s,%s,%s,%s\n" % (r["bank_index"], r["split"],
            r["sequence_index_within_split"], r["seq_id"]) for r in sequence_rows), encoding="utf-8")
        for fold in range(5):
            d = folds / ("fold_%d" % fold); d.mkdir()
            ids = [r["seq_id"] for r in sequence_rows if r["fold"] == fold]
            (d / "fold_oof_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        np.save(train / "cot_log1p.npy", farr); np.save(actual / "cot_log1p.npy", actual_cot)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        (forecast / "pack_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK", "train_forecast_oof": True,
            "forecast_bank_oof_fold_count": 5, "test_payload_opened": False,
            "buffer_payload_opened": False, "future_observed_agri_used_as_input": False,
            "cpp_used_as_input": False, "split_id": "split-test",
            "rows": {"train": len(keys)},
            "array_sha256": {"train": {"cot_log1p.npy": sha(train / "cot_log1p.npy")}},
        }))
        (actual / "pack_manifest.json").write_text(json.dumps({
            "state": "COMPLETE", "split_id": "split-test", "test_payload_opened": False,
            "array_sha256": {"cot_log1p.npy": sha(actual / "cot_log1p.npy"),
                             "rows.csv": sha(actual / "rows.csv")},
        }))
        result = fit_reliability(forecast, actual, root / "bank_index.csv", folds,
                                 root / "out", min_pairs=4)
        assert result["train_rows"] == 160 and result["unmatched_train_rows"] == 0
        assert result["test_payload_opened"] is False and result["buffer_payload_opened"] is False
        assert len(result["metrics"]) == 32
        assert len(result["parameters_excluding_each_fold"]) == 5
        assert result["parameters_excluding_each_fold"]["0"]["sili"]["15"]["n_train_oof_pairs"] == 4
    print("CROSSFIT_TRAIN_ONLY_OOF_COT_RELIABILITY_WITNESS_PASS")


if __name__ == "__main__":
    main()
