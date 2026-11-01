#!/usr/bin/env python3
"""Independent validation-output audit for Hubei forecast-GHI head runs.

Reads only the train/validation forecast pack and saved validation predictions;
never opens test or buffer payloads.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


STATIONS = {"Niushou": 2316, "Wangzhai": 2112, "Mizhuang": 2295}
ARMS = ("FA", "FAC")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_keys(d: np.lib.npyio.NpzFile) -> list[tuple[str, str, str, int]]:
    keys = []
    for st, init, target, lead in zip(
        d["station"], d["init_time_bjt"], d["target_time_bjt"], d["lead_minutes"]
    ):
        s, i, t, l = str(st), str(init), str(target), int(lead)
        if l not in range(15, 241, 15):
            raise AssertionError(f"unexpected lead {l}")
        init_dt = dt.datetime.strptime(i, "%Y%m%d%H%M")
        target_dt = dt.datetime.strptime(t, "%Y%m%d%H%M")
        if target_dt - init_dt != dt.timedelta(minutes=l):
            raise AssertionError(f"init/target/lead mismatch: {s},{i},{t},{l}")
        keys.append((s, i, t, l))
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate validation forecast key")
    return keys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    meta_path = a.run / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["state"] == "HUBEI_GHI_HEADS_SEED_COMPLETE"
    assert meta["input_mode"] == "forecast"
    assert set(meta["arms"]) == set(ARMS)
    assert meta["epochs"] == 30 and meta["batch_size"] == 1536
    assert meta["test_payloads_read"] == 0

    result = {
        "state": "HUBEI_FORECAST_GHI_SEED_VALIDATION_AUDIT_PASS_TEST_CLOSED",
        "seed": meta["seed"],
        "run_metadata_sha256": sha(meta_path),
        "pack_report_sha256": meta["pack_sha256"],
        "test_payloads_read": 0,
        "scope": "validation predictions only; no test or buffer payloads opened",
        "stations": {},
    }
    all_rows = (a.run / "all_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if len(all_rows) != len(STATIONS) * len(ARMS) * meta["epochs"]:
        raise AssertionError(f"unexpected epoch metric rows: {len(all_rows)}")

    for st, expected_n in STATIONS.items():
        arm_data = {}
        arm_result = {}
        for arm in ARMS:
            bundle = a.run / st / f"{arm}_validation_predictions_with_keys.npz"
            summary_path = a.run / st / f"{arm}_validation.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with np.load(bundle, allow_pickle=False) as d:
                keys = canonical_keys(d)
                truth = np.asarray(d["truth_ghi_wm2"], dtype=np.float64)
                pred = np.asarray(d["pred_ghi_wm2"], dtype=np.float64)
                if len(keys) != expected_n or len(truth) != expected_n or len(pred) != expected_n:
                    raise AssertionError(f"unexpected support for {st}/{arm}")
                if not np.isfinite(truth).all() or not np.isfinite(pred).all():
                    raise AssertionError(f"nonfinite validation values for {st}/{arm}")
                rmse = float(np.sqrt(np.mean((pred - truth) ** 2, dtype=np.float64)))
                mae = float(np.mean(np.abs(pred - truth), dtype=np.float64))
                if abs(rmse - float(summary["rmse_wm2"])) > 1e-7:
                    raise AssertionError(f"RMSE mismatch for {st}/{arm}")
                if abs(mae - float(summary["mae_wm2"])) > 1e-7:
                    raise AssertionError(f"MAE mismatch for {st}/{arm}")
                if summary["seed"] != meta["seed"] or summary["val_n"] != expected_n:
                    raise AssertionError(f"metadata mismatch for {st}/{arm}")
                if summary["test_payloads_read"] != 0:
                    raise AssertionError(f"test payload counter not closed for {st}/{arm}")
                arm_data[arm] = (keys, truth)
                arm_result[arm] = {
                    "n": expected_n,
                    "rmse_wm2_recomputed_fp64": rmse,
                    "mae_wm2_recomputed_fp64": mae,
                    "prediction_bundle_sha256": sha(bundle),
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "saved_metric_json_sha256": sha(summary_path),
                }
        if arm_data["FA"][0] != arm_data["FAC"][0]:
            raise AssertionError(f"FA/FAC key mismatch for {st}")
        if not np.array_equal(arm_data["FA"][1], arm_data["FAC"][1]):
            raise AssertionError(f"FA/FAC truth mismatch for {st}")
        arm_result["paired_keys_and_truth_identical"] = True
        arm_result["FAC_minus_FA_rmse_wm2"] = (
            arm_result["FAC"]["rmse_wm2_recomputed_fp64"]
            - arm_result["FA"]["rmse_wm2_recomputed_fp64"]
        )
        result["stations"][st] = arm_result

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
