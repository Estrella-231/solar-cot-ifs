#!/usr/bin/env python3
"""Small deterministic tests for direct forecast-COT GHI scorer contracts."""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("score_direct_cot_ghi_v1", HERE / "score_direct_cot_ghi_v1.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def main():
    candidates = []
    for base in HERE.parents:
        candidates.extend((base / "experiments" / name / "manifests_v1" / "day_assignment.csv"
                           for name in ("monthly_block_20261002", "hunan_monthly_block_20261002")))
    manifest = next((p for p in candidates if p.exists()), None)
    if manifest is None:
        raise FileNotFoundError("frozen day_assignment.csv not found in workspace/server project ancestors")
    dates = MOD.val_blocks(manifest)
    assert len(dates) == 52 and len(set(dates.values())) == 26
    bjt_date = (MOD.parse_target({"target_time_utc": "2024-04-18 23:00:00"})[0] + MOD.timedelta(hours=8)).date().isoformat()
    assert bjt_date == "2024-04-19" and dates[bjt_date] == "2024-04"
    m = MOD.metric(np.array([90., 790., 820.]), np.array([100., 800., 800.]))
    assert m["n"] == 3 and abs(m["rmse_wm2"] - 200. ** .5) < 1e-10
    assert m["high_ghi_n"] == 2 and m["high_ghi_underestimate_fraction"] == 0.5
    # The first block was sampled twice and the second once: multiplicity
    # changes the with-replacement bootstrap RMSE from sqrt(50) to sqrt(100/3).
    weighted = MOD.block_weighted_rmse(np.array([[2, 1]], dtype=np.int16),
                                       np.array([0.0, 100.0]), np.array([1.0, 1.0]))
    assert np.allclose(weighted, [np.sqrt(100.0 / 3.0)])
    raw = np.frombuffer(b'{"station":"sili"}\n', dtype=np.uint8)
    data, keys = MOD.read_jsonl_bytes(raw)
    assert data == b'{"station":"sili"}\n' and keys == [{"station": "sili"}]
    print("DIRECT_COT_GHI_SCORER_TESTS_PASS", json.dumps({"validation_days": len(dates), "blocks": len(set(dates.values()))}))


if __name__ == "__main__":
    main()
