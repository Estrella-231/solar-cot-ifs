import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("interval_scorer", HERE / "score_direct_cot_interval_support_hcv.py")
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def synthetic_data(candidate_error: float):
    data = {}
    for station in SCORER.STATIONS:
        for fold in range(5):
            day = datetime(2024, 1, 1) + timedelta(days=fold * 4)
            keys = [{"target_time_utc": (day + timedelta(hours=12)).isoformat(), "lead_minutes": 15}]
            truth = np.array([100.0])
            for seed in SCORER.SEEDS:
                data[(station, fold, "none", seed)] = {"keys": keys, "pred": truth + 20.0, "truth": truth}
                data[(station, fold, "endpoint3", seed)] = {"keys": keys, "pred": truth + 10.0, "truth": truth}
                data[(station, fold, "interval3", seed)] = {"keys": keys, "pred": truth + candidate_error, "truth": truth}
    return data


def test_improvement_is_negative_and_degradation_positive():
    improved = synthetic_data(candidate_error=2.0)
    result = SCORER.estimate(improved, "interval3", "endpoint3", False, 100, np.random.default_rng(0))
    assert result["delta_rmse_wm2"] < 0
    degraded = synthetic_data(candidate_error=18.0)
    result = SCORER.estimate(degraded, "interval3", "endpoint3", False, 100, np.random.default_rng(0))
    assert result["delta_rmse_wm2"] > 0


if __name__ == "__main__":
    test_improvement_is_negative_and_degradation_positive()
    print("PASS: candidate-minus-reference sign and paired block bootstrap")
