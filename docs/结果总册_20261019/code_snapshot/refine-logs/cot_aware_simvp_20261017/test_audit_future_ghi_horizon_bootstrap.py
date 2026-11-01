#!/usr/bin/env python3
"""Synthetic witness for horizon-specific date-block GHI bootstrap."""
import numpy as np

from audit_future_ghi_horizon_bootstrap import HORIZONS, STATIONS, summarize


def main():
    seeds = (42, 43)
    days = ["2026-06-01", "2026-06-02", "2026-06-03"]
    leads0 = np.asarray([15, 60, 75, 120, 135, 240], dtype=np.int16)
    errors, day_ids, leads = {}, {}, {}
    arms = ("A", "AH", "AC", "AC50")
    for seed in seeds:
        for station in STATIONS:
            lead = np.tile(leads0, len(days))
            ids = np.repeat(np.arange(len(days), dtype=np.int32), len(leads0))
            day_ids[(seed, station)] = ids
            leads[(seed, station)] = lead
            for arm, offset in (("A", 10.0), ("AH", 8.0), ("AC", 6.0), ("AC50", 5.0)):
                errors[(seed, station, arm)] = np.full(len(lead), offset, dtype=np.float64)
    comparisons = (("AC", "AH"), ("AC", "A"), ("AH", "A"),
                   ("AC50", "AH"), ("AC50", "AC"), ("AC50", "A"))
    resamples = np.random.default_rng(20261019).integers(0, len(days), size=(500, len(days)))
    result = summarize(errors, day_ids, leads, days, STATIONS, seeds, arms, comparisons, resamples)
    assert set(result) == set(HORIZONS)
    for horizon in HORIZONS:
        item = result[horizon]
        assert abs(item["delta_rmse_wm2"]["AC_minus_AH"] + 2.0) < 1e-12
        assert abs(item["delta_rmse_wm2"]["AC_minus_A"] + 4.0) < 1e-12
        assert abs(item["delta_rmse_wm2"]["AC50_minus_AH"] + 3.0) < 1e-12
        assert np.allclose(item["date_block_bootstrap_95pct_ci_wm2"]["AC_minus_AH"], [-2.0, -2.0])
        assert np.allclose(item["date_block_bootstrap_95pct_ci_wm2"]["AC50_minus_AC"], [-1.0, -1.0])
    # Every horizon must use the same resampled target-date multiplicities.
    assert np.allclose(result["short_15_60"]["date_block_bootstrap_95pct_ci_wm2"]["AC_minus_AH"],
                       result["long_135_240"]["date_block_bootstrap_95pct_ci_wm2"]["AC_minus_AH"])
    pilot_arms = ("A", "AH", "AC")
    pilot_comparisons = (("AC", "AH"), ("AC", "A"), ("AH", "A"))
    pilot = summarize(errors, day_ids, leads, days, STATIONS, seeds, pilot_arms,
                      pilot_comparisons, resamples)
    assert set(pilot["all_15_240"]["delta_rmse_wm2"]) == {
        "AC_minus_AH", "AC_minus_A", "AH_minus_A"
    }
    print("FUTURE_GHI_HORIZON_BOOTSTRAP_WITNESS_PASS")


if __name__ == "__main__":
    main()
