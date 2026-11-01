#!/usr/bin/env python3
"""Synthetic unit tests for Hubei forecast metric primitives only."""
from __future__ import annotations

import unittest

import numpy as np

from hubei_forecast_metrics import (
    forecast_metrics,
    metrics_by_lead,
    paired_date_cluster_bootstrap_rmse_difference,
    validate_common_key_support,
)


class ForecastMetricTests(unittest.TestCase):
    def test_metrics_match_direct_calculation_and_protocol_denominator(self) -> None:
        y = np.array([100.0, 200.0, 300.0])
        p = np.array([110.0, 180.0, 330.0])
        out = forecast_metrics(y, p)
        self.assertAlmostEqual(out["rmse_wm2"], np.sqrt(1400.0 / 3.0))
        self.assertAlmostEqual(out["mae_wm2"], 20.0)
        self.assertAlmostEqual(out["bias_wm2"], 20.0 / 3.0)
        self.assertAlmostEqual(out["r2"], 0.93)
        self.assertAlmostEqual(out["nrmse_mean_observed_pct"], 100.0 * np.sqrt(1400.0 / 3.0) / 200.0)

    def test_zero_mean_makes_normalized_metrics_unavailable(self) -> None:
        out = forecast_metrics([0.0, 0.0], [0.0, 0.0])
        self.assertIsNone(out["nrmse_mean_observed_pct"])
        self.assertIsNone(out["nmae_mean_observed_pct"])
        self.assertIsNone(out["r2"])

    def test_clear_sky_sensitivity_is_explicit_and_reports_support(self) -> None:
        out = forecast_metrics(
            [10.0, 100.0, 200.0], [20.0, 90.0, 230.0],
            clear_sky_wm2=[20.0, 150.0, 500.0], clear_sky_threshold_wm2=100.0,
        )
        self.assertEqual(out["clear_sky_sensitivity"]["n"], 2)
        self.assertAlmostEqual(out["clear_sky_sensitivity"]["rmse_wm2"], np.sqrt(500.0))
        with self.assertRaisesRegex(ValueError, "threshold must be finite"):
            forecast_metrics([10.0], [11.0], clear_sky_wm2=[20.0])

    def test_per_lead_metrics_keep_each_lead_support(self) -> None:
        out = metrics_by_lead([1.0, 2.0, 4.0], [2.0, 2.0, 2.0], [15, 30, 30])
        self.assertEqual(set(out), {15, 30})
        self.assertEqual(out[15]["n"], 1)
        self.assertEqual(out[30]["n"], 2)

    def test_common_key_support_rejects_duplicates_and_mismatch(self) -> None:
        keys = [("i1", "t1", 15), ("i2", "t2", 30)]
        self.assertEqual(validate_common_key_support({"A": keys, "B": list(reversed(keys))}), tuple(sorted(keys)))
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            validate_common_key_support({"A": keys, "B": [keys[0], keys[0]]})
        with self.assertRaisesRegex(ValueError, "exact key support"):
            validate_common_key_support({"A": keys, "B": [keys[0]]})

    def test_paired_date_bootstrap_fails_instead_of_skipping_invalid_draws(self) -> None:
        keys = [("202606010800", f"20260601{hour:02d}00", lead) for hour, lead in [(8, 15), (9, 30)]]
        records = {}
        for arm, pred in (("A", [10.0, 10.0]), ("B", [9.0, 9.0])):
            records[(42, "N", arm)] = {"keys": keys, "truth_wm2": [10.0, 10.0], "prediction_wm2": pred}
        # The one-date sample fails explicitly; no NaN group may be silently skipped.
        with self.assertRaisesRegex(ValueError, "at least two target dates"):
            paired_date_cluster_bootstrap_rmse_difference(
                records, control_arm="A", treatment_arm="B", seeds=[42], stations=["N"], resamples=100,
            )

    def test_paired_date_bootstrap_checks_arm_truth_equality(self) -> None:
        keys_a = [("202606010800", "202606010815", 15), ("202606020800", "202606020815", 15)]
        records = {
            (42, "N", "A"): {"keys": keys_a, "truth_wm2": [10.0, 20.0], "prediction_wm2": [11.0, 20.0]},
            (42, "N", "B"): {"keys": keys_a, "truth_wm2": [10.0, 21.0], "prediction_wm2": [10.0, 20.0]},
        }
        with self.assertRaisesRegex(ValueError, "paired keys/truth differ"):
            paired_date_cluster_bootstrap_rmse_difference(
                records, control_arm="A", treatment_arm="B", seeds=[42], stations=["N"], resamples=100,
            )

    def test_paired_date_bootstrap_accepts_compact_bjt_and_reports_point_estimate(self) -> None:
        keys = [(f"2026060{day}0800", f"2026060{day}0815", 15) for day in (1, 2, 3)]
        records = {
            (42, "N", "A"): {"keys": keys, "truth_wm2": [10.0, 20.0, 30.0], "prediction_wm2": [12.0, 20.0, 30.0]},
            (42, "N", "B"): {"keys": keys, "truth_wm2": [10.0, 20.0, 30.0], "prediction_wm2": [10.0, 20.0, 30.0]},
        }
        out = paired_date_cluster_bootstrap_rmse_difference(
            records, control_arm="A", treatment_arm="B", seeds=[42], stations=["N"], resamples=500,
        )
        self.assertEqual(out["target_dates"], ["20260601", "20260602", "20260603"])
        self.assertAlmostEqual(out["estimate_difference_wm2"], -np.sqrt(4.0 / 3.0))

    def test_two_day_bootstrap_uses_only_consecutive_calendar_blocks(self) -> None:
        keys = [(f"202606{day:02d}0800", f"202606{day:02d}0815", 15)
                for day in (1, 2, 4, 5)]
        records = {
            (42, "N", "A"): {
                "keys": keys, "truth_wm2": [10.0, 20.0, 30.0, 40.0],
                "prediction_wm2": [12.0, 20.0, 32.0, 40.0],
            },
            (42, "N", "B"): {
                "keys": keys, "truth_wm2": [10.0, 20.0, 30.0, 40.0],
                "prediction_wm2": [10.0, 20.0, 30.0, 40.0],
            },
        }
        out = paired_date_cluster_bootstrap_rmse_difference(
            records, control_arm="A", treatment_arm="B", seeds=[42],
            stations=["N"], resamples=100, random_seed=7, block_length_days=2,
        )
        self.assertEqual(out["block_length_days"], 2)
        self.assertEqual(out["eligible_block_count"], 2)
        self.assertEqual(out["invalid_resamples_skipped"], 0)
        gap_records = {
            (42, "N", "A"): {
                "keys": [keys[0], keys[2]], "truth_wm2": [10.0, 30.0],
                "prediction_wm2": [12.0, 32.0],
            },
            (42, "N", "B"): {
                "keys": [keys[0], keys[2]], "truth_wm2": [10.0, 30.0],
                "prediction_wm2": [10.0, 30.0],
            },
        }
        unsupported = paired_date_cluster_bootstrap_rmse_difference(
            gap_records, control_arm="A", treatment_arm="B", seeds=[42],
            stations=["N"], resamples=100, random_seed=7,
            block_length_days=2,
        )
        self.assertEqual(
            unsupported["status"],
            "unavailable_dates_not_partitionable_into_complete_blocks",
        )

        # A mixed pair plus an isolated date must not produce a CI that silently
        # drops the isolated date's error from the point estimate.
        mixed_keys = [keys[0], keys[1], keys[2]]
        mixed_records = {
            (42, "N", "A"): {
                "keys": mixed_keys, "truth_wm2": [10.0, 20.0, 30.0],
                "prediction_wm2": [10.0, 20.0, 60.0],
            },
            (42, "N", "B"): {
                "keys": mixed_keys, "truth_wm2": [10.0, 20.0, 30.0],
                "prediction_wm2": [10.0, 20.0, 30.0],
            },
        }
        mixed = paired_date_cluster_bootstrap_rmse_difference(
            mixed_records, control_arm="A", treatment_arm="B", seeds=[42],
            stations=["N"], resamples=100, random_seed=7,
            block_length_days=2,
        )
        self.assertEqual(
            mixed["status"],
            "unavailable_dates_not_partitionable_into_complete_blocks",
        )

    def test_target_date_parser_rejects_partial_or_trailing_timestamps(self) -> None:
        from hubei_forecast_metrics import _target_date_from_key

        self.assertEqual(_target_date_from_key(("i", "202606010815", 15)), "20260601")
        self.assertEqual(_target_date_from_key(("i", "2026-06-01 08:15:00", 15)), "20260601")
        for malformed in (
            "2026-06-01NOT_A_TIME", "202606011", "20260601999999",
            "20260601081500trailing", "2026-13-01 08:15:00",
        ):
            with self.subTest(value=malformed):
                with self.assertRaisesRegex(ValueError, "invalid target"):
                    _target_date_from_key(("i", malformed, 15))

    def test_two_day_bootstrap_reports_station_missing_a_block(self) -> None:
        keys_ab = [(f"202606{day:02d}0800", f"202606{day:02d}0815", 15)
                   for day in (1, 2)]
        keys_cd = [(f"202606{day:02d}0800", f"202606{day:02d}0815", 15)
                   for day in (3, 4)]
        records = {}
        for station, keys, values in (
            ("N1", keys_ab, [10.0, 20.0]),
            ("N2", keys_cd, [30.0, 40.0]),
        ):
            for arm, pred in (("A", values), ("B", values)):
                records[(42, station, arm)] = {
                    "keys": keys, "truth_wm2": values, "prediction_wm2": pred,
                }
        out = paired_date_cluster_bootstrap_rmse_difference(
            records, control_arm="A", treatment_arm="B", seeds=[42],
            stations=["N1", "N2"], resamples=100, block_length_days=2,
        )
        self.assertEqual(out["status"], "unavailable_station_missing_support_in_block")


if __name__ == "__main__":
    unittest.main()
