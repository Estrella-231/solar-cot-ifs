#!/usr/bin/env python3
"""Scope checks with standard full-train and opt-in cross-fitting modes."""
from __future__ import annotations

import unittest

from hubei_forecast_pack_gate import validate_input_pack_contract


class ForecastPackGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = {
            "state": "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED",
            "test_and_buffer_values_read": 0,
            "validation_forecast_source": "frozen_full_train_S_R",
        }

    def test_accepts_full_train_predictions_by_default(self) -> None:
        validate_input_pack_contract({**self.base, "train_forecast_inputs_are_oof": False}, "forecast")

    def test_legacy_oof_metadata_is_not_required_for_standard_pack(self) -> None:
        validate_input_pack_contract(self.base, "forecast")

    def test_explicit_oof_mode_rejects_non_oof_pack(self) -> None:
        with self.assertRaisesRegex(ValueError, "Explicit OOF mode"):
            validate_input_pack_contract({**self.base, "train_forecast_inputs_are_oof": False},
                                         "forecast", require_oof=True)

    def test_accepts_complete_oof_lineage(self) -> None:
        meta = {
            **self.base,
            "train_forecast_inputs_are_oof": True,
            "oof_fold_count": 5,
            "oof_train_bank_report_sha256": "a" * 64,
            "validation_forecast_source": "frozen_full_train_S_R",
        }
        validate_input_pack_contract(meta, "forecast")

    def test_accepts_valid_non_five_fold_study(self) -> None:
        meta = {
            **self.base,
            "train_forecast_inputs_are_oof": True,
            "oof_fold_count": 4,
            "oof_train_bank_report_sha256": "a" * 64,
            "validation_forecast_source": "frozen_full_train_S_R",
        }
        validate_input_pack_contract(meta, "forecast")
        with self.assertRaisesRegex(ValueError, "requested 5"):
            validate_input_pack_contract(meta, "forecast", expected_oof_folds=5)

    def test_exact_legacy_five_fold_mode(self) -> None:
        validate_input_pack_contract({**self.base, "train_forecast_inputs_are_oof": True,
                                     "oof_fold_count": 5, "oof_train_bank_report_sha256": "a"*64},
                                     "forecast", require_oof=True, expected_oof_folds=5)

    def test_oof_claim_still_needs_lineage(self) -> None:
        with self.assertRaisesRegex(ValueError, "report lineage"):
            validate_input_pack_contract({**self.base, "train_forecast_inputs_are_oof": True,
                                         "oof_fold_count": 5}, "forecast")

    def test_rejects_test_or_buffer_payload_use(self) -> None:
        with self.assertRaisesRegex(ValueError, "Train/validation-only"):
            validate_input_pack_contract({**self.base, "test_and_buffer_values_read": 1}, "forecast")

    def test_rejects_future_observations_in_forecast_input(self) -> None:
        for field in ("future_observed_agri_used_as_input", "cpp_used_as_inference_input",
                      "future_observed_qa_used_as_input"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unavailable future"):
                validate_input_pack_contract({**self.base, field: True}, "forecast")

    def test_rejects_unbound_validation_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "Validation inputs"):
            validate_input_pack_contract({**self.base, "validation_forecast_source": "unknown"}, "forecast")

    def test_invalid_declared_fold_count_is_rejected(self) -> None:
        for folds in (0, 1, True, 2.5):
            with self.subTest(folds=folds), self.assertRaisesRegex(ValueError, "valid fold count"):
                validate_input_pack_contract({**self.base, "train_forecast_inputs_are_oof": True,
                                             "oof_fold_count": folds}, "forecast")

    def test_same_time_cannot_request_oof(self) -> None:
        with self.assertRaisesRegex(ValueError, "only to forecast"):
            validate_input_pack_contract({"state": "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED",
                                          "test_and_buffer_values_read": 0}, "same_time", require_oof=True)

    def test_same_time_contract_is_unchanged(self) -> None:
        validate_input_pack_contract({
            "state": "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED",
            "test_and_buffer_values_read": 0,
        }, "same_time")


if __name__ == "__main__":
    unittest.main()
