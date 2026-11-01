#!/usr/bin/env python3
"""Standard-library unit test for the forecast OOF provenance gate."""
from __future__ import annotations

import unittest

from hubei_forecast_pack_gate import validate_input_pack_contract


class ForecastPackGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = {
            "state": "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED",
            "test_and_buffer_values_read": 0,
        }

    def test_rejects_existing_non_oof_pack(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires audited OOF"):
            validate_input_pack_contract({**self.base, "train_forecast_inputs_are_oof": False}, "forecast")

    def test_accepts_complete_oof_lineage(self) -> None:
        meta = {
            **self.base,
            "train_forecast_inputs_are_oof": True,
            "oof_fold_count": 5,
            "oof_train_bank_report_sha256": "a" * 64,
            "validation_forecast_source": "frozen_full_train_S_R",
        }
        validate_input_pack_contract(meta, "forecast")

    def test_rejects_wrong_fold_count(self) -> None:
        meta = {
            **self.base,
            "train_forecast_inputs_are_oof": True,
            "oof_fold_count": 4,
            "oof_train_bank_report_sha256": "a" * 64,
            "validation_forecast_source": "frozen_full_train_S_R",
        }
        with self.assertRaisesRegex(ValueError, "five-fold"):
            validate_input_pack_contract(meta, "forecast")

    def test_same_time_contract_is_unchanged(self) -> None:
        validate_input_pack_contract({
            "state": "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED",
            "test_and_buffer_values_read": 0,
        }, "same_time")


if __name__ == "__main__":
    unittest.main()
