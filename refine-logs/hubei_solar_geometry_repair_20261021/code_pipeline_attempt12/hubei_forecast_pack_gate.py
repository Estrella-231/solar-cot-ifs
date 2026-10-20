"""Pure-Python provenance gate for Hubei same-time/forecast GHI packs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_input_pack_contract(meta: dict, input_mode: str) -> None:
    """Reject forecast training packs whose training inputs are not audited OOF."""
    if input_mode not in ("same_time", "forecast"):
        raise ValueError(f"Unknown input mode: {input_mode}")
    expected_state = ("HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED" if input_mode == "same_time"
                      else "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED")
    if meta.get("state") != expected_state or meta.get("test_and_buffer_values_read") != 0:
        raise ValueError("Train/validation-only common pack gate failed")
    if input_mode == "forecast":
        if meta.get("train_forecast_inputs_are_oof") is not True:
            raise ValueError("Forecast-head training requires audited OOF training inputs")
        if meta.get("oof_fold_count") != 5:
            raise ValueError("Forecast-head training requires the frozen five-fold OOF contract")
        report_path = Path(meta.get("oof_train_bank_report_path", ""))
        expected_sha = meta.get("oof_train_bank_report_sha256")
        if not report_path.is_file() or not expected_sha:
            raise ValueError("Forecast pack lacks the merged OOF train-bank report lineage")
        digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise ValueError("Forecast pack OOF report SHA-256 does not match its report file")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("state") != "PASS_OOF_TRAIN_BANK_LINEAGE_AND_GHI_ROW_BINDING"
                or report.get("train_rows_checked", 0) <= 0
                or report.get("validation_rows_checked", 0) <= 0
                or report.get("test_payloads_read") != 0
                or report.get("buffer_payloads_read") != 0):
            raise ValueError("Forecast pack OOF bank audit report is not a passing train/validation receipt")
        if meta.get("validation_forecast_source") != "frozen_full_train_S_R":
            raise ValueError("Validation inputs must come from the frozen full-train S->R pipeline")
