"""Pure-Python provenance gate for Hubei same-time/forecast GHI packs."""
from __future__ import annotations


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
        if not meta.get("oof_train_bank_report_sha256"):
            raise ValueError("Forecast pack lacks the merged OOF train-bank report lineage")
        if meta.get("validation_forecast_source") != "frozen_full_train_S_R":
            raise ValueError("Validation inputs must come from the frozen full-train S->R pipeline")
