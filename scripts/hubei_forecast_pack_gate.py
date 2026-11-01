"""Pure-Python provenance gate for Hubei same-time/forecast GHI packs."""
from __future__ import annotations


def validate_input_pack_contract(
    meta: dict, input_mode: str, *, require_oof: bool = False,
    expected_oof_folds: int | None = None,
) -> None:
    """Validate input scope; cross-fitting is an optional experiment design.

    Full-train upstream predictions may train a downstream head. If a pack
    claims OOF provenance, validate that claim; historical five-fold studies
    can explicitly request their original contract.
    """
    if input_mode not in ("same_time", "forecast"):
        raise ValueError(f"Unknown input mode: {input_mode}")
    expected_state = ("HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED" if input_mode == "same_time"
                      else "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED")
    if meta.get("state") != expected_state or meta.get("test_and_buffer_values_read") != 0:
        raise ValueError("Train/validation-only common pack gate failed")
    if expected_oof_folds is not None:
        if isinstance(expected_oof_folds, bool) or not isinstance(expected_oof_folds, int) or expected_oof_folds < 2:
            raise ValueError("Expected OOF fold count must be an integer >= 2")
        require_oof = True
    if input_mode == "forecast":
        for key in ("future_observed_agri_used_as_input", "cpp_used_as_inference_input",
                    "future_observed_qa_used_as_input"):
            if meta.get(key):
                raise ValueError(f"Forecast inputs contain unavailable future observations: {key}")
        if require_oof and meta.get("train_forecast_inputs_are_oof") is not True:
            raise ValueError("Explicit OOF mode requires OOF training inputs")
        if meta.get("train_forecast_inputs_are_oof") is True:
            count = meta.get("oof_fold_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 2:
                raise ValueError("Declared OOF pack needs a valid fold count >= 2")
            if expected_oof_folds is not None and count != expected_oof_folds:
                raise ValueError(f"OOF fold count differs from requested {expected_oof_folds}")
            if not meta.get("oof_train_bank_report_sha256"):
                raise ValueError("Declared OOF pack lacks its train-bank report lineage")
        if meta.get("validation_forecast_source") != "frozen_full_train_S_R":
            raise ValueError("Validation inputs must come from the frozen full-train S->R pipeline")
    elif require_oof:
        raise ValueError("OOF requirements apply only to forecast input mode")
