#!/usr/bin/env python3
"""CPU contract witness for same-time and forecast Hubei GHI heads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("train_hubei_ghi_heads.py")
spec = importlib.util.spec_from_file_location("train_hubei_ghi_heads", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def main() -> None:
    torch.manual_seed(20261016)
    x = torch.randn(2, 15, 3, 3)
    cot = torch.rand(2, 1, 3, 3)
    checks = {}
    # Standard cascade training accepts full-train upstream predictions.
    # OOF remains an explicit historical-experiment option.
    validate_pack = module.validate_input_pack_contract
    common_forecast = {
        "state": "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED",
        "test_and_buffer_values_read": 0,
        "train_forecast_inputs_are_oof": False,
        "validation_forecast_source": "frozen_full_train_S_R",
    }
    validate_pack(common_forecast, "forecast")
    checks["accept_full_train_forecast_pack"] = "PASS"
    try:
        validate_pack(common_forecast, "forecast", require_oof=True)
    except ValueError as exc:
        assert "OOF" in str(exc)
        checks["explicit_oof_mode_rejects_non_oof_pack"] = "PASS"
    else:
        raise AssertionError("Explicit OOF mode accepted non-OOF inputs")
    audited_oof = {
        **common_forecast,
        "train_forecast_inputs_are_oof": True,
        "oof_fold_count": 5,
        "oof_train_bank_report_sha256": "a" * 64,
        "validation_forecast_source": "frozen_full_train_S_R",
    }
    validate_pack(audited_oof, "forecast", require_oof=True, expected_oof_folds=5)
    checks["accept_five_fold_oof_lineage"] = "PASS"
    try:
        validate_pack({**audited_oof, "validation_forecast_source": "oof_train_fold"}, "forecast")
    except ValueError as exc:
        assert "Validation inputs" in str(exc)
        checks["reject_oof_validation_source"] = "PASS"
    else:
        raise AssertionError("Forecast trainer accepted OOF validation inputs")
    validate_pack({"state": "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED",
                   "test_and_buffer_values_read": 0}, "same_time")
    checks["same_time_pack_gate_unchanged"] = "PASS"
    for arm in ("A", "FA", "AP", "FAP"):
        pred = module.HubeiLocalGHIHead(arm)(x)
        assert pred.shape == (2,) and torch.isfinite(pred).all()
        checks[f"{arm}_without_cot"] = "PASS"
    for arm in ("AC", "FAC"):
        model = module.HubeiLocalGHIHead(arm)
        pred = model(x, cot)
        assert pred.shape == (2,) and torch.isfinite(pred).all()
        try:
            model(x)
        except ValueError:
            checks[f"{arm}_cot_required"] = "PASS"
        else:
            raise AssertionError(f"{arm} unexpectedly accepted missing COT")
        checks[f"{arm}_with_cot"] = "PASS"
    print({"state": "HUBEI_FORECAST_GHI_HEAD_CONTRACT_PASS", "checks": checks})


if __name__ == "__main__":
    main()
