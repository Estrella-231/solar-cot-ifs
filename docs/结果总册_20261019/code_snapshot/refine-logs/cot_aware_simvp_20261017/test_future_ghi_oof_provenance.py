"""Checks forecast-bank provenance survives into GHI pack metadata."""
from __future__ import annotations

from build_future_ghi_head_pack import train_forecast_provenance


def main() -> None:
    oof = train_forecast_provenance({
        "in_sample_train_forecasts": False,
        "oof_fold_count": 5,
        "oof_train_sequences": 30_436,
        "oof_fold_model_hashes": {str(i): str(i) * 64 for i in range(5)},
        "forecast_source": "five-fold date-heldout OOF SimVP",
    }, 30_436)
    assert oof["train_forecast_oof"] is True
    assert oof["forecast_bank_oof_fold_count"] == 5
    assert "validation screening" in oof["warning"]

    in_sample = train_forecast_provenance({
        "in_sample_train_forecasts": True,
        "forecast_source": "full-train SimVP",
    }, 30_436)
    assert in_sample["train_forecast_oof"] is False
    assert "in-sample" in in_sample["warning"]

    for bad in (
        ({"in_sample_train_forecasts": False, "oof_fold_count": 4,
          "oof_train_sequences": 30_436}, 30_436),
        ({"in_sample_train_forecasts": False, "oof_fold_count": 5,
          "oof_train_sequences": 30_436,
          "oof_fold_model_hashes": {str(i): str(i) * 64 for i in range(5)},
          "forecast_source": "ordinary full-train forecast"}, 30_436),
        ({"in_sample_train_forecasts": False, "oof_fold_count": 5,
          "oof_train_sequences": 30_435,
          "oof_fold_model_hashes": {str(i): str(i) * 64 for i in range(5)},
          "forecast_source": "five-fold date-heldout OOF SimVP"}, 30_436),
        ({"in_sample_train_forecasts": "false"}, 30_436),
    ):
        try:
            train_forecast_provenance(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid provenance accepted: {bad}")
    print("FUTURE_GHI_OOF_PROVENANCE_WITNESS_PASS")


if __name__ == "__main__":
    main()
