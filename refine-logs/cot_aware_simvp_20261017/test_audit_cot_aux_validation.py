"""Synthetic unit witness for paired date bootstrap and metric aggregation."""
import numpy as np

from audit_cot_aux_validation import METHODS, aggregate_scale


def main():
    # Two target dates, two stations, sixteen leads, and a 2x2 pixel support.
    ref = np.full((2, 2, 16, 2, 2), 10.0, np.float64)
    ref[..., 0, 0] = 35.0
    ref[..., 0, 1] = 0.5
    mask = np.ones_like(ref, dtype=bool)
    pred = {
        "original_simvp": ref + 2.0,
        "mse_4000": ref + 1.0,
        "cot_aux_4000": ref.copy(),
    }
    date_index = np.tile(np.arange(2, dtype=np.int16)[:, None], (1, 16))
    result = aggregate_scale(pred, ref, mask, date_index, ["d1", "d2"], bootstrap=200, seed=1)
    assert tuple(result["scores"]) == METHODS
    assert abs(result["point_delta"]["cot_minus_original"] + 2.0) < 1e-12
    assert abs(result["point_delta"]["cot_minus_equal_budget_mse"] + 1.0) < 1e-12
    assert result["paired_bjt_target_date_bootstrap_95ci"]["cot_minus_original"] == [-2.0, -2.0]
    assert result["scores"]["cot_aux_4000"]["station_lead_macro_mae"] == 0.0
    assert result["scores"]["cot_aux_4000"]["station_lead_macro_bias"] == 0.0
    assert result["scores"]["cot_aux_4000"]["pooled_thick_csi_cot_ge30"] == 1.0
    assert result["scores"]["cot_aux_4000"]["near_zero_false_alarm_rate_cpp_lt1_pred_ge1"] == 0.0
    print("COT_AUX_PAIRED_AUDIT_SYNTHETIC_PASS")


if __name__ == "__main__":
    main()
