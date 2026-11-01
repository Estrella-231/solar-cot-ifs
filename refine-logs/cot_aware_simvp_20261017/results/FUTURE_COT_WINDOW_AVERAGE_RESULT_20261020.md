# Future COT window-average screen

This three-seed validation screen tested whether the GHI head benefits when only its target future-COT frame is replaced by the trapezoidal mean physical COT over `(T−15 min, T]`. The head was retrained for the fixed 30 epochs at each station; all arms were evaluated on exact common station/init/target/lead/GHI-row keys and 52 target-BJT date blocks.

| Arm | Equal-station mean GHI RMSE (W m⁻²) | Difference from ACREL (W m⁻²) | 95% date-block CI for difference |
|---|---:|---:|---:|
| No COT (A) | 130.9323 | −1.0957 | [−1.3441, +0.1647] |
| History COT (AH) | 131.0653 | −1.2287 | [−1.3291, +0.0363] |
| History + predicted future COT (AC) | 130.0711 | −0.2345 | [−0.4077, +0.1595] |
| Train-only calibrated future COT (ACREL) | 129.8097 | reference | — |
| Calibrated 15-minute window-mean future COT (ACRELWIN) | 129.8366 | +0.0270 | [−0.0278, +0.0515] |

At Sili, ACRELWIN was 143.8656 versus 143.8404 W m⁻² for ACREL; at Zhujia it was 115.8077 versus 115.7790 W m⁻². Both stations changed negligibly. The main ACRELWIN−AH difference was −1.2287 W m⁻², but its interval crosses zero. The candidate therefore does not establish a forecast-COT GHI gain and does not improve on the existing calibration.

The result rules out this simple endpoint-window averaging operation as the missing fix. It does not rule out forecast-COT noise in general. A train-only lead-phase diagnostic is the next useful test: compare fixed −15/0/+15-minute shifts of the predicted target COT against exact frozen-R train references, with the shift selected inside the training folds and scored on held-out training folds. Only a robust COT-reference improvement with no station regression will unlock a three-GPU GHI-head screen. Validation labels will not select the shift; independent test and buffer remain closed.

Provenance: `future_ghi_oof_cot_window_average_validation_audit_v1.json`, candidate run root `future_ghi_oof_cot_window_average_v1`, PBS `215128.tc6000` (exit status 0). Three seeds (42/43/44), 30 epochs per station and seed, one A800 per seed. CPP, future observed AGRI as inference input, test and buffer were not used.
