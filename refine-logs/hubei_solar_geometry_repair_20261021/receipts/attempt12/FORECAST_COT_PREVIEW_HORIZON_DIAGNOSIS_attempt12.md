# Forecast-COT preview by horizon

This is a descriptive re-aggregation of `FORECAST_COT_PREVIEW_CPU_attempt12.csv`, using the full-train SimVP → corrected-R CPU FP32 preview on validation windows. It is not the final OOF assessment or an independent test result.

For each lead, define `ΔRMSE = RMSE(SimVP-predicted AGRI → R) − RMSE(history-COT persistence)` on the same CPP common-valid support and target pairs. Positive values mean the forecast chain is worse. The table averages the four lead-specific RMSE differences in each one-hour horizon; it does **not** recompute a pooled RMSE across all pixels or leads. COT RMSE is dimensionless.

| Forecast horizon | 16×16 patch: mean lead ΔRMSE (leads worse) | Station pixel: mean ΔRMSE (station × lead comparisons worse) | Station 3×3: mean ΔRMSE (station × lead comparisons worse) |
|---|---:|---:|---:|
| 0–1 h (+15…+60 min) | +1.830 (3/4) | +2.665 (11/12) | +2.679 (11/12) |
| 1–2 h (+75…+120 min) | +0.568 (4/4) | +1.264 (12/12) | +1.256 (12/12) |
| 2–3 h (+135…+180 min) | +0.679 (4/4) | +1.682 (12/12) | +1.717 (12/12) |
| 3–4 h (+195…+240 min) | +3.073 (4/4) | +4.114 (12/12) | +4.198 (12/12) |

Across all 16 leads, the preview has higher RMSE than persistence at 15/16 leads for the full patch, and 47/48 station-pixel and 47/48 station-3×3 station/lead comparisons. The largest gap is at 3–4 h, especially in the station neighborhoods. This points to the forecast-COT chain as a real bottleneck in this preview; it does not establish whether the cause is cloud displacement, cloud growth/decay, or retrieval error. Formal corrected-geometry OOF banks and the same-support evaluation must confirm the pattern before drawing a conclusion or changing the model.

Source: `FORECAST_COT_PREVIEW_CPU_attempt12.csv`; generated from the 407-window validation bank. Per-lead sample/date coverage varies (see source CSV), so the horizon means are descriptive and no combined confidence interval is implied.
