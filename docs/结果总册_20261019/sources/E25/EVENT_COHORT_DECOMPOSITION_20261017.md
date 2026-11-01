# Validation thick-cloud event decomposition

Date: 2026-10-17. This is a validation-only diagnostic for deciding how to improve the forecast COT chain. No test payload was read and no model was trained.

## Fixed event and forecast-selection rule

CPP station-pixel COT at `(8, 8)` must be at least 30 for two or more consecutive 15-minute targets. A missing/invalid CPP or R frame breaks the event. Overlapping forecast windows are collapsed to a unique physical event on the target-time axis. For each event, the selected forecast is the available validation initialization with the longest start lead that still covers the complete event. Scores use the common support of the current compact CPP mask and current R frame-valid mask, over the local 3×3 station neighbourhood. Events and dates are few and dependent; no confidence interval is reported.

The complete current validation sequence pack contains 4,025 unique target times per station. The fixed rule finds seven unique CPP-referenced thick-cloud events: four at Sili and three at Zhujia, across three BJT dates (Sili: 15 June and 9 July 2025; Zhujia: 4 January 2025). Event start leads range from +30 to +225 minutes. The event list and selected forecasts are in `event_forecast_pairs.csv`.

## Same-event results

RMSE and peak error use physical COT units. Thick CSI pools local-3×3 pixel-leads over the listed events. `Observed future AGRI→R` is a hindsight diagnostic and is not available to a deployable forecast.

| Station | Method | Events | Mean event local-3×3 RMSE | Mean station-pixel peak error | Pooled thick CSI |
|---|---|---:|---:|---:|---:|
| Sili | SimVP forecast AGRI→R | 4 | 19.53 | −25.92 | 0.149 |
| Sili | Observed future AGRI→R (hindsight) | 4 | 6.87 | −3.71 | 0.814 |
| Sili | Direct COT, R-target | 4 | 24.22 | −31.66 | 0.065 |
| Sili | Direct COT, CPP/R hybrid target | 4 | 24.47 | −30.37 | 0.120 |
| Sili | Direct COT, hybrid plus amplitude term | 4 | 24.42 | −28.88 | 0.272 |
| Zhujia | SimVP forecast AGRI→R | 3 | 13.48 | −13.52 | 0.000 |
| Zhujia | Observed future AGRI→R (hindsight) | 3 | 7.38 | +5.36 | 0.617 |
| Zhujia | Direct COT, R-target | 3 | 25.26 | −26.71 | 0.000 |
| Zhujia | Direct COT, CPP/R hybrid target | 3 | 26.79 | −28.12 | 0.000 |
| Zhujia | Direct COT, hybrid plus amplitude term | 3 | 26.76 | −28.03 | 0.000 |

The event-level pattern is more informative than an overall COT RMSE alone. SimVP→R detects some Sili thick events but misses or strongly underestimates others; it misses all three Zhujia station-pixel thick events. Applying the same frozen R to the observed future AGRI substantially reduces event RMSE and increases thick CSI at both stations. This indicates that errors in the forecasted image field are a material bottleneck in these events. The hindsight result is still not exact agreement with CPP: its peak bias differs by station, and CPP is only a retrieval reference. Direct history-COT forecasting performs worse than SimVP→R on the event cohort, so another direct-COT loss sweep is not supported by these results.

These seven episodes are concentrated on three dates, not seven independent days. They do not provide inferential evidence or establish a station-wide causal effect. The event decomposition should guide a candidate intervention, while the full common-support validation metrics remain the overall acceptance criterion.

## Source integrity and the stale-bank finding

The forecast bank was checked against its complete validation profile, including the SimVP→R output checksum, the current monthly-R checkpoint hash, 3,140 validation sequence IDs, target timestamps, and the direct-model key table. The direct COT rows were keyed by `(sequence_index, station)`. CPP labels and masks were rebuilt from `sequence_indices_val.npy` into the current compact `cpp_cot_physical.npy` and `cpp_common_mask.npy`, with the current R frame-valid sidecar applied.

During this audit, the older CPP value array embedded in the full-validation forecast bank differed from the current exact-frame CPP values by as much as 68.20 COT, even though its mask count matched. The bank's CPP values were therefore quarantined and not used for this report. The exact current compact pack is the sole reference used in the scores above. The machine-readable joins and source hashes are recorded in `manifest.json`; `test_payload_opened=false`.

## Candidate for the next model test

The event cohort supports testing a **COT-aware auxiliary loss on the SimVP AGRI forecast**, rather than another direct-COT loss tweak. Keep the current SimVP AGRI objective and forecast chain; freeze the current train-only R weights while retaining gradients from R's COT output back to predicted AGRI. Add one normalized COT loss on the two station patches, using CPP at valid pixels and current R-on-observed-AGRI elsewhere, with shared common support. Calibrate and freeze one loss weight from train-only gradient norms before validation. First verify that the added loss changes COT skill without materially degrading AGRI forecast metrics; then require an overall common-support COT RMSE improvement over the fixed SimVP→R baseline and no thick-cloud/low-COT guardrail violation. This is a new SimVP-training contract and still requires a separate preflight, train/validation provenance audit, and one-factor protocol before any PBS submission.
