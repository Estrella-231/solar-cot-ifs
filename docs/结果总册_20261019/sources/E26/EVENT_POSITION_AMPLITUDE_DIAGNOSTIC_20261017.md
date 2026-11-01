# Position, timing and thickness diagnostics on preselected COT events

Date: 2026-10-17. Validation only; no test payload was opened.

## Evidence and limits

Four event initializations were selected before model-error inspection from the frozen CPP event manifest: one reference-change case and one deterministic-hash ordinary case for each Hunan station. All four are JJA because the source CPP product has no complete 17-frame sequences for other seasons. The new monthly validation sequence keys, station keys, and all 16 target times were checked exactly against saved model predictions and the original SimVP→R forecast bank. All methods use the same per-pixel intersection of the compact CPP validity mask and current R frame-valid mask.

CPP is a retrieval reference, not independent truth. The true-future-AGRI→R curve is a hindsight retrieval diagnostic, not a deployable forecast. These four selected cases are descriptive and are not an independent sample for significance testing.

## Event-equal scores

RMSE and bias are in dimensionless COT units. Thick CSI pools valid pixels across the four events and 16 leads; the centroid distance is in 16×16 grid pixels and is conditional on both reference and method detecting at least one COT≥30 pixel. Arrival is the first +15…+240 min at which the local 3×3 common-support mean reaches COT 30; misses are retained in the event table.

| Method | Event-equal patch RMSE | Event-equal local 3×3 RMSE | Pooled thick CSI | Event-equal centroid shift (px; both detect) | Event-equal local peak error |
|---|---:|---:|---:|---:|---:|
| SimVP_to_R | 8.197 | 9.467 | 0.006 | 3.797 | -17.815 |
| P_Hybrid | 8.025 | 9.119 | 0.000 | — | -17.942 |
| P_Hybrid_amp01 | 8.181 | 9.219 | 0.000 | — | -17.952 |
| future_observed_AGRI_to_R_oracle | 3.526 | 2.293 | 0.572 | 2.042 | -3.374 |
| Persistence_R | 7.799 | 8.560 | 0.102 | 8.836 | -19.210 |

## Case-level behavior on the two preselected maximum-change events

The Sili event initialized at 2025-07-09 13:15 BJT had a local 3×3 CPP-reference peak of 67.49. SimVP→R peaked at 13.09, detected 10/1,504 CPP-thick pixel-leads (CSI 0.0066), and missed the local-threshold arrival. P-Hybrid and its fixed-amplitude variant detected no CPP-thick pixel-leads. The direct P-Hybrid local 3×3 RMSE was 27.03; SimVP→R was 30.27. Applying the same frozen R to observed future AGRI gave local 3×3 RMSE 5.37, thick CSI 0.582 (1,003 hits, 220 false alarms, 501 misses), and the correct local-threshold arrival lead. This is evidence that the forecasted image sequence is a major limitation for this particular Sili event; it does not show that image forecasting is the only error source.

The Zhujia event initialized at 2025-07-09 14:30 BJT had no thick-cloud hits from SimVP→R, P-Hybrid, or the amplitude variant. Observed-future-AGRI→R achieved local 3×3 RMSE 3.79 versus 7.59 for SimVP→R, but thick CSI was only 0.105 (4 hits, no false alarms, 34 misses); the peak error remained −8.28 COT. Thus replacing the forecast image with the observed future image improves this event, but leaves a substantial CPP disagreement. The signature is consistent with a combination of image forecast error and R/CPP or case-specific mismatch; this small sample cannot separate them.

The hash-selected ordinary cases at both stations were predominantly clear and contain no thick-cloud support, so they function as clear-sky controls rather than evidence about thick-cloud recovery. See `event_rollup.csv` for all four events and `event_method_metrics.csv` for the underlying metrics, arrival, peak timing, and mask counts. Do not aggregate these four deliberately selected events into a validation skill estimate.

The next paper-relevant discriminator is to extend the same event-level decomposition to the complete set of CPP-supported validation cloudy sequences, stratified by station and by event severity, using a preregistered event definition and the existing saved forecasts only. This is an analysis, not a new training run. If the Sili-versus-Zhujia signature persists, it supports separate work on image-field evolution and retrieval/reference consistency; only after that broader diagnostic should a new forecast model or loss be specified.

Test-payload state: `test_payload_opened=false`.
