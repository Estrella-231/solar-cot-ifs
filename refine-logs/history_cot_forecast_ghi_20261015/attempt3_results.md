# attempt3: dense validation and held-out initialization-date scoring

Status: PASS_INDEPENDENT_FP64_RECOMPUTE. Scope: old-contract Sili validation pilot; 4472 scored rows over 48 target dates. Model-selection and scoring initialization dates are disjoint. Metrics were independently recomputed in FP64 from saved per-row predictions. No new test payload was read.

## Overall result (all 16 leads pooled)

| Arm | No-COT RMSE | COT RMSE | Delta (COT - A), W/m^2 | RMSE change (positive = worse) | Date-block 95% CI for delta |
|---|---:|---:|---:|---:|---:|
| History-transport COT | 158.76 | 167.22 | +8.46 | +5.33% | [+2.55, +15.49] |
| History + hybrid COT | 158.76 | 166.79 | +8.04 | +5.06% | [+2.35, +14.76] |

## Selected leads

| Lead | Transport COT delta | Transport RMSE change | Hybrid COT delta | Hybrid RMSE change |
|---|---:|---:|---:|---:|
| +1 h | -0.82 | -0.53% | -1.83 | -1.18% |
| +2 h | +5.87 | +3.83% | +14.74 | +9.60% |
| +3 h | +6.63 | +4.35% | +11.65 | +7.65% |
| +4 h | +20.48 | +11.51% | +11.31 | +6.36% |

## Protocol and interpretation

- Nine runs: three seeds and three arms; each run used 1200 updates, with data order, initialization, optimizer, and all architecture inputs paired. Validation was checked every 10 updates.
- Sixty initialization dates were deterministically split by the frozen SHA256 rule into 12 checkpoint-selection dates (1383 rows) and 48 scoring dates (4472 rows). Selection used only the first group; scoring was performed once on the second.
- Overall pooled RMSE increased 5.33% for transport COT and 5.06% for hybrid COT; both date-block confidence intervals for the RMSE difference are above zero. The +1 h point estimates are slightly favorable but uncertain; later leads are generally worse.
- This improves protection against checkpoint-selection overfitting relative to the earlier sparse-checkpoint run, but remains an old-split, single-station exploratory pilot, not the paper-wide two-station result. The hybrid arm also uses frozen SimVP auxiliary forecasts and is not a pure COT-only head.
- The result says these forecast-COT inputs did not improve GHI on this held-out subset. It does not show that COT itself lacks value in same-time irradiance retrieval. Forecast-COT evolution error and GHI-head generalization are candidate explanations, not isolated causes.

Recompute script: `independent_score_attempt3.py`; figure: `forecast_ghi_attempt3_dateheldout.png`.
