# One-time frozen-checkpoint monthly-validation COT score

## Question and scope

This score asks whether attempt15 checkpoints trained with date-held-out OOF-R history COT outperform checkpoints trained with full-R history COT when forecast on the frozen Hubei monthly validation split. The six checkpoints, validation cache, prediction inputs, station pixels, leads, masks, and metric definitions are unchanged. The forecasts were generated once by PBS 215551.tc6000 with no optimizer updates, then scored from the saved predictions by CPU-only PBS 215552.tc6000.

This is a retrospective validation diagnostic at three Hubei station-center pixels. CPP is a satellite retrieval reference, not independent physical truth. The result is not independent-test skill, full-region COT skill, or GHI evidence. Future CPP values and masks were read only after inference for scoring. Test and buffer payload reads are zero.

## Common support and headline result

The six prediction files share 786 validation initialization windows. Of 4,116 cache init/lead pairs, 3,220 had at least one valid station pixel, yielding 9,644 station-pixel records over 16 target dates (2025-08-22 through 2026-03-02).

| Model input source for training-history COT | Equal-site/equal-lead COT RMSE | MAE | Bias (prediction minus CPP reference) |
|---|---:|---:|---:|
| Full-R control, three-seed mean | 12.296 | 7.971 | -1.532 |
| OOF-R history, three-seed mean | 13.639 | 8.917 | +2.322 |
| Persistence | 15.379 | 9.253 | +7.425 |

OOF-R history is worse than full-R control by 1.343 COT RMSE units, a 10.92% increase at the point estimate. The paired date-block 95% interval for OOF-R minus full-R is [-2.592, +4.985], which crosses zero. The sign of the point estimate is therefore adverse, but the available validation dates do not establish a precise population difference.

All three OOF-R seeds have higher overall RMSE than their paired full-R seeds:

| Seed | Full-R RMSE | OOF-R RMSE | OOF-R minus full-R |
|---:|---:|---:|---:|
| 42 | 11.853 | 13.199 | +1.346 |
| 43 | 12.102 | 13.607 | +1.505 |
| 44 | 12.934 | 14.110 | +1.176 |

The OOF-R arm is worse at all three stations: Niushou +8.46%, Wangzhai +13.70%, and Mizhuang +10.65% relative to full-R. Its equal-site/equal-lead RMSE is also higher at each of the 16 forecast leads from +15 to +240 minutes.

## CPP-reference thickness strata

| CPP retrieval-reference stratum | Records | Target dates | Full-R seed-mean RMSE | OOF-R seed-mean RMSE | OOF-R minus full-R |
|---|---:|---:|---:|---:|---:|
| Near-zero, COT < 1 | 4,124 | 9 | 9.129 | 11.796 | +2.667 |
| Thin, 1 <= COT < 10 | 1,660 | 6 | 4.273 | 6.458 | +2.185 |
| Moderate, 10 <= COT < 30 | 1,831 | 6 | 8.379 | 16.971 | +8.591 |
| Thick, COT >= 30 | 2,029 | 6 | 21.473 | 16.975 | -4.498 |

The apparent thick-stratum improvement comes with a detection tradeoff. Across the three seeds, thick-event POD averages 0.581 for full-R and 0.779 for OOF-R, while FAR rises from 0.195 to 0.430 and CSI falls from 0.511 to 0.490. This suggests more aggressive thick-cloud prediction, including many additional false alarms. These are small-date-count CPP-reference strata, so they are exploratory retrieval-reference diagnostics rather than physical cloud-truth findings.

## Predeclared decision

The OOF-R candidate fails the frozen monthly-validation decision gate: it does not improve overall RMSE over full-R, its date-block interval is not wholly below zero, not all seeds favor OOF-R, and each station regresses by more than the 5% limit. It does beat persistence on overall RMSE and has lower thick-reference RMSE, but those two results do not override the failed primary gates.

Attempt15's internal train-date holdout had a favorable OOF-R point estimate but a date-block interval crossing zero. The monthly validation result points in the opposite direction. Because those supports differ, they must not be pooled; together they show that the candidate's apparent OOF-R advantage is not stable across the intended evaluation supports. Stop further fitting or threshold selection for this OOF-history candidate on this cohort.

## Provenance and reproducibility

- Frozen-model inference: PBS 215551.tc6000, node22, six A800s, GPU IDs 1, 3, 4, 5, 6, 7; six predictions completed, zero optimizer updates.
- Saved predictions and inference receipts: server path /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/oof_source_cv_attempt15/monthly_val_score_attempt16/retry04/predictions/.
- Scoring only: PBS 215552.tc6000, normal queue, node7, 8 CPUs, no GPU request, exit status 0, walltime 32 s.
- Score artifact: monthly_validation_score.json; paired support: common_scored_keys.csv; execution and prediction-input receipts: score_execution_receipt.json and prediction_input_manifest.json.
- Score JSON SHA-256: 6eb7a21a7efa41b210a65a237529b6a47dba285c9d158c9d938b0c93ebcfa1d3.
- Test payloads read: 0. Buffer payloads read: 0. Optimizer updates: 0.
