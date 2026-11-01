# Staged near-zero/cloud classification and conditional COT regression

Date: 2026-09-24

## Motivation and evidence boundary

The weighted pilot reduced near-zero COT error at a cost to other strata. Hurdle v1 had an output-scale bug; v2 fixed the `log1p(COT)` initialization scale, but the jointly trained soft-gate model still increased near-zero false-cloud predictions. This follow-up follows the requested order strictly: first fit a classifier, then freeze it and fit separate class-conditioned regressors. It is a validation-only retrieval pilot against CPP product labels, not a clear-weather GHI experiment. CPP is a reference product, not independent physical truth.

Use the same verified real-AGRI train/validation pack, frozen R checkpoint, normalization, and sample split. Test rows/payloads, forecast AGRI, station GHI, and future products are excluded.

## Frozen model contract

1. Cache the frozen R decoder features from train/validation AGRI only; no gradient reaches the feature extractor.
2. Stage 1 fits a pixelwise `near-zero (CPP COT <0.1) / cloud (CPP COT >=0.1)` classifier using masked natural-frequency binary cross-entropy. Select its checkpoint by lowest validation Brier score. The threshold denotes a product-label class, not a meteorological clear-sky label.
3. Stage 2 freezes the selected classifier and the cached features. A low-COT regressor trained only on low-class training pixels predicts within `[0,0.1]`; a cloud regressor trained only on cloud-class training pixels predicts `>=0.1`. Both use `log1p(COT)` Huber loss and initialize from the frozen R output layer.
4. At inference, fixed `p_cloud >=0.5` selects the cloud regressor; otherwise select the low-COT regressor. There is no probability-weighted mixture and no validation-tuned threshold.

The single-head fine-tune from the paired v2 run and frozen R are retained as comparisons. The final regressor checkpoint uses the same balanced validation selector for all four normalized physical-COT RMSEs: near-zero, full grid, COT>=30, and station-centered 5x5. A pass requires near-zero RMSE below frozen R while full-grid, thick-cloud, and station-neighborhood RMSE are each no worse than frozen R. Train uses seed 42, 24 epochs per stage, batch 256, FP32; classifier AdamW lr `1e-3`, regressors AdamW lr `1e-4`, both wd `1e-4`.

## Execution and result

PBS job `210466.tc6000` ran on 2026-09-24 on `node21`, GPU 1 (one A100-SXM4-40GB, 8 CPUs), and completed in 65 seconds (`exit_status=0`). The frozen-R baseline recomputed on the same validation cohort within `1e-7` of the preregistered values. The pack had 11,978 train rows and 6,536 validation rows; `test_used=false` and the test payload count remained zero.

Stage 1 selected classifier epoch 11 by validation Brier (`0.06872`). Stage 2 selected the two conditional regressors at epoch 2 by the four-stratum normalized RMSE score. Physical-COT validation metrics:

| Model | Near-zero COT<0.1 RMSE | Full-grid COT RMSE | Thick COT>=30 RMSE | Station 5x5 COT RMSE | Near-zero false-cloud rate |
|---|---:|---:|---:|---:|---:|
| Frozen R | 0.7712 | 4.2462 | 11.9498 | 3.9436 | 11.79% |
| Staged classifier + experts | 0.8249 | 4.3382 | 11.6746 | 4.0458 | 5.92% |

The staged classifier sharply reduces false-cloud predictions and slightly improves thick-cloud RMSE, but misses more cloudy pixels (cloud recall `0.8913` versus frozen-R thresholded recall `0.9332`). Near-zero COT RMSE, full-grid RMSE, and station-neighborhood RMSE all worsen, so the candidate fails the retention gate. On near-zero-reference pixels, the staged prediction median and 90th percentile are `0.0024` and `0.011`, but its 99th/99.9th percentiles are `3.429`/`10.277` COT, with a maximum of `36.94`; the frozen R's corresponding tail is `3.111`/`9.720`, maximum `36.02`. Thus the hard classifier lowers the fraction above 0.1, while rare false-cloud cases still carry very large COT estimates and dominate squared error. Fewer false-cloud classifications do not by themselves guarantee lower numerical COT error. This does not support replacing frozen R or claiming improved GHI forecasts.

All local artifacts, including classifier/regressor checkpoints, validation probabilities, predictions, and logs, are stored under `results/cot_real_agri_staged_hurdle_seed42_20260924/`. No GHI model was retrained, and no test data were used.
