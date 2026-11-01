# Near-zero/cloud classification with conditional COT regressors: v2 scale-corrected pilot

Date: 2026-09-24

## Question and scope

The preceding low-COT weighted pilot reduced CPP-reference COT error for pixels with COT `<0.1`, but worsened full-grid, thick-cloud, and station-neighborhood COT RMSE. Version 1 of this hurdle implementation also failed because it copied an R output layer trained in `log1p(COT)` but interpreted that layer directly as physical COT for the cloud expert. Its artifacts and failed result are retained in the v1 run directory. This v2 keeps the original R output scale when initializing the conditional heads. It tests a hurdle-style alternative: predict the probability that a valid pixel is in the near-zero class (`CPP COT <0.1`) or cloud class (`CPP COT >=0.1`), then estimate COT with separate conditional regressors. The class boundary is a CPP product threshold, not a meteorological clear-sky label. CPP remains a retrieval reference, not independent physical truth.

Use the same verified real-AGRI train/validation pack and frozen original-R checkpoint. Do not use forecast AGRI, future CPP as model input, station GHI, or any test payload. This is COT retrieval validation only; it does not retrain or evaluate the GHI head.

## Model and training contract

The candidate shares the frozen COT U-Net encoder/decoder and adds (1) a pixelwise cloud-class logit, (2) a low-COT regression head constrained to `[0,0.1]`, and (3) a cloud-COT regression head constrained to `[0.1,infinity)`. Both experts are parameterized in `log1p(COT)`: the low expert lies in `[0,log1p(0.1)]`, while the cloud expert lies in `[log1p(0.1),infinity)`. The class gate and both regression heads are trained jointly. The class gate is initialized from the train-only cloud prevalence. Conditional regressors are initialized from copies of the frozen R output layer, whose output is also in `log1p(COT)`.

The final physical estimate is the soft mixture

`COT_hat = P(near-zero) * COT_low + P(cloud) * COT_cloud`.

The threshold classes are derived from training CPP labels only. Use natural-frequency valid-pixel binary cross-entropy plus the mean of the low-class and cloud-class masked Huber losses on `log1p(COT)`; the predeclared classification coefficient is 0.05. FP32, seed 42, AdamW (`lr=1e-4`, `weight_decay=1e-4`), batch 256, and 24 epochs are shared with a single-head fine-tuned control initialized from the same frozen R.

Both arms use the same validation checkpoint score: the arithmetic mean of the four RMSE ratios against the frozen-R initial validation values for (i) near-zero COT, (ii) full-grid COT, (iii) COT>=30, and (iv) the station-centered 5x5 COT neighborhood. The candidate passes only if its near-zero RMSE improves and none of the other three RMSEs exceeds frozen R. Report class accuracy, precision, recall, Brier score, near-zero false-cloud rate, and physical-COT RMSE/bias for all four strata.

## Execution and result

PBS job `210465.tc6000` ran on 2026-09-24 on `node21`, GPU 3 (one A100-SXM4-40GB, 8 CPUs), and completed both arms for 24 epochs in 70 seconds (`exit_status=0`). The frozen-R recomputation matched the same-cohort baseline within about `1e-7`. The contract records 11,978 train rows, 6,536 validation rows, and `test_used=false`; test data were not opened. Artifacts are retained under `results/cot_real_agri_hurdle_v2_seed42_20260924/`.

The same four-stratum normalized RMSE selector chose epoch 22 for both arms and epoch 23 for v2's hurdle candidate. Validation metrics are physical-COT RMSE:

| Model | Near-zero COT<0.1 | Full-grid COT | Thick COT>=30 | Station 5x5 COT | Near-zero false-cloud rate |
|---|---:|---:|---:|---:|---:|
| Frozen R | 0.7712 | 4.2462 | 11.9498 | 3.9436 | 11.79% |
| Single-head control | 0.7771 | 4.1786 | 11.1446 | 3.8586 | 9.83% |
| Hurdle v2 | 0.8682 | 4.0670 | 12.2307 | 3.7396 | 19.85% |

V2 improves full-grid and station-neighborhood RMSE but worsens near-zero and thick-cloud RMSE; its near-zero false-cloud fraction rises from 11.79% to 19.85%. It fails the retention gate and does not meet the near-zero objective. This is a validation-only COT result; no GHI model was evaluated and there is no independent test result. Keep frozen R for formal use.
