# Near-zero/cloud classification with conditional COT regressors

Date: 2026-09-24

## Question and scope

The preceding low-COT weighted pilot reduced CPP-reference COT error for pixels with COT `<0.1`, but worsened full-grid, thick-cloud, and station-neighborhood COT RMSE. This paired validation pilot tests a hurdle-style alternative: predict the probability that a valid pixel is in the near-zero class (`CPP COT <0.1`) or cloud class (`CPP COT >=0.1`), then estimate COT with separate conditional regressors. The class boundary is a CPP product threshold, not a meteorological clear-sky label. CPP remains a retrieval reference, not independent physical truth.

Use the same verified real-AGRI train/validation pack and frozen original-R checkpoint. Do not use forecast AGRI, future CPP as model input, station GHI, or any test payload. This is COT retrieval validation only; it does not retrain or evaluate the GHI head.

## Model and training contract

The candidate shares the frozen COT U-Net encoder/decoder and adds (1) a pixelwise cloud-class logit, (2) a low-COT regression head constrained to `[0,0.1]`, and (3) a cloud-COT regression head constrained to `[0.1,infinity)`. The class gate and both regression heads are trained jointly. The class gate is initialized from the train-only cloud prevalence. Conditional regressors are initialized from copies of the frozen R output layer.

The final physical estimate is the soft mixture

`COT_hat = P(near-zero) * COT_low + P(cloud) * COT_cloud`.

The threshold classes are derived from training CPP labels only. Use natural-frequency valid-pixel binary cross-entropy plus the mean of the low-class and cloud-class masked Huber losses on `log1p(COT)`; the predeclared classification coefficient is 0.05. FP32, seed 42, AdamW (`lr=1e-4`, `weight_decay=1e-4`), batch 256, and 24 epochs are shared with a single-head fine-tuned control initialized from the same frozen R.

Both arms use the same validation checkpoint score: the arithmetic mean of the four RMSE ratios against the frozen-R initial validation values for (i) near-zero COT, (ii) full-grid COT, (iii) COT>=30, and (iv) the station-centered 5x5 COT neighborhood. The candidate passes only if its near-zero RMSE improves and none of the other three RMSEs exceeds frozen R. Report class accuracy, precision, recall, Brier score, near-zero false-cloud rate, and physical-COT RMSE/bias for all four strata.

## Execution and result

PBS job `210463.tc6000` ran on 2026-09-24 on `node21`, GPU 3 (one A100-SXM4-40GB, 8 CPUs), and completed both arms for 24 epochs in 79 seconds (`exit_status=0`). Frozen-R recomputation passed the same-cohort validation baseline check. The pack recorded 11,978 training rows, 6,536 validation rows, and `test_used=false`; no test payload was opened. Artifacts are retained under `results/cot_real_agri_hurdle_seed42_20260924/`.

The common selector chose epoch 22 for both arms. The single-head control reached full-grid / near-zero / COT>=30 / station-5x5 RMSE of `4.1786 / 0.7771 / 11.1446 / 3.8586`. The v1 hurdle candidate reached `10.4447 / 0.2504 / 39.2266 / 10.0533`, respectively, and failed the retention gate. Its COT>=30 bias was `-37.6404`. Classification accuracy was `0.9129`, but that did not prevent severe conditional-regression failure.

Pixel-level diagnosis found the physical-scale mismatch: for COT>=30 pixels, mean reference COT was `42.47`, while the mixture prediction mean was only `4.83`. The source R head emits nonnegative `log1p(COT)` values; v1 copied its weights into the cloud expert but treated the output as physical COT before mixture. This pilot is an implementation failure and cannot be used to reject the hurdle idea. The corrected v2 uses explicit log-space expert outputs and has its own run contract and result directory. Neither pilot supports a GHI claim or replaces frozen R.
