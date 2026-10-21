# Persistence-residual COT forecast — attempt13 result

**Status: validation-only exploratory result; the predeclared COT gate is not passed. No GHI-head experiment was started.**

## Question

Does anchoring each forecast to the most recent valid historical COT, then learning a bounded residual, improve the existing station-patch COT forecast? This tests one output-parameterization change. It does not test a new AGRI extrapolator, a full-region model, or COT's effect on GHI.

## Model and execution

The model receives eight historical 16×16 AGRI-to-frozen-R COT maps, their validity masks, and solar-geometry channels; known future solar geometry is also supplied. It predicts each future log-COT as the last valid historical log-COT plus a bounded `tanh` residual. The final layer starts at zero so the initial model is exactly persistence. No future AGRI, CPP, COT label, or GHI is an input.

Training used the existing Hubei station-patch train/validation cache, seed 42, global batch 32, 10 epochs / 1,810 updates, and a date-block holdout made only from training dates for checkpoint selection. The best checkpoint was epoch 5. The monthly validation was scored once after training, not used to select the checkpoint. PBS `215535.tc6000` completed successfully on Zhejiang Normal University node22, physical GPU0, with one A800; walltime was 84.305 s. Test and buffer payload reads were both zero.

The same-cache fixed-global-batch profile had throughput 2,041.58, 1,845.11, and 1,572.77 samples/s on 1, 2, and 4 A800s. This small model was therefore run on one GPU because it was the measured fastest option; multi-GPU was benchmarked rather than assumed to be faster.

## Main validation comparison

All COT numbers below are evaluated only where the CPP target retrieval and initial historical COT are valid. CPP is a satellite retrieval reference, not independent physical truth. This Hubei station-patch diagnostic is not an independent test result.

| Measure | Learned residual forecast | Historical persistence | Interpretation |
|---|---:|---:|---|
| Equal-station/equal-lead RMSE | 11.991 | 15.379 | Point estimate is 22.03% lower |
| RMSE difference, forecast − persistence | −3.388 | — | 95% paired date-block interval: [−7.165, +1.060]; includes zero |
| Equal-station/equal-lead bias | +1.026 | +7.425 | Forecast removes much of persistence's positive bias |
| RMSE by station (Niushou / Wangzhai / Mizhuang) | 11.575 / 12.320 / 12.078 | 14.848 / 15.617 / 15.672 | Point estimate favors forecast at all three sites |
| RMSE at 1 h / 2 h / 3 h / 4 h | 11.414 / 11.830 / 12.004 / 12.290 | 14.733 / 15.516 / 15.454 / 16.116 | Point estimate favors forecast at each lead |

The 3,220 init/target/lead pairs cover 16 BJT dates and yield 9,644 station-center records. The windows overlap; these are not 9,644 independent weather events. The date-block confidence interval crosses zero, so the apparent overall gain is not established.

## Thickness and event behavior

Thickness groups are defined by CPP-retrieved target COT:

| CPP retrieval group | Forecast RMSE | Persistence RMSE | Forecast bias | Persistence bias |
|---|---:|---:|---:|---:|
| Near-zero, COT < 1 | 9.926 | 15.301 | +4.406 | +5.980 |
| Thin, 1–10 | 4.476 | 6.810 | +0.735 | +3.405 |
| Moderate, 10–30 | 8.737 | 17.878 | +3.735 | +14.571 |
| Thick, ≥30 | 19.784 | 17.566 | −8.194 | +7.257 |

For CPP COT≥30 events, POD/FAR/CSI are 0.703/0.225/0.584 for the forecast and 0.873/0.433/0.524 for persistence. The residual model makes fewer false cloud detections and improves CSI, but misses more thick retrievals; its thick-cloud RMSE remains worse than persistence. The thick-cloud problem therefore is reduced in bias, not solved.

## Paired comparison with the previous learned model

To isolate the candidate change on the exact records available in the existing attempt9 contextual file, saved attempt13 predictions were joined by `(station, init_time, target_time, lead)` to the same 1,824-key subset. An independent FP64 script reproduced the existing attempt9 and persistence scores before adding attempt13:

| Model on the 1,824-key subset | RMSE | MAE | Bias |
|---|---:|---:|---:|
| Previous learned forecast (attempt9) | 13.871 | 9.578 | −0.318 |
| Persistence-residual candidate (attempt13) | 12.947 | 7.944 | +2.200 |
| Historical persistence | 16.097 | 9.505 | +7.545 |

Attempt13's point RMSE is 0.924 lower than attempt9 on this subset, but the paired date-block 95% interval for `attempt9 − attempt13` is [−1.230, +2.321], which crosses zero. On CPP-thick cases in this smaller subset, attempt13 RMSE is 21.439, improved from attempt9's 23.415 but still worse than persistence at 18.551. This is a post-hoc validation comparison on a previously examined cohort, not independent confirmation.

## Decision and limits

The candidate improves the point estimate against persistence and the earlier learned model, but neither date-block interval establishes a reliable gain. Thick-cloud RMSE remains worse than persistence. The G1S COT gate therefore remains **not passed**: do not train paired GHI heads, expand seeds, or claim that COT improves GHI forecasts from these data. Continue the full-region line only after the read-only CPP source/access gate is resolved; this station-patch pilot is not a substitute.

The saved scoring JSON reports 3,468 values outside the nominal 0–100 COT range. This is a numerical boundary artifact, not actual overflow: all saved normalized predictions are within [0,1], and FP64 `expm1(log(101))` evaluates to 100.00000000000003. The reported count should not be interpreted as physical COT beyond 100.

## Reproduction artifacts

- Full validation score: `evidence/validation_score.json`
- Saved predictions: `evidence/validation_forecast_predictions.npz`
- Best and last checkpoints: `evidence/best.pt`, `evidence/last.pt`
- Paired attempt9 rescore: `evidence/paired_attempt13_vs_attempt9_common_support.json`
- Paired rescore code: `compare_attempt13_attempt9_common_support.py`
- Physical-range check for the `expm1` upper-endpoint rounding: `evidence/prediction_bound_audit.json`, reproduced by `audit_attempt13_prediction_bounds.py`
- Training completion receipt and PBS output: `evidence/training_complete.json`, `evidence/training_215535.tc6000.log`
- Server SHA-256 receipt: `evidence/SHA256SUMS`
- Full execution contract and status: `EXPERIMENT_PLAN.md`
