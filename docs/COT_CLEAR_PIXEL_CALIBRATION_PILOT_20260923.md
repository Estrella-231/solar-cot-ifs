# Near-zero-COT retrieval calibration pilot

Date: 2026-09-23

## Question and evidence boundary

The continued real-AGRI R reduces full-grid and thick-cloud CPP-reference error but increases error on pixels with CPP COT `<0.1`. This pilot tests whether a small training-only weight increase on those near-zero-reference pixels can reduce that regression while preserving the previous frozen-R full-grid, thick-cloud, and station-neighborhood levels. It is a COT retrieval validation pilot, not a clear-weather GHI experiment; CPP remains a product reference, not independent physical truth. No test split is read.

## Paired training contract

Both arms start from the same frozen original-R checkpoint and use the same real-AGRI train/validation pack, masks, normalization, sample order, COTUNet, FP32 precision, seed 42, AdamW (`lr=1e-4`, `weight_decay=1e-4`), batch 256, and 24-epoch limit. The control retains the equal-valid-pixel `log1p(COT)` Huber objective. The candidate changes only training-pixel weighting: a train pixel with CPP COT `<0.1` receives twice the base weight; all other weights remain 1. Validation never determines weights.

Both arms use the same predeclared checkpoint selector: lowest validation RMSE on valid pixels with CPP COT `<0.1`. For the selected checkpoints, report that metric plus whole-grid RMSE/MAE/bias, COT≥30 RMSE/bias, station 5×5 RMSE, near-zero bias, and the share of near-zero reference pixels predicted at COT≥0.1. The frozen initial R is retained as the benchmark. This is validation-only selection and its result is a development signal, not independent confirmation.

## Pass condition

The candidate is useful only if its selected checkpoint reaches the frozen-R near-zero RMSE (`0.7712`) while keeping whole-grid RMSE ≤`4.2462`, COT≥30 RMSE ≤`11.9498`, and station 5×5 RMSE ≤`3.9436`. If it only lowers near-zero error while exceeding any of those limits, record the tradeoff and do not replace the frozen R. Even a pass requires independent time validation and CPP provenance closure before any formal model change.

## Execution and validation result

PBS job `210458.tc6000` ran on 2026-09-24 on `node21`, one NVIDIA A100-SXM4-40GB, with 8 CPUs. It completed both arms for 24 epochs in 93 seconds (`exit_status=0`). The account queue was empty before submission. The training contract records 11,978 train rows, 6,536 validation rows, and `test_used=false`; test data were not opened. Checkpoints, epoch logs, validation predictions, and `complete.json` are retained under `results/cot_real_agri_lowcot_weighted_seed42_20260923/`.

The common validation selector chose epoch 2 for the control and epoch 7 for the weighted candidate. All metrics below are COT retrieval metrics on real observed AGRI validation inputs; CPP is the reference product. COT values and RMSEs use the product's physical COT scale.

| Model | COT<0.1 RMSE | COT<0.1 bias | Near-zero pixels predicted ≥0.1 | Full-grid COT RMSE | COT≥30 RMSE | Station 5×5 COT RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R initial | 0.7712 | +0.1530 | 11.79% | 4.2462 | 11.9498 | 3.9436 |
| Equal-pixel control | 0.7356 | +0.1403 | 11.20% | 4.3411 | 11.5731 | 4.0402 |
| Low-COT weighted | **0.5950** | **+0.1001** | **7.15%** | 4.4783 | 12.2702 | 4.1119 |

Against the frozen R, the weighted candidate reduces near-zero RMSE by 0.1762 and the near-zero false-positive share by 4.64 percentage points. It increases full-grid RMSE by 0.2321, COT≥30 RMSE by 0.3203, and station 5×5 RMSE by 0.1684. Against the paired equal-pixel control, it lowers near-zero RMSE by 0.1406 but raises full-grid RMSE by 0.1371, thick-cloud RMSE by 0.6971, and station 5×5 RMSE by 0.0717. It therefore fails the preregistered retention gate and must not replace the frozen R.

Interpretation is limited to the validation retrieval objective: a 2× training weight on CPP-reference COT<0.1 pixels corrects part of the positive near-zero bias, at a measurable cost to thick-cloud and whole-grid fit. This does not establish improved clear-weather GHI or downstream station forecast accuracy. No GHI model was retrained or evaluated in this pilot, and there is no independent test result. Keep the frozen R for formal use; any follow-up should first examine a separately named clear/cloud classification-plus-regression objective under a new preregistered validation contract rather than increasing this weight blindly.
