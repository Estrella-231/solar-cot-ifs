# Persistence-residual COT candidate — attempt13

Status: **COMPLETE — exploratory COT candidate; G1S skill gate not passed.** Attempts10–12 are preserved as preflight-only failures; none loaded the cache or made optimizer updates. Attempt13 trained, scored the monthly validation once, and is now closed without starting GHI training.

The model, data, loss, seed42, batch32, 10-epoch/1,810-update ceiling and train-only date-block selection are unchanged. The only functional repair is to assert the frozen base model at `$PROJECT/scripts/retrieved_cot_forecaster_20261013.py`, matching the import path already used by the trainer. The ERR trap remains enabled to record any new failure. The output path is new.

A same-cache benchmark measured 1/2/4 A800 throughput 2041.58/1845.11/1572.77 samples/s at the fixed global batch32. One GPU was selected because it was fastest for this small patch model; the actual run used one A800 and took 84.305 s. The run is exploratory COT-only; GHI training and test/buffer reads remain disabled.

The server Torch smoke and `bash -n` passed. PBS `215535.tc6000` was submitted on local 2026-10-22 after a live `qstat -u slfu` showed zero unfinished jobs (the only prior row was attempt12 in state `C`). It completed with `exit_status=0` on node22 physical GPU0; verified walltime was 84.305 s. The selected checkpoint is epoch5; the full run finished at global step1810. Train-only internal date-holdout RMSE was 8.82459. This is selection evidence, not monthly-validation skill.

## Closed result and decision

The saved predictions were scored by the separate FP64 validation evaluator on the frozen monthly validation support: 3,220 `(init,target,lead)` pairs, 9,644 station-center records, and 16 BJT target dates. All comparisons use the same CPP-valid and history-valid records; CPP is a retrieval reference, not independent truth. The test and buffer payload read counts are both zero.

| Validation comparison | Attempt13 | Comparator | Result |
|---|---:|---:|---|
| Equal-station/equal-lead COT RMSE | 11.991 | persistence 15.379 | Point reduction 22.03%; paired date-block 95% interval for forecast-minus-persistence RMSE is [-7.165, +1.060], crossing zero |
| Niushou / Wangzhai / Mizhuang RMSE | 11.575 / 12.320 / 12.078 | persistence 14.848 / 15.617 / 15.672 | All three point estimates favor attempt13 |
| 1 h / 2 h / 3 h / 4 h COT RMSE | 11.414 / 11.830 / 12.004 / 12.290 | persistence 14.733 / 15.516 / 15.454 / 16.116 | Point estimates favor attempt13 at each lead |
| CPP-retrieved COT≥30 RMSE | 19.784 | persistence 17.566 | Thick-cloud RMSE remains worse; attempt13 bias is -8.194 versus +7.257 |
| Thick event POD / FAR / CSI | 0.703 / 0.225 / 0.584 | persistence 0.873 / 0.433 / 0.524 | Fewer false alarms and higher CSI, but lower POD (more missed thick retrievals) |

The equal-station/equal-lead point RMSE is promising but not decisive: the date-block interval crosses zero, and thick-cloud RMSE still regresses. On a previously used 1,824-key common-support subset, an independent FP64 join against attempt9 and persistence records gives RMSE 13.871 (attempt9), 12.947 (attempt13), and 16.097 (persistence). The paired attempt9-minus-attempt13 date-block interval is [-1.230, +2.321], also crossing zero. CPP thickness on this smaller subset shows attempt13 thick RMSE 21.439 versus 23.415 for attempt9 and 18.551 for persistence; its thick bias improves from -22.477 to -5.482 but remains below the persistence estimate. This subset analysis is a validation diagnostic, not a fresh or independent evaluation.

Therefore the predeclared G1S COT gate remains **NOT PASSED**. Do not start paired GHI-head training, expand seeds, or claim a GHI benefit from this candidate. Continue the full-region route only after the raw CPP source/access gate is resolved. The validation score's `out_of_training_range_forecast_pixel_count_cot_lt0_or_gt100` is a floating-point boundary artifact: scaled predictions are within [0,1], and `expm1(log(101))` evaluates to 100.00000000000003 in FP64. It does not indicate forecasts physically above the model's 0–100 bound.

Reproducible paired rescore code and JSON are in this directory. Full details and limitations: `RESULT_REPORT_20261022.md`.
