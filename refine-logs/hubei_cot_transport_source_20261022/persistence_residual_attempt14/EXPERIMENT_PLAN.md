# Persistence-residual COT forecast: physical-error alignment attempt14

**Status:** pre-registered train/validation diagnostic; no training result yet. This is an exploratory Hubei 16×16 station-patch experiment, not a full-region result, independent test, or GHI experiment.

## Why this experiment

The previous persistence-residual candidate had lower overall COT RMSE than persistence as a point estimate, but its date-block interval crossed zero and its CPP-retrieved thick-cloud RMSE was worse. A train-only audit of the exact forecast-pair supervision found 10,694,840 valid target-pixel exposures over 41,933 train init/lead pairs and 152 target dates. Pixels with CPP-retrieved COT≥30 account for 1,713,794 exposures (16.02%); their fraction is stable at roughly 15.86–16.24% across leads. This weakens a simple “there are too few thick-cloud labels” explanation.

The implemented objective is equal-lead masked MSE in normalized `log1p(COT)`, while the reported physical score is RMSE in COT units. For a small physical error around COT `c`, the log-space squared error scales approximately as `ΔCOT² / ((1+c)² log(101)²)`. Thus equal absolute COT errors at high thickness receive less training weight than at low thickness. The hypothesis is that adding a small physical-unit term will improve absolute COT accuracy, particularly for thick retrievals, without abandoning relative/log-space supervision.

## Single change and paired design

Keep the persistence-residual architecture, input tensors, masks, train/validation cache, date-grouped train-only checkpoint selection, optimizer, learning-rate schedule, global batch, sample order contract, update cap, and physical evaluation unchanged. Change only the loss:

```text
L = equal-lead masked MSE(log1p(COT) / log(101))
  + 0.10 × equal-lead masked MSE(COT / 100)
```

The physical coefficient `0.10` is frozen before evaluation; it will not be tuned on monthly validation. The physical component is computed only on valid target pixels, with invalid values zeroed before arithmetic and with the same global valid-count reduction as the existing loss.

Use paired random seeds 42, 43, and 44. Seed42's log-only arm is the already completed attempt13 checkpoint and predictions. Train fresh log-only and hybrid-loss arms for seeds43/44, and a hybrid-loss arm for seed42. This gives three same-seed comparisons while avoiding redundant seed42 control training. Every new run uses global batch32, width96, seed-matched initialization, the existing 10-epoch / 1,810-update cap, patience8, and the same date-grouped train-only holdout. Monthly validation dates are not used for checkpoint or loss selection.

## Compute plan

Run the five new fits in two waves within one PBS job on three allocated node22 A800 GPUs: one independent single-GPU fit per card, with no DDP. Each fit retains global batch32 and its own optimizer/update schedule. Wave one: hybrid42, log-only43, hybrid43. Wave two: log-only44, hybrid44. This uses multiple cards for independent paired runs and avoids the previously measured DDP slowdown of the small model. Before submission, query the full `slfu` queue and node allocation; count this as one unfinished PBS task. Do not assume a GPU is available from an empty queue. Preserve attempt13 and all prior outputs.

## Evaluation and fixed decision rule

Score saved predictions once on the existing common monthly-validation support, using the same CPP-valid target pixels and physical COT conversion as the prior report. CPP remains a retrieval reference rather than independent cloud truth. Report each seed and the seed-averaged paired result, paired by target BJT date; include equal-station/equal-lead RMSE, MAE, bias, per-site and per-lead scores, and COT≥30 RMSE/bias/POD/FAR/CSI. This validation cohort has already been examined in prior pilot work, so all outcomes remain exploratory and cannot be described as independent confirmation.

The loss-alignment candidate passes the local COT continuation gate only if all of the following hold:

1. All three hybrid seeds have lower overall RMSE than their same-seed log-only controls on common support.
2. The date-block 95% interval for the seed-averaged hybrid-minus-log-only RMSE is wholly below zero.
3. The hybrid seed-averaged overall score beats historical persistence with a date-block interval wholly below zero.
4. COT≥30 RMSE is no worse than both the matched log-only arm and persistence, and no station has a consistent material regression.

If any condition fails, stop tuning this loss/model on these validation dates. No GHI head is trained in this attempt. A passing patch diagnostic still does not unlock full-region claims; the raw full-resolution Hubei CPP source and its time/grid/QC inventory must be restored first. Test and buffer payloads remain closed.
