# Persistence-residual COT forecast: physical-error alignment attempt14

**Status:** Training and independent paired validation audit complete; exploratory only; all four predeclared COT gates fail. No GHI experiment or full-region claim is supported.

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

## Retry receipt and single fix (2026-10-22)

PBS attempt `215536.tc6000` requested three A800s on ZJNU node22 but all three model processes failed before optimizer updates. The trainer rejects non-empty output directories; the launcher had redirected each process log to `runs/<arm>/train.log`, making the just-created output directory non-empty before the trainer's guard. The failed job and its logs remain preserved under the original `persistence_residual_attempt14/` remote directory; no checkpoints were produced and no existing attempt13 artifacts were changed.

Retry `persistence_residual_attempt14_retry01` changes only execution plumbing: each stdout/stderr log is written under a sibling `logs/` directory, keeping each model output directory empty at trainer startup. It also removes the recursive `tee` to the same path PBS is already writing, so the scheduler remains the single owner of `pbs.log`. Model, loss, sample keys, initialization, global batch, optimizer, update cap, selection rule, and validation scoring are unchanged. The retry uses a fresh output directory and preserves the failed attempt.


## Execution and validation receipt

The first qsub was rejected before job creation because the script had DOS line endings. It was converted to UTF-8 without BOM and Unix LF, passed remote `file` and `bash -n` checks, and was submitted as PBS `215537.tc6000`. The job used node22 GPUs 0, 3, and 4, exited 0, and recorded 175.826 s wall time. Three independent single-GPU fits ran at a time; per-fit global batch stayed 32 and DDP was not used.

Five fit outputs completed and were rescored from saved predictions. The independent audit requires exact common prediction initialization keys, reproduces each saved per-arm RMSE within 1e-8, uses 3,220 common init/lead pairs / 9,644 center-pixel records / 16 BJT target dates, and performs 5,000 paired date-block bootstrap replicates. The result and gate decision are in `RESULT_REPORT_20261022.md`; test and buffer reads are zero.

The fixed physical-COT term gives a three-seed mean RMSE point reduction of 0.367 (3.21%) versus log-only, but its 95% date-block CI is [−0.735,+0.196]. Thick-cloud RMSE is 18.151 versus 17.566 for persistence. Therefore the candidate fails the fixed continuation gate. Freeze this station-patch model family on these validation dates; do not train a GHI head from this bank. The next allowed work is read-only CPP source recovery and provenance inventory.
