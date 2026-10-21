# OOF-history source correction: train-date holdout experiment

**Status (2026-10-22):** All six paired fits and the FP64 outer-train-date score are complete. The seed-mean OOF-history RMSE is 10.751 vs 11.332 for full-R history training (−0.581 COT; −5.13% point estimate), but the paired date-block 95% interval is [−3.497,+1.990], crossing zero. Four of five predeclared gates pass; the uncertainty gate fails, so stop further source-correction model tuning on this cohort. CPP-reference thick-stratum RMSE improves (16.121 vs 20.366), while thin/moderate strata worsen; these are retrieval-reference diagnostics, not independent cloud truth. See `RESULT_REPORT_20261022.md` and `evidence_remote_215542_215544/`. Training used four A800s concurrently for seeds42/43 and two A800s concurrently for seed44, with unchanged per-fit global batch32 and 1,350 updates. The full-region CPP source inventory remains blocked. This is a separate 16×16 Hubei station-patch diagnostic using only frozen training dates; it does not use or replace official monthly validation, test, or buffer.

**Source-route follow-up (2026-10-22):** `/public/home/slfu/ttzhou/swc/Data_Pool_3` is newly visible on `zjnu-hpc`, but a filename-only inventory found no CPP/COT-named files and only one unrelated January 2023 CLP file; the expected Hubei GHI directories are absent. `FD-201` and `FD-107` SSH probes failed before remote commands. Full evidence is `evidence_remote_215542_215544/source_route_recheck_20261022.txt`; no product, test, or buffer payload was opened.

## Question and hypothesis

Does training a causal future-COT forecaster with date-held-out R-COT histories reduce the train/inference feature-source mismatch and improve future COT estimates when inference histories come from the final frozen R?

The only treatment difference is the historical R-COT source for forecaster training rows:

- Full-R control: existing final-R history maps for fit and outer-holdout rows.
- OOF-history arm: five-fold date-held-out R-COT maps for forecaster fit and inner-selection rows; final-R maps for the untouched outer holdout, matching the source available at inference.

Both arms use the same CPP retrieval-reference targets and masks, geometry, exact sample/window keys, model, loss, initialization seed, sample order, global batch, optimizer, learning-rate schedule, and number of optimizer updates. CPP is a retrieval reference, not independent cloud truth.

## Unexposed date-block evaluation

Use only keys whose frozen split is train. From train-only CPP-supported target-date keys, select two dates per month by fixed one-third/two-thirds quantiles; this uses keys/support metadata only, never COT values or model errors. Remove all other train dates within ±1 calendar day of selected dates. The selected dates form an outer train-date holdout; buffered and monthly-validation rows are excluded from both caches. Require at least 12 outer-holdout target dates and at least 200 supported init/lead pairs at every lead before fitting.

The immutable attempt15 cache was built from the existing train-only bank. It retained 14 outer target dates, 25 purged neighboring dates, 181 remaining fit dates, 5,344 train windows, and 639 outer windows. All 16 leads pass support gates (outer minimum 215 pairs/lead and 14 target dates/lead; nested inner-selection minimum 182 pairs/lead and 23 target dates/lead). The independent CPU audit passed exact pair/row alignment and confirms identical geometry, target maps/masks, and outer holdout histories between arms; only train-history COT source differs. It indexed no official monthly-validation, test, or buffer payloads. Reproduce with `build_oof_source_cv_caches.py` and `audit_oof_source_cv_caches.py`; receipt is `CACHE_AUDIT.json`.

For each arm, the existing central-date, train-only holdout selects best.pt; it is strictly nested inside the remaining fit dates and is disjoint from the outer holdout. Run exactly 10 epochs with identical paired update counts and use the same frozen architecture/loss contract. Set early-stop patience to 11 (greater than the epoch cap) so every arm receives the full, paired update budget; select the best checkpoint using only the nested train-date holdout. The outer date holdout is scored once after checkpoint selection. The official monthly validation, test, and buffer remain unopened.

## Paired run and compute

Use seeds 42, 43, and 44 in both arms, pairing each seed across arms. Global batch is 32; width 96; AdamW; learning rate 1e-4 with cosine decay to 1e-6; weight decay 1e-4; exactly 10 epochs; patience 11; same common train-window order and masked log-COT loss. Do not resume a previous checkpoint.

Request one node22 allocation with four A800 GPUs and run four independent one-GPU fits concurrently (no DDP), followed by the remaining two fits. This parallelizes independent paired runs while preserving each model’s global batch and update contract. The same patch model previously profiled 1/2/4-GPU DDP at 2,041.58/1,845.11/1,572.77 samples/s; therefore this plan uses multi-GPU fit-level concurrency, not slower per-model DDP. If four A800s are unavailable, wait for the allocation; do not silently switch the contract.

The attempt15 scheduler script stages the audited cache to node-local scratch once, pairs seeds 42/43 on four cards, then trains the seed-44 pair on two cards. It uses patience 11 with a ten-epoch cap so all six fits receive the same complete update budget. It verifies the allocated physical GPU IDs and refuses any GPU with a pre-existing compute process. Saved forecasts are independently scored by `score_oof_source_cv_outer_holdout.py` in FP64 on the exact common outer-holdout support.

Before qsub, inspect the complete slfu queue and node22 allocation state; count running/queued/held jobs, and keep the workspace-wide unfinished PBS total at or below three. Verify the actual allocated device IDs and model-to-GPU mapping inside the job.

## Outcomes and stop rule

Primary score: equal-station/equal-lead physical COT RMSE at the three fixed station pixels on identical CPP-valid and history-valid target pixels. Report paired date-block 95% intervals for OOF-history minus full-R, each seed, and the seed-averaged result. Also report MAE, bias, per-site, per-lead, COT>=30 RMSE/bias/detection, persistence context, exact common keys, and saved-prediction FP64 reproduction.

Proceed only if the seed-averaged RMSE point estimate improves, its paired date-block 95% interval is entirely below zero, at least two of three paired seeds improve, no site regresses by more than 5%, and thick-cloud RMSE is no worse than both the paired control and persistence. If any gate fails, stop source-correction model tuning on this cohort. A passing internal train-date diagnostic is still not an independent-test or full-region result; it does not by itself authorize a GHI-head retrain. Test and buffer remain closed.
