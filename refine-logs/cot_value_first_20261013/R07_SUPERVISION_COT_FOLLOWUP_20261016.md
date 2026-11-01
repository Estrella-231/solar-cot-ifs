# R07 follow-up: SimVP supervision candidates through the frozen COT retriever

**Status:** completed validation-only comparison; not a training run.  
**Question:** do the already-completed equal-budget temporal-change or spatial-gradient supervision candidates improve forecast COT after the same frozen monthly R, relative to the MSE fine-tuning control and the original seed-42 SimVP baseline?

## Frozen comparison

All candidates use the existing `hunan_monthly_block_2d_buffer1d_seed42_v1` train/validation split and train-only normalization. The A/B/C checkpoints were each fine-tuned for 4,000 updates from the same epoch-10 seed-42 SimVP checkpoint, with the same sample order and effective batch 8. Only the loss differs: A=MSE, B=MSE plus temporal signed-change loss, C=MSE plus spatial signed-gradient loss. The model architecture and parameter count are unchanged.

Inference runs each completed checkpoint over all 3,140 validation sequences, using the same 16x16 station patch geometry and the frozen `monthly_station16_COTUNet_seed42_v1` retriever. The existing baseline full-validation bank is retained as the original SimVP reference. CPP is a sparse retrieval reference; it is not independent physical truth. `R(observed future AGRI)` is an offline proxy for the effect of the AGRI forecast error, not a deployable input or independent COT truth. No test payload is read.

## Required scoring

On common keys, report by station and each +15 to +240 minute lead:

- 3x3 local and full 16x16 patch COT RMSE, MAE and bias against CPP, only where CPP is valid;
- thick-cloud (`COT >= 30`) POD, FAR and CSI on that same sparse CPP support;
- forecast-to-`R(observed future AGRI)` discrepancy on the full validation cohort, explicitly labeled as a proxy-response diagnostic;
- matched persistence-R comparison and sample/date/pixel support.

Scoring verified checkpoint, manifest, normalization and frozen-R hashes; identical forecast target keys, CPP arrays and masks across banks; finite values; canonical lead spacing; and zero test reads. Keep overlapping windows and target dates visible in the report. CPP support spans 46 BJT target dates and remains a retrieval reference, not independent physical truth; date-block intervals describe this fixed validation split.

## Decision rule

This is a screening step, not proof of a paper claim. A candidate advances only if it improves 3x3 CPP-reference COT RMSE over both the MSE fine-tuning control and the original SimVP baseline without degrading thick-cloud CSI or the full-cohort proxy response. A passing screen requires a separately frozen multi-seed validation and then a matched forecast-GHI head experiment. A negative result closes this loss candidate; it does not imply that COT has no downstream value.

## Provenance

| Arm | Checkpoint | SHA-256 |
|---|---|---|
| MSE control | `hunan_simvp_supervision_20261003_v2/results_v2/mse/step_4000.pt` | `881f249aebf501220f451ca018d193d38a88746289c7a3ac1d16dce86f54665f` |
| Temporal change | `hunan_simvp_supervision_20261003_v2/results_v2/temporal/step_4000.pt` | `fea2d406459a3a91c6ce6d11146be63affaaadc16203733d075304aa40338c34` |
| Spatial gradient | `hunan_simvp_supervision_20261003_v2/results_v2/spatial/step_4000.pt` | `24ae31c5fe882588a4afc30495523f003f420f3f1228e1d9f9be9e97ad40c691` |

The output directory is uniquely named `results/simvp_supervision_cot_allval_20261017_v2/`; it did not overwrite the original full-validation bank or any prior experiment.


## 2026-10-16 execution attempt status

PBS `214497.tc6000` exited with status 1 before loading SimVP or running inference. All three checkpoint hashes and the split, manifest, normalization, and zero-test-read metadata matched the frozen contract; the completed training checkpoints do not contain a `geometry_contract` metadata field, which the inherited evaluator requires. The output directory contains no prediction bank, metrics, or completion receipt. This attempt is not an experiment result. Do not patch checkpoint metadata or weaken the gate. A new validation run requires independent evidence for the geometry contract and a versioned evaluator update; test remains closed.

## 2026-10-17 execution: corrected all-validation screen completed

The v2 evaluator and its dependencies were staged remotely after the previous v1 attempt was confirmed to have produced no inference output. Local CPU witnesses passed: exact CPP join on the 271-window audited sample (668,672 common valid pixels), pooled CSI check, paired bootstrap check, and Python compilation of all three v2 scripts.

A source-backed geometry receipt was generated in the target `swc` environment. It verifies the pinned 24-frame chronological manifest (`35,320 train / 3,140 validation / 3,202 test metadata rows`), train-only normalization, SimVP/data/training source hashes, and all three 4,000-update candidate checkpoint hashes. It records `history_8_then_target_16_v2` from source and manifest chronology; no checkpoint metadata was patched. The receipt reports zero test payload reads.

PBS `214498.tc6000` completed inference and PBS `214500.tc6000` completed final FP64 scoring. Local 3x3 station-and-lead-macro COT RMSE was 10.664 for original SimVP, 10.750 for MSE, 10.756 for temporal-change, 10.759 for spatial-gradient, and 20.487 for last-frame persistence. Candidate-minus-original paired date-block intervals all cross zero; persistence-minus-original is +9.823 (95% CI [+7.722,+12.154]). Thick-cloud CSI was 0.423, 0.431, 0.430, 0.428, and 0.306 in the same order. CPP covers 46 BJT dates and 183,564 local pixels, and remains a retrieval reference rather than independent truth. No test payloads were read. None of the three loss candidates passed the advance rule. Final metrics are in `results/simvp_supervision_cot_allval_20261017_v2/scored_final/`.
