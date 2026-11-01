# One-time frozen-checkpoint monthly-validation score

**State: prepared; inference only. No model fitting or checkpoint selection is allowed in this follow-up.**

## Question

Does replacing full-R training-history COT with date-held-out OOF-R training-history COT, as in attempt15, improve station-center future-COT predictions on the frozen Hubei monthly validation split? The attempt15 internal train-date result favored OOF-history by 5.13% at the point estimate but its date-block interval crossed zero. This follow-up scores those already frozen checkpoints once on monthly validation; it does not alter the weights or select a new epoch.

## Fixed inputs and support

- Use only the audited `cache_attempt5/cache` train/validation cache and the six attempt15 `best.pt` checkpoints. The validation history COT is final frozen-R output in both arms; only the training-history source differs between the two trained arms.
- Cache audit must remain `PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED`, with `test_payloads_read=0` and `buffer_payloads_read=0`. The validation rows and CPP fields are read only for inference-time inputs and post-inference scoring; no future target is passed to the model.
- Before inference, verify all six checkpoint receipts, architecture width, common model-input shapes, cache split/key integrity, and the three station-center indices. Abort on any mismatch.
- Expected metadata-only coverage in the audited cache: 1,855 validation COT-bank frames and 4,116 init/lead pairs before CPP/site-pixel quality masks; all 16 leads are present. The final score records actual common supported keys and target dates.
- CPP is a retrieval reference, not independent physical truth. This is a monthly-validation diagnostic for the station-center pixels, not full-region skill or independent test.

## Inference and score

Run the six frozen models as independent inference processes, one process per allocated GPU; no DDP or parameter updates. Prefer six A800s in one PBS allocation; if only four are granted, run four models then the remaining pair in a second wave within that same job. Save each model's prediction array and a hash-bound receipt. The input arrays, model weights and sample order are identical across arms except for the trained history source encoded in the checkpoint.

Score the intersection of all six prediction key sets on valid station-center CPP targets and valid causal history pixels. Report physical-COT RMSE/MAE/bias with equal station/equal lead weighting, paired date-block intervals for OOF minus full-R and OOF minus persistence, per-station and per-lead scores, CPP-reference thickness strata and thick-cloud detection. Do not tune on this score.

## Decision

Apply the attempt15 five checks unchanged: point improvement over full-R and persistence, date-block interval entirely below zero versus full-R, all three seed point directions favorable, no site regression above 5%, and thick-reference RMSE no worse than either comparator. If any fail, stop this OOF-history candidate without further fitting on these data. If all pass, record a station-patch COT result only; it still does not authorize a GHI claim, full-region claim, or independent-test claim. Keep test and buffer closed throughout.
