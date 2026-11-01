# G1S persistence-residual COT candidate — attempt10

Status: user-directed exploratory candidate; not regional COT, not GHI, and not independent-test evidence.

## Hypothesis

The current absolute-value decoder compresses an eight-frame history into a 2×2 latent and reconstructs each future COT map from scratch. The saved monthly-validation context shows strong negative bias for CPP-reference thick clouds. A fixed 50/50 forecast/persistence diagnostic reduced thick-stratum point error, with a wide overall interval. This motivates one architectural change: predict a bounded future **log-COT residual** around the most recent valid historical log-COT map, rather than regenerating absolute COT.

The candidate starts exactly at persistence and can learn growth/decay and spatial changes. It does not add optical-flow inputs, CPP, future AGRI, GHI labels, or new geometry channels.

## Frozen one-factor contrast

- Scope: the existing Hubei 16×16 three-station patch and its previously audited train/validation cache. No Hunan assets or full-region claims.
- History: the same eight AGRI→frozen-R COT maps and validity masks; same 24 geometry frames and future leads +15…+240 min.
- Target/loss: same CPP-retrieval targets, masks, physical range, per-lead masked MSE in `log1p(COT)/log(101)`.
- Changed factor only: `forecast_log = clamp(last_valid_history_log + tanh(delta_logits), 0, 1)`. The final decoder layer is zero-initialized so epoch-zero output equals persistence.
- Training: random initialization, seed 42, global batch 32, same optimizer/LR schedule, date-blocked train-only holdout and purge, maximum 10 epochs / 1,810 updates / patience 8, same stopping rule. Previous attempt9 checkpoint remains immutable; no resume or overwrite.
- Hardware: one A800 on ZJNU node22. The existing identical-compute profile measured one GPU faster than 2/4 GPUs (2,041.6 vs 1,845.1/1,572.8 samples/s); this residual head adds no meaningful compute, so DDP would waste wall time. Any larger future regional model will be profiled at 1/2/4 GPUs and distributed if throughput improves.
- Test and buffer payloads remain closed. No paired GHI head is trained in this attempt.

## Evaluation and gates

1. Run the same train-only selection protocol; save `best.pt`, `last.pt`, epoch records, the full validation prediction bank, and manifests/hashes.
2. Independently score the full 16×16 valid support and the exact existing three-station common support against persistence and the saved forecast-AGRI→same-R contextual baseline. CPP remains a retrieval reference, not independent physical truth.
3. Report equal-station/equal-lead RMSE, MAE, bias, date-block intervals, per-lead/site behavior, and COT≥30 RMSE/POD/CSI. Keep the validation reuse explicit: this is an exploratory follow-up on an already inspected cohort.
4. Do not claim a COT skill pass unless the frozen common-support score improves over both persistence and the same-R comparator with date-block intervals excluding zero, no material site regression, and thick-cloud behavior improves. A pass would still not establish GHI improvement; a later paired GHI experiment needs separate authorization and identical data/training contracts.

## Execution record

The local Python syntax check and the Torch smoke test passed. The Torch smoke ran in the ZJNU `swc` environment and verified output shape, bounded values, fallback to the latest valid historical pixel, exact persistence at zero residual, and a nonzero learnable residual.

The same-cache 1/2/4-A800 benchmark from attempt9 measured 2,041.58 / 1,845.11 / 1,572.77 samples/s at global batch32. As this head adds negligible compute, the measured fastest allocation is one A800; multi-GPU DDP would increase time for this small patch workload.

PBS `215532.tc6000` was submitted on local date 2026-10-22 (scheduler UTC timestamp 2026-10-21T19:56:50Z) to ZJNU node22 requesting one GPU. The immediate pre-submit full `qstat -u slfu` contained zero unfinished jobs; post-submit showed the job queued. It received node22 GPU index0 and exited with status1 after 3.8 seconds. The training log shows the first two path checks passed, then a preflight path assertion failed because the profile JSONs live under `gpu_training_attempt8/profile_v8/`, not the path encoded in attempt10. The cache arrays were not loaded, the smoke wrapper did not run, and no optimizer update/checkpoint was produced. Preserve the failure log and receipts; attempt9 remains read-only. The corrected script is in a new attempt11 directory.
