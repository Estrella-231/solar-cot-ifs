# OOF historical COT input-origin diagnostic

Status: pre-registered, inference-only diagnostic. It does not train a model, score COT against CPP, score GHI, or release the full-region CPP source gate.

## Question

Does the current train-time historical COT bank differ materially from a date-held-out historical COT bank because the full R model was trained on the same dates used to produce its training inputs? This tests an input-provenance hypothesis only; it cannot establish that a future-COT forecaster or GHI forecast improves.

## Frozen inputs and scope

- 20,206 current Hubei train AGRI frames only; each BJT date is assigned to exactly one of five existing train-date folds.
- For each frame, infer with the corresponding existing R fold checkpoint, whose training excludes that date. Use that fold's frozen, train-only normalisation and the exact already-audited staged AGRI/geometry input.
- Compare to the existing frozen full-R bank for the same train keys, keyed by exact `(time_bjt, split)`; never compare by row position alone.
- Do not infer validation rows, read CPP targets/QC, GHI labels, test, or buffer payloads. Do not alter the current validation scores or select a model/loss.
- Report predictions as retrieved COT inputs, not independent COT truth.

## Multi-GPU execution

Use one PBS job on node22 with four allocated A800 GPUs and four independent fold-inference workers. This is data/model-parallel sharding across folds, not DDP: each worker loads one frozen R checkpoint and writes a disjoint fold shard. The fifth fold is queued onto the first worker that finishes. Record exact allocated device IDs, model/source hashes, per-fold rows and throughput. If the scheduler cannot grant four GPUs, do not silently run with unverified devices; preserve the job receipt and reschedule after a fresh resource check.

## Acceptance and analysis

1. Revalidate the staged-index and fold-map hashes, all five R checkpoint/source hashes, fold-pack status and train-only normalization before inference.
2. Require exactly 20,206 unique train keys assigned once, no validation/test/buffer key, no missing output, and a key-exact join to the existing full-R bank.
3. Apply the same day-mask support. Do not clip R outputs; report out-of-range daytime predictions separately and compute a common in-range [0,100] diagnostic subset.
4. Summarize in-sample-full-R versus date-held-out-fold-R map differences (RMSE, MAE, median/p95 absolute difference, correlation, distribution summaries, out-of-range fraction), overall and by BJT date. These are distribution diagnostics, not forecast-skill metrics.
5. Keep the G1 full-region CPP-source gate blocked and the failed G1S COT/GHI gates unchanged. Any downstream retraining requires a new scientifically justified plan and fresh, unexposed evaluation support.

## Data boundary

Only current train AGRI inputs, their deterministic solar geometry/day mask, train fold assignments, frozen R weights/norms, and existing train COT predictions are read. `test_payloads_read=0`, `buffer_payloads_read=0`, `cpp_target_or_quality_payloads_read=0`, `ghi_labels_read=0` are mandatory receipts.

## Attempt history

- Attempt 1 (`215539.tc6000`) was allocated four node22 GPUs (indices 0, 1, 3, 4), then exited during bootstrap before model inference. The immediate `qstat -f` query did not expose `exec_gpus` to the script's first parser call, so its four-device assertion failed. CPU time, memory use, and model/GPU inference were zero. Its PBS log and scheduler receipt are preserved under the attempt1 directory.
- Attempt 2 changes only the PBS allocation-field parser: it reads the documented `node22-gpu/<index>` entries and retries the scheduler query for up to 30 seconds while the allocation metadata becomes visible. The data, model, fold assignment, outputs, and analysis contract are unchanged; attempt2 has a new output directory.
