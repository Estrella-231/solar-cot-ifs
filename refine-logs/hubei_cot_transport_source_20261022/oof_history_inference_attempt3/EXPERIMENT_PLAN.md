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
- Attempt 2 (`215540.tc6000`) passed the four-GPU bootstrap on A800 indices 0,1,3,4, then all four fold workers failed the nonnegative/finite output check before saving a shard. Live R normalization shows `day_mask_mean=1.0` and `day_mask_std=1e-6`; normalizing night pixels literally therefore produced about `-1e6`, unlike the established full-bank inference which replaces off-day normalized values by neutral zero. Attempt 3 changes only this inference implementation to match that existing treatment and adds log-output diagnostics; it does not clip predictions or change model/data/analysis. Attempt2's logs and empty partial output directory remain preserved.

## Execution outcome (2026-10-22)

PBS `215541.tc6000` completed with `Exit_status=0` on ZJNU node22 using A800 GPU indices `0,1,3,4`. Four independent fold workers ran concurrently; fold 4 was assigned to GPU0 after fold0 completed. The receipt reports 20,206 train frames and 56.502 s for inference/audit; no validation AGRI inference occurred. All test, buffer, CPP/QC, and GHI-label read counters are zero.

The strict train-only v3 audit found 20,206 unique train keys across five folds (4,174 / 4,052 / 4,008 / 4,031 / 3,941). On the 2,545,664 common daytime pixels with both outputs in `[0,100]`, the date-held-out-minus-full-R map disagreement is MAE 2.926, RMSE 5.968, median absolute difference 0.667, p95 13.246, and pixel correlation 0.942. The common-support full-R and OOF means are 12.632 and 11.256 COT. The full-R output also has 6,970 daytime values above 100 with maximum 449,380; OOF has 2,564 above 100 with maximum 12,649. These unbounded values were not clipped; they are counted and summarized separately. Therefore the common-support agreement metrics do not describe the full raw tail.

A scope issue was found in the original generic auditor: its whole-bank finite scan and file hash touched existing validation model-output values. The first local train-only revision also hashed the entire bank file. Neither performed validation AGRI inference, read validation labels, or used validation values for metrics or selection. The accepted train-only v3 evaluator avoids that scan, hashes only selected train rows, and reproduces the same common-support metrics. This correction and the original scope deviation remain documented in `evidence/OOF_HISTORY_INPUT_ORIGIN_REPORT.md`.

The result indicates an input-source distribution mismatch worth accounting for, but it cannot establish COT accuracy against physical/reference truth or any GHI benefit. It does not pass the full-region CPP source gate or the G1S COT/GHI gates. No downstream training is authorized by this diagnostic. Reproducible report, figure, date-level CSV/JSON, analysis source, PBS logs, fold shards, and train-only audit artifacts are stored under `evidence/`.
