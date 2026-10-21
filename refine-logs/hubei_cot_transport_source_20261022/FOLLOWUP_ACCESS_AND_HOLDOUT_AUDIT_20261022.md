# Follow-up source-route and inner-holdout artifact audit

Date: 2026-10-22 (Asia/Shanghai)

## Why this check was made

The current Hubei COT plan blocks another station-patch training run on the already exposed monthly validation cohort. Before proposing any training-only correction, this audit checked whether the completed attempt14 package contains per-case predictions from its train-date inner holdout, and whether the original full-resolution CPP source route has become available.

## Findings

1. **No new CPP route was recovered.** A fresh batch-mode SSH probe to `FD-201` closed before running a remote command; the same happened for fallback `FD-107`. The reachable `zjnu-hpc` login reported the expected `/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP` directory absent. This is evidence about the current routes/mount only, not proof that the upstream CPP product is absent.
2. **No train-only per-case forecast file is present in the staged attempt14 package.** The training receipts and JSONL files contain epoch-level train loss and inner-holdout RMSE, and record the held-out dates/window count. The saved score JSONs contain aggregate validation metrics and prediction keys, not per-example prediction values. The staged attempt14 tree has no inner-holdout prediction array. This prevents an honest residual-calibration analysis from the existing attempt14 artifacts without a new inference/export run.
3. **The live ZJNU account queue is empty, but that does not grant GPU access.** `qstat -u slfu` returned no jobs at server time 2026-10-22 06:25 CST. No GPU was allocated to this task, and no PBS job was submitted.

## Decision

Do not tune another blend or loss against the exposed validation dates, and do not start another patch-model training run under the current plan. The regional source/time/grid/QC gate remains the next scientific unlock. If that gate passes, prepare the full-resolution inventory first; then profile the authorized regional run on 1/2/4 GPUs (and expand only when the live allocation and measured throughput justify it), keeping the fixed global batch, optimizer updates, schedule, and loss reduction unchanged. Test and buffer payloads remain closed.
