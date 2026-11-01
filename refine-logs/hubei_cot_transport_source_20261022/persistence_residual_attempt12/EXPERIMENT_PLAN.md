# Persistence-residual COT candidate — attempt12

Status: preflight-debug retry. Attempt10 and attempt11 are preserved failed attempts; neither produced optimizer updates or checkpoints.

The candidate network, train/validation cache, target/loss, seed42, global batch32, 10-epoch/1,810-update cap and train-only date-block selection are unchanged. Attempt12 adds an ERR trap to identify the exact failing preflight line. It may train only after the cache/source/profile/GPU assertions and persistence-residual smoke all pass. All outputs use this new directory.

The existing same-cache global-batch32 throughput measurements remain 2,041.58 / 1,845.11 / 1,572.77 samples/s for 1/2/4 A800s, so one GPU remains the time-efficient choice for this 16x16 model. This is exploratory COT-only work: no GHI training, test read or buffer read.

Server-side unit smoke and `bash -n` passed. PBS `215534.tc6000` was submitted on local 2026-10-22 after live `qstat -u slfu` showed zero unfinished jobs. It received node22 GPU index0 and exited status1 after 3.98 seconds. The ERR trap identified line 28: the base model assertion searched in `cache_attempt6`, but the module lives in the project's `scripts/` directory, already present on the trainer's `PYTHONPATH`. This was a script-only error before data loading; preserve attempt12 logs. Attempt13 corrects only this assertion.
