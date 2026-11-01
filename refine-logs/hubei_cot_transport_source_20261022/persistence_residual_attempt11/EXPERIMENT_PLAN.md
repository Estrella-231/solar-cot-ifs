# Persistence-residual COT candidate — attempt11

Status: corrected retry of attempt10 preflight only. Attempt10 remains preserved and failed before training.

The candidate code and the full train/validation/training contract are unchanged. The only repair is the remote GPU profile path: the archived 1/2/4-GPU measurements are at `gpu_training_attempt8/profile_v8/`. A fresh attempt11 directory is used for all outputs. The PBS script asserts the train/validation-only cache audit, the saved 1/2/4-GPU throughput ordering, the PBS GPU allocation, and the zero-initialized persistence-residual smoke test before starting training.

The 1/2/4-A800 benchmark at global batch32 measured 2041.58 / 1845.11 / 1572.77 samples/s, so one GPU remains the fastest setting for this 16x16 workload. Attempt11 retains seed42, global batch32, the 10-epoch/1810-update ceiling, and the existing train-only date-block holdout. It does not train GHI and does not read test or buffer payloads.

After a fresh full-account queue check, PBS `215533.tc6000` was submitted on 2026-10-22 local time. The pre-submit listing contained only completed attempt10 (`C`) and no unfinished work; the helper conservatively counted that completed row as one, still below the three-job cap. It received node22 GPU index0 and exited with status1 after 3.7 seconds. Despite correcting the profile path, the job log stops after the first two file checks and contains no training, cache-copy, or optimizer activity. Independent checks on node22 confirmed the expected audit, profile and candidate files are present. Preserve this as a failed preflight; attempt12 adds explicit path reporting and an ERR trap before it may train.
