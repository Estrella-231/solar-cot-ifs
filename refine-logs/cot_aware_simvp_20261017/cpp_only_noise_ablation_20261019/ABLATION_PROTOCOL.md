# CPP-only COT auxiliary ablation (submitted; PBS 214884 running)

Question: does the mixed CPP + observed-AGRI-to-R pseudo-target auxiliary loss inject harmful label noise into future-cloud forecasting?

Evidence motivating the single-factor test: the active M1 training contract uses CPP where valid and R fallback elsewhere. The saved train-coverage diagnostic reports about 32% CPP and 68% R fallback pixels at every lead and both stations. A separate 1,024-window validation diagnostic on common valid support reports pooled R-vs-CPP RMSE 8.96 COT units, near-zero CPP (CPP<1) R mean 4.93 with 6.1% R>=30 at Sili, and thick CPP (CPP>=30) R mean 32.9 vs CPP mean 41.8 with only 56.4% R>=30. The audit explicitly labels CPP as a retrieval reference rather than independent truth; these discrepancies motivate a test but do not establish causality or GHI impact.

## One-factor comparison

- Control: current mixed-target M1 (CPP valid support plus R fallback).
- Candidate: same SimVP initialization, split, 32,000 ordered training rows, seed 42, batch 8, optimizer, schedule, 4,000 updates, AGRI objective, frozen R, validation and audit; only COT target source changes to CPP-only on common valid support. No R pseudo-labels fill CPP gaps.
- To avoid changing loss coefficient as a second factor, the candidate pins lambda_COT to the already calibrated mixed-M1 value 0.005891630984842778. The CPP-only train-only gradient-ratio calculation is recorded only as a diagnostic and does not set the coefficient.
- This estimates the effect of restricting supervision to CPP-supported pixels under the fixed coefficient; it does not separately identify label correctness versus the resulting spatial coverage reduction.
- Do not submit while M1 is active. After the M1 validation and independent same-mask audit, re-check full-account PBS queue and current output/manifest state. If the audit indicates the mixed fallback is a plausible degradation source, submit this one-factor candidate in its unique output directory. The PBS file contains a one-GPU profile gate and uses a unique lock/output.
- Compare candidate, original SimVP and equal-budget MSE control on same CPP support, including thick-cloud CSI, near-zero false alarms and AGRI guardrails. No test payloads. Do not claim future GHI gain until the separate saved-prediction A/AC GHI audit passes.

Status as of 2026-10-19 04:37 CST: submitted as PBS `214884.tc6000`; the job passed server-side SHA-256 checks and `CPP_ONLY_TARGET_WITNESS_PASS`. Its train-only preflight passed with the fixed lambda `0.005891630984842778`, CPP-only target source, frozen R, and `test_payloads_read=0`. The training process is active on node22 with one allocated A800 80 GB GPU; it is reading/preparing inputs and has not yet emitted the first step record. At the latest sample it used about 14 GiB GPU memory with near-zero GPU utilization, while process I/O counters increased; treat this as active preparation, not as training progress or a stall. The forecast-COT bank job `214883.tc6000` remains active at 14,912/33,253 sequences (477,184 forecast instances), with test and future-observed-AGRI exclusion flags false.
The PBS recipe first reuses the node-local raw-frame cache from M1 when it is present, since the source frames and cache key contract are identical; otherwise it falls back to a new private cache. The candidate is not run concurrently with M1.
The PBS job runs the standalone synthetic mask witness directly, without depending on pytest in the cluster environment.
