# Exploratory validation: SimVP vs SHADECast through one frozen COT inverter

Date: 2026-10-08. This is a validation-only, cross-experiment diagnostic. The comparison covers the new monthly split's same 3,140 validation sequences, 13 AGRI channels, 16 leads (+15 to +240 min), and identical valid-pixel masks. It never opens test payloads and does not select weights.

## Methods

1. The random-initialized monthly SimVP seed-42 checkpoint.
2. The monthly SHADECast deterministic nowcaster.
3. The monthly SHADECast diffusion ensemble-mean AGRI map (10 members, official PLMS, 20 steps, eta=0, EMA denoiser).

The same already-frozen regional R256 is applied to each forecast AGRI map. The reference is `R(real future AGRI)`, not observed COT. Thus COT scores are retrieval-proxy consistency, not physical COT accuracy. The R checkpoint is an older CPP-paired exploratory model; its CPP physical QA is unresolved, and it is not the monthly-contract R. To prevent direct frame leakage into this diagnostic, any target frame present in either the R model's training or checkpoint-selection validation records is excluded from **all** paired AGRI/COT supports. The exact overlap counts are recomputed from record metadata at run time and recorded. CPP files are not read.

The direct AGRI score is reported both on the entire monthly validation support and on the stricter daytime/all-13-channel R-safe support. COT proxy scores compare `R(forecast AGRI)` with `R(real future AGRI)` over only the latter. This is not a formal new-contract COT result; no monthly R exists and the source CPP chain is not fully quality-closed.

## Frozen provenance

- Monthly split ID: `hunan_monthly_block_2d_buffer1d_seed42_v1`.
- Manifest SHA-256: `f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2`.
- Train-only normalization SHA-256: `6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d`.
- SimVP best SHA-256: `f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4`.
- SHADECast VAE / nowcaster / diffusion SHA-256: `b5a940c943e51c8ccd37f2ab6331bcd21d268d8ed0b23d6538fb9ebec27069d1`, `76dd84a81d9d368e6b6fdf509fc731e501713dc33ee872c312407e3bcb3171ce`, `3a4985d17070ade7c79aaac3b4326b846b4f8b920b3e4a4540f9d25b815a7a95`.
- Frozen R256 best SHA-256: `74ddd7b2e78c29716260c9a790e0824659f2976d95b4598c3511c9816e77ad25`.

## Procedure and stop gates

First run a deterministic 32-sequence, full-pipeline one-GPU profile and confirm finite outputs, memory headroom, and output provenance. Only then run the full validation set on one 4-GPU PBS allocation if the profile justifies parallelism and live account queue is below the project cap of three unfinished jobs. No training, R adaptation, CPP QA, or test evaluation is authorized by this plan. Any hash/shape/mask mismatch, unsupported R provenance, test payload access, or nonfinite output stops the run without a scientific claim.

The final report will include direct AGRI short (+15…+60 min) and long (+120…+240 min) scores; retrieval-proxy log1p(COT) errors on common R-safe support; proxy thick-cloud miss/false-alarm rates at retrieved COT>=10; paired monthly-block bootstrap intervals; the exact R-leakage exclusions; and a plain-language source caveat. No CPP comparison is included because it is not an independent truth and its full-region QA is unresolved.

## Execution log (2026-10-08)

- Metadata-only preflight passed with 3,140 validation sequences and 4,025 unique target frames. It verified the manifest/norm hashes, all five model checkpoint hashes, SHADECast completion/gate receipts, and source metadata without opening image/CPP/test payloads.
- The old pair-bank record field is named `utc`; the bank builder joins it directly to the matched AGRI filename stem. Runtime asserts this identity and does not shift the keys. Of the 4,025 monthly validation target-frame keys, 431 were present in old-R training records and 191 in old-R checkpoint-selection validation records; 3,403 remain eligible. Every paired method's scored target support excludes the 622 overlapped timestamps.
- Unit-style synthetic checks passed for metric tensor shapes, direct/R-safe grouped summaries, and all four paired bootstrap paths. Python syntax and both PBS shell scripts passed compile/syntax checks.
- One-GPU profile submitted as PBS `212526.tc6000` on node21 (A100 40GB), 32 deterministic validation sequences, 10 PLMS members, 20 steps, member batch 2. This remains engineering profiling only. No full-set inference is submitted until finite outputs, throughput, and memory headroom are checked.
- Profile `212526.tc6000` exited with status 1 at the post-inference shape assertion; it produced only the immutable run-contract audit, no COT metrics or performance profile. The source-layout audit identified that SHADECast decode is already `[B,C,T,H,W]`, while the initial assertion incorrectly assumed a time-first decode. The comparison code now standardizes all outputs to `[method,channel,lead,y,x]`; the corrected profile uses a fresh `profile_r2` output directory so failed evidence is retained unchanged.
- Profile `212527.tc6000` passed the three model forward paths and PLMS sampling, then exited at inverse normalization because normalization mean/std were Python lists indexed as arrays. It produced only `profile_r2/run_contract.json`, no metric/performance outputs. The script now canonicalizes mean/std as float32 NumPy arrays immediately after metadata audit; the next attempt uses `profile_r3`, preserving both failed attempts.
- The corrected `audit_r3/preflight_audit.json` passed metadata/provenance checks with the same split/model hashes, 3,403 R-safe target frames, `cpp_used=false`, and `test_payloads_read=0`.
- Profile `212528.tc6000` completed on one node21 A100-SXM4-40GB: 32/32 deterministic validation samples, 389.20 s (0.08222 samples/s), peak reserved 7.70898 GiB, finite outputs. This is engineering throughput/memory evidence only. Full validation is now justified on four GPUs; no model-quality result has yet been produced.
- No training, CPP payload access, or test payload access has occurred.

## Full validation result and recovery (2026-10-08)

- PBS `212530.tc6000` ran the complete 3,140-sequence validation inference on four node21 A100 GPUs. Every rank completed all 785 assigned sequences and wrote chunks `0000`–`0015`; peak reserved memory was 7.71 GiB per rank. The PBS job exited 1 only after inference, when the postprocessor's R-safe count tensor retained an extra method axis and failed a broadcast during per-channel aggregation. No inference chunks were lost.
- Fixed the count-axis selection in both point aggregation and paired-bootstrap aggregation, and added a CPU-only `--mode aggregate` recovery path. Re-aggregation consumed only the already saved validation chunks/statistics; it did not rerun GPU inference, open CPP/test payloads, or change checkpoints.
- Inference source SHA-256: `0da737168dad7bd44b6d41014b4b26dee131dc9d055f87b11aa8236463a3f0e9`; corrected aggregation source SHA-256: `3efd1ae4ed18bc1169a90727f99d2b2541c6a3631bfd9fcb9dcad11eb8632472`. The remote result bundle is under `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_simvp_shadecast_val_20261008/results_v1/full`; verified copies are also stored in this local experiment folder.
- `audit_outputs.py` independently verified summary/archive hashes; all 3,140 sample indices exactly once; 26 monthly blocks; common masks/support; all point metrics recomputed from saved per-sample sufficient statistics; and all 10,000 paired monthly-block bootstrap deltas/CIs. Audit state: `PASS_INDEPENDENT_OUTPUT_AUDIT`. The final result is complete despite the original inference job's post-processing exit status 1; retain this distinction in any report.
- Final supports: 3,140 validation sequences and 4,025 unique future-frame keys; 431 keys overlap old-R training and 191 overlap old-R checkpoint-selection validation, leaving 3,403 R-safe frame keys. Those 622 overlapped timestamps are excluded from every paired AGRI/COT score. CPP and test payloads were not read.

### Point metrics

AGRI RMSE is in train-normalized units over the full spatial/QA support after old-R timestamp exclusions. “R-safe AGRI” uses the stricter all-channel/daylight/R-disjoint mask. COT is `log1p` retrieval-proxy RMSE from the same frozen old R; thick miss is against `R(real future AGRI) >= 10`, not CPP truth.

| Model | Lead group | AGRI RMSE (full) | AGRI RMSE (R-safe) | COT proxy RMSE, log1p | Thick-proxy miss |
|---|---|---:|---:|---:|---:|
| SimVP | +15–60 min | 0.2330 | 0.2764 | 0.4731 | 6.77% |
| SHADECast nowcaster | +15–60 min | 0.2204 | 0.2609 | 0.4425 | 5.55% |
| SHADECast diffusion mean | +15–60 min | 0.3794 | 0.4380 | 0.4832 | 10.29% |
| SimVP | +120–240 min | 0.4337 | 0.4923 | 0.8000 | 11.53% |
| SHADECast nowcaster | +120–240 min | 0.4587 | 0.5195 | 0.9139 | 14.18% |
| SHADECast diffusion mean | +120–240 min | 0.6455 | 0.7260 | 0.9155 | 25.73% |

### Paired candidate-minus-SimVP deltas

Intervals are 95% percentile intervals from 10,000 paired resamples of the same 26 monthly validation blocks. Negative is better for every metric below.

| Candidate | Lead group | Full AGRI RMSE Δ [95% CI] | R-safe AGRI RMSE Δ [95% CI] | COT proxy RMSE Δ [95% CI] | Thick miss Δ [95% CI] |
|---|---|---:|---:|---:|---:|
| Nowcaster | +15–60 min | -0.01265 [-0.01456, -0.01084] | -0.01548 [-0.01793, -0.01319] | -0.03060 [-0.04358, -0.01861] | -1.22 pp [-2.04, -0.57] |
| Nowcaster | +120–240 min | +0.02506 [+0.01797, +0.03266] | +0.02724 [+0.01741, +0.03778] | +0.11389 [+0.04870, +0.18590] | +2.65 pp [+0.50, +4.79] |
| Diffusion mean | +15–60 min | +0.14634 [+0.11764, +0.17724] | +0.16161 [+0.12413, +0.19917] | +0.01013 [-0.02006, +0.04095] | +3.51 pp [+2.22, +4.72] |
| Diffusion mean | +120–240 min | +0.21180 [+0.17442, +0.25121] | +0.23371 [+0.18158, +0.28666] | +0.11555 [+0.04544, +0.18691] | +14.21 pp [+11.74, +16.32] |

Interpretation is limited to this validation comparison: the deterministic nowcaster improves both AGRI and retrieval-proxy COT at +15–60 min, but is significantly worse at +120–240 min, including thick-proxy misses. The diffusion ensemble mean is substantially worse than SimVP for direct AGRI at both horizons; its short-lead COT proxy interval includes zero while thick misses are higher, and its long-lead COT/thick-miss results are significantly worse. This does not establish physical COT accuracy, CPP agreement, or independent-test performance; the old R's CPP chain remains unresolved.

Local result bundle: `results_v1/full/summary.json`, `complete.json`, `per_sample_totals.npz`, `run_contract.json`, and `full.pbs.log`; independent verifier: `audit_outputs.py`.
