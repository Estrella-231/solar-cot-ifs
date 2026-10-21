# Attempt14: physical-COT loss alignment result (2026-10-22)

## Decision

The fixed physical-error loss reduced the three-seed mean validation COT RMSE from **11.431 to 11.064** (−0.367 COT, a 3.21% point reduction versus the log-only controls). The paired target-date bootstrap 95% interval for hybrid minus log-only is **[−0.735, +0.196]**, so the gain is not established. The candidate also fails the predeclared thick-cloud and persistence gates. **Do not train a GHI head from this bank and do not tune this patch-model loss again on these validation dates.** This is an exploratory validation result, not an independent test or a claim of general COT skill.

## Evaluation contract

All methods were independently rescored from saved predictions in FP64 on the common Hubei station-center support: **3,220 init/lead pairs**, **9,644 station-pixel records**, **16 BJT target dates**, and **786 common forecast initialization keys**. The cohort covers Niushou, Wangzhai, and Mizhuang, with leads +15 to +240 min. The target is CPP-retrieved COT and is used as a retrieval reference, not independent physical truth. Test and buffer payload reads are both zero. All six saved model-arm scores reproduced the stored evaluator RMSE to 1e-8.

The only candidate change was the fixed loss

```text
L = equal-lead masked MSE(log1p(COT)/log(101))
  + 0.10 × equal-lead masked MSE(COT/100).
```

Each training fit used random seed 42, 43, or 44 as paired, global batch 32, the same optimizer/LR schedule and train-date-only selection, and 1,810 updates. The five new fits ran as independent single-GPU processes on three allocated A800s in two waves, with no DDP. PBS `215537.tc6000` ran on the ZJNU college server, node22 GPUs 0, 3, and 4, exited 0, and used 175.826 s wall time. This run confirms concurrent multi-GPU execution; there was no same-workload one-GPU timing control, so no speedup factor is claimed.

## Paired results

| Seed | Log-only COT RMSE | Physical-loss COT RMSE | Hybrid − log-only |
|---:|---:|---:|---:|
| 42 | 11.991 | 11.024 | −0.967 |
| 43 | 11.085 | 11.161 | +0.076 |
| 44 | 11.218 | 11.008 | −0.210 |
| Seed mean | 11.431 | 11.064 | −0.367 |

Against historical COT persistence, the seed-averaged hybrid point RMSE is 11.064 versus 15.379, but the paired date-block interval for hybrid minus persistence is **[−7.970, +0.546]**, which also crosses zero.

For CPP-reference thick targets (COT ≥ 30), mean RMSE is **18.151** for the physical-loss candidate, **18.487** for log-only, and **17.566** for persistence. The hybrid helps this stratum versus log-only in two seeds but worsens it by 3.211 COT in seed 43; on average it remains worse than persistence. Thus the extra physical-unit term does not reliably solve thick-cloud underestimation.

| Predeclared gate | Result |
|---|---|
| All three hybrid seeds improve overall RMSE | Fail (seed 43 is worse) |
| Date-block paired hybrid − log-only 95% CI below zero | Fail: [−0.735, +0.196] |
| Hybrid beats persistence with date-block CI below zero | Fail: [−7.970, +0.546] |
| Thick-cloud RMSE no worse than log-only and persistence | Fail: 18.151 > 17.566 persistence |

## Interpretation and next step

This attempt supports a small overall point-estimate improvement from adding physical-COT emphasis to the training loss, but seed variability and the date-block interval prevent a reliable gain claim. The thick-cloud result remains below the required gate, so this model family is frozen on the current validation cohort. The next scientifically useful step is read-only recovery and inventory of the full-resolution Hubei CPP source (time, grid, and QC); a live check on 2026-10-22 found the source path absent on `zjnu-hpc`, while SSH to `FD-201` closed before command execution. This is a route/source blocker, not evidence that the data do not exist. No new training or GHI experiment is authorized by this diagnostic result.

## Reproduction and artifacts

- Independent paired scorer: `code/audit_attempt14_paired.py`
- FP64 result: `evidence/attempt14_paired_audit.json`
- Seed table: `evidence/paired_seed_metrics.csv`
- Per-arm evaluator scores, training receipts, epoch logs, and SHA-256 lists: `evidence/`
- Retry PBS job log: `evidence/pbs_215537.log`
- Remote checkpoints and saved prediction NPZs remain under `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/persistence_residual_attempt14_retry01/runs/`; their SHA-256 values are listed in the corresponding evidence files.
- The failed first allocation `215536.tc6000` is preserved separately under `persistence_residual_attempt14/evidence/failure_215536/`; it made no optimizer updates.
