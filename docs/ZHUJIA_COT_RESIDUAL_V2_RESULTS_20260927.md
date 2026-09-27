# ZhuJia COT residual pilot v2 results (2026-09-27)

## Scope

Exploratory validation-only pilot for ZhuJia. The fixed cohort contains 512 training sequences and 128 validation sequences (seed 42); each arm received 600 optimizer updates with batch 512. It compares the no-COT anchor, direct COT fusion, and bounded signed COT residual. The residual arm selects its checkpoint with step 0 included, so it can fall back exactly to the frozen no-COT baseline. No test data were opened. Sili weights were preserved.

## Paired audit

All three saved prediction files have the same 1,624 validation row IDs, lead indices, and observed GHI values. They represent one Zhujia station over 70 BJT initialization dates (2025-07-01 to 2025-09-10). The audit uses saved predictions and a 5,000-repetition paired initialization-day bootstrap (seed 20260927); source files and the audit JSON are retained under `results/zhujia_cot_residual_v2_20260927/`.

| Arm | RMSE (W m-2) | MAE (W m-2) | Bias (W m-2) | Best step |
|---|---:|---:|---:|---:|
| No-COT anchor | 127.295 | 93.014 | -11.703 | 132 |
| Direct COT fusion | 134.911 | 93.723 | -21.094 | 156 |
| Bounded COT residual | 127.295 | 93.014 | -11.703 | 0 |

The direct-fusion minus anchor RMSE difference is +7.615 W m-2; the paired day-bootstrap 95% interval is [-1.330, 17.340] W m-2 and only 5.34% of bootstrap draws favor fusion. The residual arm's selected predictions are exactly identical to the anchor because every trained correction was worse on validation than the zero-correction fallback. Therefore this pilot provides no evidence of a ZhuJia COT gain; it also does not establish that COT is generally unhelpful. The one-seed, small-cohort result is not suitable as a final paper claim or deployment decision.

At +15/+30/+45 minutes, anchor RMSE is 109.843/114.130/109.930 W m-2; direct fusion is 127.334/126.563/121.509 W m-2; selected residual equals the anchor. Direct fusion is worse at 15 of 16 leads (the only exception is +120 min). This suggests the next step is to inspect the residual-learning signal and its train/validation behavior on a broader, still validation-only cohort before allocating multi-seed runs. Keep the test set closed and retain the fixed Sili model.

The saved curves also show that 600 updates is not a stable endpoint for these arms: no-COT best validation is at step 132 then reaches RMSE 161.08 at step 600; fusion best is step 156 then reaches 167.67; residual never beats its step-0 anchor and reaches 147.64 by step 600. The training objective decreases within the residual run, yet the selected validation checkpoint remains step 0. This points to poor generalization of the learned correction on this small cohort; it does not isolate whether the cause is COT quality, correction capacity, initialization, or optimization. The next pilot should first widen the train/validation sequence cohort and predefine an update/early-stop schedule, then examine residual magnitude and skill by lead and cloud regime before considering multiple seeds.

## Reproduction and provenance

- PBS job: `210962.tc6000`, completed with exit status 0; wall time 13m22s.
- Remote run: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925/zhujia_cot_residual_seed42_20260927_v2`
- Audit script: `scripts/audit_zhujia_cot_residual_v2_20260927.py`
- Prediction, metric, completion, witness, and paired-audit files: `results/zhujia_cot_residual_v2_20260927/`
- Frozen Sili checkpoint SHA-256: `4d243d0c8c3be33cff83ad8307c6d2e1bcde59d8a9bae6cac17bfd8e87ea7d4c`
- The result is validation-only; `complete.json` records `test_used=false` and the kernel witness confirms exact zero-correction equivalence.
