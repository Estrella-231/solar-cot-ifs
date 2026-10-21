# OOF-history source correction: outer train-date COT score

**Disposition: diagnostic complete; predeclared source-correction gate not passed; stop further training on this cohort.** This score is not a formal monthly-validation/test result and does not measure GHI forecast skill.

## Question and comparison

The experiment tests whether a future-COT forecaster is affected by a training/inference mismatch in its historical COT input. The control used final frozen-R histories for training; the OOF-history arm used date-held-out R outputs for training histories. Both used final frozen-R histories at inference. The architecture, three seeds, cache windows, CPP retrieval-reference targets/masks, geometry, optimizer and update budget were paired.

The scored value is the station-center pixel of the predicted 16×16 patch at Niushou, Wangzhai, and Mizhuang, compared with the coincident CPP COT retrieval reference. The outer dates were held out from model fitting and checkpoint selection, but were selected from the frozen training-date pool. They are therefore an **internal train-date holdout**, not monthly validation or independent test.

Common support: 639 forecast initializations, 2,761 init/lead pairs, 8,278 valid station-pixel records, and 14 BJT target dates. All 16 leads from +15 to +240 minutes are represented. Official monthly-validation, test, and buffer payload reads are all zero.

## Main result

Metrics use physical COT units and equal-station/equal-lead weighting. Negative paired differences favor OOF-history training.

| Method | Mean COT RMSE | Mean COT MAE | Mean bias |
|---|---:|---:|---:|
| Full-R history training | 11.332 | 7.365 | −3.914 |
| OOF-history training | 10.751 | 6.863 | −0.914 |
| Historical COT persistence | 13.054 | 7.905 | +5.148 |

OOF-history training lowers the point-estimate RMSE by 0.581 COT, or 5.13%, relative to the full-R control. The paired date-block 95% interval is **[−3.497, +1.990]**, so the data remain consistent with no improvement. Relative to persistence, OOF-history has a −2.304 point difference, with a 95% interval of **[−4.452, +2.012]**, which also crosses zero.

| Seed | OOF minus full-R RMSE | Paired date-block 95% interval |
|---:|---:|---:|
| 42 | −0.068 | [−2.272, +2.040] |
| 43 | −0.511 | [−3.608, +2.455] |
| 44 | −1.165 | [−4.720, +1.717] |

All three seed point estimates favor OOF-history, but every seed interval crosses zero.

## Site and CPP-reference thickness diagnostics

| Station | Full-R RMSE | OOF-history RMSE | Point change |
|---|---:|---:|---:|
| Niushou | 11.616 | 10.857 | −6.53% |
| Wangzhai | 11.074 | 10.996 | −0.71% |
| Mizhuang | 11.306 | 10.400 | −8.02% |

| CPP-reference stratum | Valid station-pixel records / target dates | Full-R RMSE | OOF-history RMSE | Point change |
|---|---:|---:|---:|---:|
| COT < 1 | 2,474 / 8 | 3.657 | 2.567 | −29.80% |
| 1 ≤ COT < 10 | 2,078 / 8 | 2.295 | 2.815 | +22.70% |
| 10 ≤ COT < 30 | 1,825 / 7 | 10.960 | 14.873 | +35.70% |
| COT ≥ 30 | 1,901 / 6 | 20.366 | 16.121 | −20.84% |

The strata are defined by CPP, which is a retrieval reference rather than independent cloud truth. Their support is concentrated on only 6–8 dates per stratum, so these point comparisons are exploratory and no thickness-stratum confidence interval is claimed. Thick-cloud RMSE bias improves from −17.587 to −9.070 COT, while thick-cloud POD rises from 0.478 to 0.746; FAR is similar (0.384 to 0.389). Thin and moderate CPP-reference strata worsen, so this is not a uniform accuracy gain.

The per-lead, per-seed scores and all saved-prediction FP64 metrics are in `evidence_remote_215542_215544/summary.json`; paired seed scores are in `evidence_remote_215542_215544/seed_pair_metrics.csv`.

## Gate and interpretation

Four predeclared checks pass: the seed-mean point estimate improves, all three seeds improve by point estimate, no station regresses by more than 5%, and thick-stratum RMSE is below both the paired control and persistence. The primary date-block confidence interval is not entirely below zero, so the combined gate **fails**. By the frozen stop rule, there will be no more source-correction training on these same dates.

The result is consistent with historical COT source mismatch affecting forecast behavior, especially for thick CPP-reference cases. It does not prove that OOF-history is physically more accurate than CPP, establish full-scene COT skill, demonstrate a GHI benefit, or pass independent evaluation. No GHI head was trained or rescored in this experiment.

## Compute and provenance

Six independent single-GPU fits were completed with three paired seeds. The first wave used four ZJNU node22 A800s (physical IDs 1, 3, 4, 5) for the seed42/43 fits; the second used two A800s (IDs 1, 3) for seed44. Each fit retained global batch 32, world size 1, 10 epochs and 1,350 optimizer updates; this used fit-level concurrency rather than DDP, so the per-model training contract did not change. The post-score `qstat -u slfu` snapshot contained no unfinished jobs.

- `215542.tc6000`: all four seed42/43 fits and outputs completed; PBS exited 1 only in the post-fit receipt writer. The preserved logs, checkpoints, predictions, and epoch records were checked; reconstructed receipts explicitly record the original exit status and reconstruction mode.
- `215543.tc6000`: failed before model initialization because the seed44 launcher pointed at the documentation directory; it produced no optimizer updates.
- `215544.tc6000`: corrected seed44 pair completed on node22 GPUs 1 and 3 and wrote `pair_complete.json`.
- Remote attempt root: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/oof_source_cv_attempt15`.
- Scorer SHA-256: `31d124d5e1aeef7844b10dde6aaa138441ed964f706f4d2437c956d93f6eef4e`.
- FP64 score JSON SHA-256: `84e80685a60cdf15d5f4172a38681b1870cf09599432b161803068cb11442fae`.
- Seed-pair CSV SHA-256: `cb2fd248c7d25b749102d51944dd85d10106a1ecf2fe4dfb0a838c831be6fe82`.

## Follow-up source-route check

A fresh read-only route check found `/public/home/slfu/ttzhou/swc/Data_Pool_3` now visible on `zjnu-hpc`, while `/home/Data_Pool_3` is absent. The mounted tree has no filename containing `CPP` or `COT` through depth 14; it exposes one CLP file, dated 2023-01-23 06:30 UTC, and neither expected Hubei GHI directory is present. `FD-201` and `FD-107` SSH BatchMode probes both exited 255 before their remote commands ran. This is a changed mount-visibility fact, but it does not unlock the required Hubei CPP target source or date coverage. The audit only listed paths and filenames/sizes; no product payload or test/buffer file was opened. Full receipt: `evidence_remote_215542_215544/source_route_recheck_20261022.txt`.

The next unlock is recovery of the correct full-region CPP source with auditable coverage and quality fields. Until that source gate passes, attempt15 remains a station-patch diagnostic; it does not authorize a GHI experiment or a claim that forecast COT improves irradiance forecasts.
