# Paired R geometry and downstream GHI results

## Question

Does the solar-geometry correction in the AGRI→COT retrieval model (R) change
the downstream value of COT for station GHI? We first compare the two R models
on the same real AGRI validation scenes, then compare three otherwise matched
GHI heads.

## Real-cloud comparison of the two R models

The figure uses six deterministic validation scenes: per station, the minimum
near-zero CPP-reference center, the maximum thick-cloud CPP-reference center,
and the largest old-versus-corrected R disagreement. Every column uses the
same real AGRI C03/C02/C01 false-colour image, CPP retrieval reference, station
pixel, and shared scales. CPP is a retrieval reference, not independent COT
truth.

The physical COT patterns do change, but the direction depends on the scene.
For Sili's thick-reference case, the center estimate changes from 57.67 to
52.61 while the reference is 68.75. For Zhujia's thick-reference case, it
changes from 36.58 to 42.52 while the reference is 69.75. In the largest
disagreement examples, the corrected estimate moves closer to the reference at
Sili (59.50→40.81 versus 41.79) but farther away at Zhujia (57.44→40.81 versus
56.13). The examples therefore show a real input-response change, not a
uniform accuracy improvement.

In the full paired R validation comparison, corrected-minus-old equal-station
RMSE was +0.086 COT units (95% date-cluster interval −0.282 to +0.438). The
correction also lowered pooled MAE slightly while increasing RMSE. The
station-specific changes had opposite signs, so the R validation result did
not establish a reliable overall gain.

## Three paired GHI arms

All arms use the same 51,929 train/validation rows (47,666 train, 4,263
validation), two stations, 15-minute interval labels, and real AGRI at
T−30/T−15/T. Each station is trained independently with the same 3×3 local
head, 41,665 parameters, 30 epochs, batch size 1,792, and paired seeds 42/43/44.
The R models are frozen. The three arms are:

1. `A_history`: AGRI and corrected nominal-UTC geometry, with the standardized
   COT slot held at zero.
2. `AC_history_old_R`: the same inputs plus COT from the freshly retrained
   original-geometry R.
3. `AC_history_corrected_R`: the same inputs plus COT from the paired
   corrected-geometry R.

The primary estimator is the mean of station-specific RMSE, averaged equally
across the two stations and three seeds. Confidence intervals use 2,000 paired
BJT target-date cluster bootstrap resamples, conditional on validation-based
checkpoint selection.

| Arm or paired contrast | Validation RMSE (W m⁻²) | ΔRMSE (W m⁻²) | Relative change | 95% date-cluster interval |
|---|---:|---:|---:|---:|
| AGRI only | 108.236 | — | — | — |
| AGRI + old-formula R-COT | 107.818 | −0.418 vs AGRI only | −0.39% | [−2.236, +1.136] |
| AGRI + corrected-geometry R-COT | 107.784 | −0.452 vs AGRI only | −0.42% | [−2.294, +1.101] |
| Corrected-geometry R-COT − old-formula R-COT | — | −0.034 | −0.03% | [−0.365, +0.244] |

Both COT arms have slightly lower point RMSE than AGRI only, but all three
intervals include zero. The difference between corrected and old R is especially
small and unstable. Corrected R is lower-error than old R at Sili by 0.355
W m⁻², but higher-error at Zhujia by 0.288 W m⁻². Across the three paired seed
averages, corrected-minus-old RMSE is −0.054, +0.158, and −0.204 W m⁻².

MAE is lower in both COT arms: the equal-station/seed mean changes from 66.43
W m⁻² for AGRI only to 64.26 for old R-COT and 63.83 for corrected R-COT.
However, their mean bias becomes more negative (−0.46, −2.50, and −2.99
W m⁻², respectively), and the experiment did not define a separate MAE
uncertainty interval. This is a descriptive secondary result, not evidence of
a robust overall gain.

The sensitivity score that excludes 50 source-flagged long-zero rows gives
corrected-R-COT minus AGRI-only ΔRMSE −0.855 W m⁻², with a 95% date-cluster
interval [−2.653, +0.801]. Its interval also crosses zero; this score-only
subset does not change training or checkpoint selection.

## Interpretation

This matched experiment does not show that corrected solar geometry repairs
the downstream COT-to-GHI mapping. The old and corrected R outputs differ on
real cloud fields, yet changing between them shifts GHI RMSE by only
−0.034 W m⁻² on average, with uncertainty spanning both signs. Adding either
R-COT history yields a small favorable RMSE point estimate, but the paired
date uncertainty is too wide to distinguish it from no change. The MAE and
bias changes also show that COT alters the errors rather than uniformly
correcting them.

This result supports a narrow conclusion: under this two-station,
validation-only contract, the solar-geometry correction did not produce a
measurable downstream GHI improvement, and the incremental value of these
three observed COT frames remains unresolved. It does not establish that COT
has no physical value or predict future forecast-chain performance.

The GHI input includes the real AGRI frame at nominal T, whose satellite scan
may occur after T. This is an offline synchronous reconstruction of the same
15-minute GHI interval, not a causal forecast and not the SimVP→R deployment
chain. Test payloads were not read. No paper manuscript was changed.

## Reproduction and evidence

- Frozen experiment definition: [`PLAN.md`](PLAN.md)
- Live execution record: [`STATUS.md`](STATUS.md)
- Real AGRI/R figure and selected cases: [`visuals/paired_r_real_clouds.png`](visuals/paired_r_real_clouds.png), [`visuals/paired_r_real_agri_cases.json`](visuals/paired_r_real_agri_cases.json)
- GHI comparison figure: [`results/paired_three_arm_v1/analysis/three_arm_ghi_validation.png`](results/paired_three_arm_v1/analysis/three_arm_ghi_validation.png)
- Independent FP64 score and paired bootstrap: [`results/paired_three_arm_v1/analysis/independent_score.json`](results/paired_three_arm_v1/analysis/independent_score.json)
- Second row-weighted FP64 recomputation: [`results/paired_three_arm_v1/analysis/secondary_row_weighted_score.json`](results/paired_three_arm_v1/analysis/secondary_row_weighted_score.json)
- Row-aligned validation predictions: [`results/paired_three_arm_v1/analysis/paired_validation_scoring.npz`](results/paired_three_arm_v1/analysis/paired_validation_scoring.npz)
- R-bank mapping and validation-prediction agreement: [`audit/paired_r_cot_bank_manifest.json`](audit/paired_r_cot_bank_manifest.json), [`audit/r_bank_vs_saved_predictions.json`](audit/r_bank_vs_saved_predictions.json)
- Job outputs were produced under `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/r_geometry_downstream_ghi_20261016`; PBS jobs were `214295.tc6000` (R-COT bank) and `214305.tc6000` (three-arm GHI run).

The bank, checkpoints, and full per-run training traces remain on the server.
The repository contains the source scripts, validation-only predictions,
compact scores, figures, manifests, and logs needed to inspect this result.
