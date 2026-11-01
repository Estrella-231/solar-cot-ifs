# COT-trajectory GHI H-CV result — 2026-10-21

## Question and scope

This fixed-budget experiment asked whether the recent trajectory of forecast COT maps adds useful information to a station GHI head beyond the target-lead COT map. It is a date-block out-of-fold cross-validation wholly within the outer-training partition H. It is development evidence, not an untouched validation or independent test result. No test or buffer numeric payload, future observed AGRI, CPP, or GHI was used as model input.

The three arms share the same station-neighbourhood AGRI, solar geometry, lead encoding, architecture, folds, initialization seeds, normalization rule, and 30-epoch training budget: no COT; target-frame COT only; four ordered maps at L−45, L−30, L−15, and L. The primary contrast was frozen in advance as trajectory minus target-frame COT. Each method predicts all 16 leads (+15 to +240 min); the short-lead gate is +15 to +60 min.

## Results

Absolute scores below are the arithmetic mean of the six station-by-seed RMSEs (two stations × three seeds), with each model's out-of-fold rows pooled across the five date folds before calculating RMSE. They are descriptive; the confidence intervals below apply to paired differences.

| COT input | All leads RMSE (W m⁻²) | +15–60 min RMSE (W m⁻²) | All leads MAE (W m⁻²) | All leads bias (W m⁻²) |
|---|---:|---:|---:|---:|
| None | 150.05 | 140.84 | 95.87 | +1.07 |
| Target-frame map | 150.93 | 141.26 | 96.10 | +2.38 |
| Four-map trajectory | 152.82 | 142.80 | 97.00 | +3.27 |

Paired, station-and-seed-equal differences (positive means worse):

| Contrast | All leads ΔRMSE (95% two-day-block CI) | +15–60 min ΔRMSE (95% two-day-block CI) |
|---|---:|---:|
| Trajectory − target-frame (primary) | **+1.89** [−0.03, +4.45] | **+1.54** [−0.79, +4.74] |
| Trajectory − no COT | +2.77 [−0.73, +6.69] | +1.96 [−1.66, +6.17] |
| Target-frame − no COT | +0.88 [−1.47, +3.28] | +0.42 [−1.70, +2.48] |

The predeclared gain gate failed all five conditions. The primary estimate is worse for both stations on average: trajectory−target-frame is +1.72 W m⁻² at Sili and +2.06 W m⁻² at Zhujia. Seed consistency is poor: region-equal primary differences for seeds 42/43/44 are +1.41, −0.55, and +4.81 W m⁻², so only one of three improves. The descriptive trajectory−target-frame RMSE difference is positive at every lead, ranging from +1.35 to +2.58 W m⁻². Removing any one two-day block leaves the primary estimate positive (+0.82 to +2.21 W m⁻²); for +15–60 min it remains +0.13 to +1.92 W m⁻². The bootstrap uses 112 fixed calendar two-day blocks and 5,000 resamples.

![Trajectory H-CV GHI results](results/direct_cot_trajectory_hcv_v1_local4060/score/trajectory_hcv_summary.png)

## Interpretation and decision

The tested four-map COT history did not provide reliable incremental GHI information; its point estimate was worse than both the target-frame and no-COT arms, with a larger positive GHI bias as well. Since the paired uncertainty intervals still cross zero, this screening does not establish a general causal harm. It does reject the predeclared claim that this trajectory representation is a promising H-CV candidate under the current head and training contract.

This result is consistent with the earlier finding that slightly better COT scores against CPP do not ensure GHI improvement. Possible explanations include correlated errors across forecast COT maps, residual COT timing/amplitude error, or the small 3×3 readout being unable to use trajectory structure; these are hypotheses, not diagnosed causes. The plan's stop rule applies: do not tune leads, dates, normalizers, or thresholds on these folds, and do not reuse the already examined outer V as untouched confirmation. A later confirmation requires genuinely unused ground-GHI dates with verified label provenance; the accessible Hunan inventory currently does not verify such an extension past 2026-07-01.

## Reproduction and provenance

- Run receipt: `results/direct_cot_trajectory_hcv_v1_local4060/run_manifest.json` (`COMPLETE_COT_TRAJECTORY_H_CV`; 90 heads, 30 epochs, batch 512).
- Paired score: `results/direct_cot_trajectory_hcv_v1_local4060/score/trajectory_hcv_metrics.json` (`COMPLETE_COT_TRAJECTORY_H_CV_SCORE`).
- Absolute/leadwise summary: `results/direct_cot_trajectory_hcv_v1_local4060/score/trajectory_hcv_summary.json`.
- Plot source: `summarize_cot_trajectory_hcv.py`; scorer: `score_direct_cot_trajectory_hcv.py`; scorer regression: `test_score_direct_cot_trajectory_hcv.py`.
- H-only rows: Sili 72,345; Zhujia 79,417. The 90 heads were trained on one RTX 4060; each fold fits normalization on training dates only and removes targets within ±1 BJT day of the held-out two-day blocks.
- Test/buffer payload, future observed AGRI/CPP/GHI model input remain false. Manuscript body was not edited.
