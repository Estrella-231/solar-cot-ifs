# Same-support diagnosis: direct COT misses thick-cloud amplitude

Date: 2026-10-17. Validation only; test payloads remain closed.

## Comparison contract

The saved seed-42 direct COT forecasts were compared with the original frozen SimVP forecast-AGRI→R bank on the same 3,140 validation sequences, both stations, all 16 leads (+15…+240 min), the same compact CPP reference, and the same `CPP-common × new-R-frame-valid` pixel mask. Sequence IDs and every target timestamp were checked exactly before scoring. The original SimVP bank's older CPP sidecar was not used; its forecast COT array alone was paired to the current compact CPP keys and mask. The COT transformation was inverted once, and RMSE/bias were accumulated in FP64.

CPP is a retrieval reference, not independent physical truth. The direct forecaster's best epoch was selected on this validation cohort, so these are screening point estimates rather than independent inference.

## What the matched evaluation shows

| CPP reference stratum | Original SimVP→R, 16×16 | P-Hybrid, 16×16 | Difference | Original SimVP→R, local 3×3 | P-Hybrid, local 3×3 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| All | 10.953 | 12.407 | +1.454 | 10.664 | 12.162 | +1.498 |
| COT < 1 | 10.955 | 8.988 | −1.967 | 10.355 | 8.540 | −1.815 |
| 1–5 | 5.223 | 5.140 | −0.084 | 6.481 | 5.469 | −1.011 |
| 5–30 | 8.721 | 9.594 | +0.873 | 8.670 | 9.724 | +1.053 |
| ≥30 | 18.263 | 25.668 | +7.405 | 17.911 | 25.511 | +7.600 |

Units are COT RMSE. Negative difference favors P-Hybrid. Its thick-cloud mean bias is also more negative: −21.636 versus −13.962 on the full patch, and −21.388 versus −13.735 locally. Both stations show the same direction. Thus, the current direct model is not failing uniformly: it improves the near-zero stratum, but loses substantially on moderate and especially thick reference COT. The thicker-cloud amplitude underestimation is a concrete candidate mechanism for its worse overall score.

## Fixed physical-amplitude loss follow-up

The single-factor RC12 screen added `0.1 × MSE((COT_prediction − COT_target)/100)` to the original equal-lead masked log-space loss. All other training settings followed the selected P-Hybrid seed-42 screen. The selected epoch was 11. Independent FP64 rescoring on the same validation support gives:

| Candidate | Full 16×16 RMSE | Local 3×3 RMSE |
|---|---:|---:|
| Original SimVP→R | 10.953 | 10.664 |
| P-Hybrid base | 12.407 | 12.162 |
| P-Hybrid + physical-amplitude term | 12.473 | 12.209 |

Against P-Hybrid base, the augmented loss worsens the overall scores by 0.066 (full patch) and 0.046 (local 3×3). It improves the COT≥30 stratum by only 0.133 and 0.358, respectively, while worsening COT<1 by 0.160/0.197 and COT 5–30 by 0.182/0.213. It remains well above the original SimVP→R point RMSE. RC12 therefore fails its predeclared all-stratum advancement criterion and closes this direct-COT loss-tuning branch. No extra weight sweep, seed expansion, GHI head, or test access follows from this screen.

This result does not identify the residual error as amplitude alone. Cloud displacement/arrival, cloud evolution, retrieval disagreement, and optimization remain unresolved explanations. The next diagnostic is event-level separation of position/timing from thickness evolution on the frozen common-support cases; no new training should be selected from this result alone.

## Reproducibility

- Scorer: `../analyze_fulltrain_vs_simvp_same_support_20261017.py`
- Machine-readable results: `simvp_vs_direct_cot_strata_20261017.json`
- Direct-model source predictions and checkpoint selection: `../FULLTRAIN_RESULTS_20261017.md`
- Original SimVP forecast bank source: `../../../refine-logs/cot_value_first_20261013/results/forecast_agri3x3_allval_20261014_v1/profile.json`; only its `forecast_log1p_cot` and verified sequence/time keys were used. Its historical CPP join is invalid and was deliberately excluded.
- `test_payload_opened=false`.
- RC12 run record and stop decision: `../AMPLITUDE_LOSS_RUN_20261017.md`.
