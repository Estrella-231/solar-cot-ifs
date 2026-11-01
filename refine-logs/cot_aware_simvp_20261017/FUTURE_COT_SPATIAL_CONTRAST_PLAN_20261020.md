# Future-COT spatial-contrast calibration screen

Date: 2026-10-20. Status: protocol drafted; no training or validation result is implied by this plan.

## Question

The three-seed frozen-head oracle substitution improved validation GHI RMSE, especially at longer leads, but predicted-COT spatial anomalies have weak alignment with the oracle patch. The existing station/lead affine calibration acts on every 3×3 pixel with the same slope, so it changes the center amplitude and neighbor contrast together. This screen tests whether the neighbor-pixel deviations add useful spatial information or mostly add error after the center COT has been calibrated.

## Single change

Use the existing train-only, fold-excluded center-pixel reliability calibration unchanged. For future frames only, leave the calibrated center pixel fixed and multiply the eight neighbor deviations in log1p-COT space by one station/lead coefficient `gamma`:

```text
COT_spatial(r,c) = COT_center_calibrated
                    + gamma[station, frame_lead] *
                      (COT_center_calibrated_map(r,c) - COT_center_calibrated)
```

Fit `gamma` by least squares through the origin on the eight centered 3×3 neighbor anomalies, using only date-heldout five-fold OOF training forecasts paired with frozen-R COT derived from observed training AGRI. For each training fold, fit gamma on the other four folds; fit validation gamma on all five training folds. Clip gamma to `[0,2]` only to prevent an unstable extreme amplification. The center value is unchanged exactly; historical COT, AGRI, geometry, GHI labels, masks, sample keys, head architecture, initialization, optimizer, batch size, epochs and selection rule remain fixed. Apply the existing train-only COT normalization after the transform.

This is a calibration hypothesis, not a claim that spatial contrast is noise. The CPP/AGRI retrieval reference is not independent truth. The validation COT comparison is descriptive and will not be used to refit gamma.

## Evaluation

- Training-reference COT audit: same-key center and 3×3 log1p-COT RMSE for raw forecast COT, existing center-only ACREL, and ACREL with spatial-contrast calibration, with fold-excluded training estimates.
- The CPU pre-gate passes only if the equal-station/equal-lead cross-fitted 3×3 patch-RMSE difference (ACRELSP−ACREL) has a date-block 95% interval wholly below zero and neither station has a worse pooled patch RMSE. This decision rule is fixed before reading the audit output.
- GHI head: train only the new spatial-contrast arm for seeds 42/43/44 with the existing fixed IntervalHead budget; compare saved predictions on exact common validation keys against existing A, AH, AC, and ACREL outputs.
- Primary test: spatially calibrated forecast-COT arm minus AH; also require comparison against A, and report the increment over ACREL. Use the existing 52 target-BJT-date paired bootstrap and inspect both stations, +15–60 min, all leads, and +120–240 min.
- Screening criterion remains the established gate: negative paired RMSE deltas, date-block interval below zero for both total-COT and future-COT contrasts, both stations without unacceptable degradation, and no concealed early-lead harm. No validation-tuned gamma, threshold, COT stratum, or sample mask.
- Test/buffer payloads and future true AGRI/CPP inference inputs remain closed. Future observed AGRI is used only to construct the train/validation frozen-R diagnostic COT reference, never as a GHI inference feature.

## Decision

If fold-excluded COT spatial error does not pass the predeclared date-block gate, stop this branch without a GHI run. If it passes, run the three GHI seeds concurrently on three separate GPUs in one PBS allocation. A GHI gain remains unproven until the paired saved-prediction audit passes; this validation screen cannot replace the independent-test gate.
