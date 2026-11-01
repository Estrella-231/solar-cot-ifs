# Cross-fitted temporal-increment calibration for future COT

Date: 2026-10-20. This is the next single-factor candidate after (i) spatial-neighbor contrast scaling failed its COT pre-gate and (ii) removing future-COT neighbor texture made essentially no GHI difference.

## Question and transform

The oracle-COT substitution has a much larger GHI effect at long leads than early leads, while the future-COT center is already reasonably correlated with the same-time frozen-R reference. Test whether the predicted *change* from the preceding 15-minute COT frame is mis-scaled even when the level is calibrated.

For each station and target lead, calibrate only the temporal innovation in log1p-COT:

```text
COT*_t = COT_ACREL_(t-15) + alpha[station, lead] *
         (COT_ACREL_t - COT_ACREL_(t-15))
```

Fit alpha by through-origin least squares of predicted versus observed training-date 15-minute COT increments. Training features use OOF forecast AGRI→frozen-R COT and date-fold-excluded alpha; validation uses coefficients fit on all five training folds. Both increments are paired to the same station and exact target times in the actual-R train reference. Clip alpha to [0,2] as predeclared stability bounds. The GHI input changes only the target future COT frame; its preceding context frame, earlier COT history, AGRI, geometry, label, mask, sample keys, GHI-head architecture, normalization convention, seed order, optimizer, batch and 30-epoch budget stay unchanged.

The CPP product is not used. Frozen-R COT from observed future AGRI is used only as a training label/reference for alpha and fold-heldout COT error, never as an inference input. No validation target is used to fit alpha.

## COT pre-gate and GHI screen

First calculate fold-excluded COT level and increment errors on the training OOF rows, paired by target BJT date. The pre-gate passes only if the equal-station/equal-lead center-pixel log1p-COT RMSE difference (temporal candidate minus ACREL) has a date-block 95% interval wholly below zero and neither station's pooled center RMSE worsens. Report full 3×3 patch RMSE as a secondary check. If it fails, do not train this candidate's GHI heads.

If the pre-gate passes, train seeds 42/43/44 concurrently, one per A800 in one PBS allocation. Compare exact-key paired saved GHI predictions against A, AH, AC, and ACREL using the same 52 target-BJT-date bootstrap. Report both stations, every lead, and +15–60, +75–120, and +135–240 minute bands. The primary comparisons remain candidate−AH and candidate−A; also report candidate−ACREL. This validation screen does not open test/buffer or replace the independent-test gate.
