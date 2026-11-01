# Attempt12 forecast-COT failure diagnosis (validation only)

**Question.** Why did the geometry-corrected SimVP→R forecast-COT chain produce almost no overall GHI gain, despite the frozen GHI head responding to COT?

## Evidence

The figure uses one validation initialization, 2025-09-27 13:30 BJT (sample 77), with the four +60/+120/+180/+240 min target images. The event was selected because its station-local COT trajectories show a large mismatch; it is illustrative and is not a population estimate.

- The first two rows compare observed and SimVP-forecast AGRI C02 visible reflectance on one fixed 0–0.8 scale. The lower two map rows compare corrected frozen-R COT from the future observed AGRI (offline reference) with COT retrieved from SimVP forecast AGRI.
- At Niushou, forecast-COT vs offline future-AGRI→R COT is 17.48 vs 39.79 at +120 min, then 47.90 vs 24.51 at +240 min. This is a timing/evolution mismatch in this event, not a constant bias.
- Reusing the same frozen FAC GHI checkpoint and changing only the COT input gives event RMSEs: Niushou 115.1 vs 125.6, Wangzhai 107.0 vs 112.5, Mizhuang 72.5 vs 78.4 W m⁻² (forecast COT vs offline future-AGRI→R input). The offline path is not available at issue time and is not a deployable score.
- Across all validation rows in attempt12, equal-station/equal-seed GHI RMSE is 95.703 without forecast COT and 95.609 W m⁻² with it, a 0.098% point reduction; date-block confidence intervals cross zero. Across the full validation CPP-common support, forecast COT RMSE is worse than history persistence at 1, 2, 3 and 4 h (10.255/9.787/10.194/12.063 vs 8.946/9.238/9.617/9.086). CPP is a retrieval reference, not independent COT truth.

## Interpretation and boundary

This selected event is consistent with a forecast-COT trajectory error limiting the GHI benefit: the frozen head reacts to the input, while predicted COT can put a cloud-thickness change at the wrong lead. It does not establish that COT forecast error is the only cause, nor that the shown event is typical. The global paired GHI result remains the relevant outcome and is inconclusive. The offline future-AGRI→R curve is an attribution reference only; it is neither truth nor a usable forecast input. No test or buffer payload was read.

## Next controlled training candidate

Use the already verified attempt12 corrected train/validation package and frozen corrected R. Compare a matched AGRI-MSE retrain against the same SimVP with one added CPP-supported COT auxiliary loss computed through frozen R on forecast AGRI. Hold architecture, initialization, sample order, optimizer, epochs, validation keys and all other inputs fixed. Determine the loss weight from training-only gradient ratios, not validation. First score forecast COT against persistence and the existing SimVP→R baseline on common CPP support, with AGRI degradation and thick-cloud/near-zero safeguards. Only if that gate passes should fold-specific OOF retraining and another downstream GHI-head pairing begin. Test and buffer stay closed.

The corrected-geometry attempt12 retraining itself is already complete; this candidate is a separate attempt and must not overwrite it.
