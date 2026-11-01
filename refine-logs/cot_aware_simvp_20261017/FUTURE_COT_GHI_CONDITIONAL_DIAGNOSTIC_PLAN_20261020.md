# Validation diagnostic: when does forecast-COT error translate into GHI error?

Date: 2026-10-20. This is a descriptive validation diagnostic after the affine and temporal-increment calibration screens. It does not tune a model or choose a validation stratum for deployment.

## Question

The three-seed results show a clear station/lead pattern: forecast-COT helps Sili, most at long leads, but not Zhujia; temporal-increment calibration improves train-only frozen-R COT RMSE slightly and changes GHI negligibly. Test whether per-instance center COT retrieval error is associated with the paired GHI benefit of predicted future COT, and whether that relationship differs by station, lead band, and the predeclared forecast-COT regimes `<1`, `1–10`, `10–30`, and `≥30`.

## Data and analysis

- Use only the existing validation future-GHI pack and saved A/AH/AC/ACREL/ACRELTEMP predictions, exact `(station, init_time, target_time, lead, ghi_row_index)` matches, and the validation AGRI→frozen-R COT reference at the exact target timestamp.
- Treat the reference as a retrieval comparator, not physical truth. It is used only to diagnose COT forecast error; it is never an inference input. CPP, test, and buffer are not read.
- Report center log1p-COT error and paired GHI RMSE change for ACREL−AH and ACREL−A, by station, predefined lead bands, and predefined predicted-COT regimes. Include target-date-block bootstrap intervals. Overlapping forecast instances are not independent; uncertainty is clustered by target BJT date.
- Use forecast-COT regime thresholds fixed in `FUTURE_GHI_GATE.md`; do not choose thresholds or model coefficients from validation results. This analysis cannot itself authorize a gain claim.

## Decision use

If lower COT reference error consistently coincides with stronger paired GHI improvement, the next candidate may fit a sample-adaptive reliability correction using only training OOF forecast features and training frozen-R references, cross-fitted by date fold, then pass a COT pre-gate before any multi-GPU GHI retraining. If the association is absent, prioritize the GHI input/target temporal-support mismatch or head representation rather than making the COT field cosmetically smoother.
