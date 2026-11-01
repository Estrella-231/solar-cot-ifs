# Attempt12 train-only future-COT reliability calibration

## Question

Can a fixed, train-derived correction of the future COT amplitude reduce validation COT error versus (i) the uncorrected SimVP-to-R forecast and (ii) holding the initialization-time R-COT field unchanged?

This is a post-hoc diagnostic, not a new physical forecast model. It tests calibration only; it does not introduce cloud-water conservation, transport equations, or radiative closure.

## Frozen comparison

- Fit exactly one two-parameter affine map in `log1p(COT)` using only the five date-held-out training OOF banks and their correctly aligned CPP targets from `prepared_attempt12/R_packs/full/{rows.csv,target.npy,mask.npy}`.
- Use one global map for all leads and all pixels: `z_cal = a*z_forecast + b`, with `a` clipped to `[0,1]`, and physical output clipped to `[0,100]`. Fit by masked squared error with each BJT target date contributing equal total weight.
- Report fold-excluded OOF calibration diagnostics to show whether the fixed form generalizes across held-out training-date folds. Do not pick or alter the transform from validation.
- Apply the all-train OOF fit once to the full-train validation COT bank. Compare raw forecast, calibrated forecast, and history persistence on the pre-existing exact 1,164 common `(init,target,lead)` keys and the same CPP-valid pixel mask. CPP is a satellite retrieval reference, not independent truth.
- Report full 16x16 patch, three station pixels, and three station 3x3 neighborhoods; pooled and lead-wise physical RMSE/MAE/bias; paired BJT target-date bootstrap intervals for RMSE differences; and a predeclared near-zero false-alarm diagnostic (`CPP <= 0.5`, forecast `COT > 1`).

## Data and safety boundary

Only attempt12 train OOF COT banks, train CPP target/mask arrays, and the existing validation COT/reference/mask/persistence arrays are used. No GHI labels, future observed AGRI inputs, new training, test payloads, or buffer payloads are used. The numerical inputs are copied once from the existing server assets to a temporary local cache for a CPU-only analysis; the saved receipt binds the source SHA-256 values. The temporary cache is removed after scoring.

## Interpretation gate

Calibration is useful only if it improves over raw forecast on common validation support and does not claim superiority over persistence unless the paired date-block interval supports it. Even a positive result is a validation-only reliability correction against CPP retrieval. If the global correction does not help, stop this correction and do not tune it against the same validation dates; prioritize a separately justified space-time evolution method instead.
