# Cross-fitted A residual experiment

## Question

Does the signed AGRI+COT residual advantage remain when the residual targets
are built from out-of-fold (OOF) A predictions rather than A predictions on
the same rows used to fit A?

## Frozen design

- Use only the existing monthly train/validation pack. Validation labels select
  the residual checkpoint and support diagnostics; the test payload remains
  closed.
- For each station and seed (42, 43, 44), divide the station's sorted BJT
  training dates into five contiguous, non-overlapping folds. To reduce
  adjacent-day leakage, exclude the nearest available training date on either
  side of the held-out block from that fold's A-training set.
- Fit one A-only LocalGHIHead per fold, using only the remaining training rows,
  the pack's frozen train-only normalization, AdamW (1e-3, weight decay 1e-4),
  cosine schedule, batch 2000, and 30 fixed epochs. Do not select fold models
  using validation. Save each held-out row's physical A prediction and frozen
  A encoder features. Each train row receives exactly one OOF prediction.
- Train the same signed AGRI and AGRI+COT residual heads as in the prior
  experiment, with equal parameter counts, zero initialization, same rows,
  seeds, permutations, optimizer and epochs. On residual-training rows, feed
  the fold-specific OOF A prediction and OOF encoder features and learn the
  signed correction to that OOF forecast. On validation rows, use the original
  full-training A checkpoint/features, and select by full validation RMSE.
- Recompute step-zero identity against the full A checkpoint on validation.
  Verify OOF row coverage is exactly one per train row, no val/test row is in
  OOF targets, fold date boundaries/buffers, finite values, and equal residual
  parameter counts.

## Interpretation

This is a sensitivity test for in-sample residual-target optimism. Because the
OOF A models train on fewer dates than the full A checkpoint, OOF predictions
may be less accurate and have a different error distribution. The comparison
is therefore not an estimate of a deployed cross-validated ensemble. Report
OOF A error and residual performance separately, and do not compare it as a
forecast-chain result. A positive result supports robustness of the signed
correction to cross-fitted targets; a negative result requires inspecting the
OOF/full-A gap before concluding that COT itself is unhelpful.

## Data and compute provenance

Pack: `hunan_monthly_block_2d_buffer1d_seed42_v1`; station GHI is a target only.
No real future AGRI/CPP or test payload enters model inputs. PBS submission is
limited to one GPU job at a time after live account queue inspection.
