# AC COT counterfactual forward protocol (2026-09-12)

## Question

For the frozen AC station head, which of the four explicit predicted-COT inputs
(`mean`, spatial population `std`, `p90`, or center pixel of `log1p(COT)`)
drives the adverse correction at Sili?

## Frozen scope

- Hunan validation rows only: the existing 99,849 canonical `+15...+240 min`
  rows in `head_pack_trainval_20260908_v1`; test is not read.
- Original-cohort AC checkpoints at the fixed learning rate `3e-4`, seeds
  42/43/44. Each checkpoint is loaded without changing any weight.
- For every checkpoint and intervention, `x` (forecast AGRI), geometry,
  station, lead, clear-sky GHI, observation, row order, and batch order are
  identical. Only the four-dimensional normalized COT vector is changed.
- The original-COT pass must reproduce the checkpoint's saved validation
  predictions within a fixed numeric tolerance before counterfactual results
  are accepted.

## Interventions

The packed COT feature order is
`[mean_log1p_cot, std_log1p_cot, p90_log1p_cot, center_log1p_cot]`.

1. `original`: unchanged normalized COT vector.
2. `zero_all`: all four normalized values set to zero. Since the pack used
   train-only z-score normalization, this is the physical training-mean vector,
   not physical zero COT.
3. `sunny_median`: coordinate-wise median normalized COT among original-training
   rows with a valid CLP-derived sunny label, calculated separately for each
   station and lead. It is a diagnostic reference and does not use validation
   labels or statistics.
4. `drop_mean`, `drop_std`, `drop_p90`, `drop_center`: set only the named
   normalized coordinate to zero and leave the other three at their row values.

The CLP-derived sunny label is used only to construct the train-only reference
vector. It is an instantaneous patch cloud-coverage proxy, not ground daily
weather and not a model input.

## Outputs and interpretation

Save every per-row counterfactual prediction and recompute GHI metrics from
`pred_kt * interval_clear_sky_GHI`. Report each seed, station, predicted-weather
group, lead, and the across-seed mean/sample SD. For each intervention report
the paired GHI shift and RMSE/MAE change relative to `original`.

For a single-feature drop, a positive mean prediction shift means removing that
feature raises GHI, so the feature participated in a downward AC correction in
that context. A negative RMSE delta means the removal improves that subset.
Because the MLP is nonlinear, the four one-at-a-time effects can interact and
need not sum to the all-zero effect; they identify model response to controlled
input interventions, not a unique physical cause.

