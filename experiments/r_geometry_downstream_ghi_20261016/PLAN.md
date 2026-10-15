# Downstream GHI effect of the corrected solar geometry in R

## Question and ordering

This experiment asks whether the equation-of-time correction in the AGRI→COT
retriever changes station GHI skill. It follows the paired R validation audit:

1. Show the two retrained R models on the same real AGRI validation scenes,
   alongside CPP retrieval reference, using identical cases and color scales.
2. Build both frozen R-COT inputs for the exact three observed frames used by
   every row in the existing `cot_interval_history_20261015` GHI pack.
3. Train three paired GHI arms: `A_history` (no COT),
   `AC_history_old_R` (freshly retrained R with original geometry formula), and
   `AC_history_corrected_R` (same training contract with corrected geometry).

The R pair is the matched fresh seed-42 pair in
`r_solar_geometry_corrected_20261015`, not the archived R checkpoint. This
isolates the geometry-formula change from retraining history.

## Fixed contract

- Region/stations: Hunan, Sili and Zhujia, each trained independently.
- Split: `hunan_monthly_block_2d_buffer1d_seed42_v1`; use only train/validation.
- GHI rows and labels: exact row order, target keys, labels and daylight support
  from `results/cot_interval_history_20261015/pack_manifest.json` (47,666 train,
  4,263 validation; all three-frame histories shared across arms).
- Inputs: same real AGRI sequence at offsets `[-30,-15,0]` min, same corrected
  nominal-UTC geometry, same local 3×3 readout, same 15-minute interval GHI
  target. COT is the paired frozen-R prediction, cropped to those same pixels.
- `A_history` has a zero standardized COT slot. Both COT arms share a train-only
  COT scaler fit on the pooled old/corrected-R training predictions; AGRI and
  geometry normalization are identical across all arms.
- Head, initialization, station/seed, row order, optimizer, batch 1,792, 30
  epochs and checkpoint selection rule match the prior three-frame GHI run.
  Seeds are 42/43/44 for both stations and all three arms (18 fits total).
- Frozen R models receive no gradients. No GHI observation, CPP value, future
  AGRI, or test payload enters model input/training selection.

## Interpretation boundary

The GHI arms use observed real AGRI at T−30/T−15/T, including a T frame whose
scan occurs after its nominal timestamp. This is the same offline synchronous
15-minute interval-retrieval study as the existing GHI contract, not a causal
forecast or SimVP→R deployment result. Validation selects the GHI checkpoint;
the validation result is not an independent test estimate. CPP is a retrieval
reference, not independent COT truth.

The primary contrasts are `AC_history_old_R − A_history`,
`AC_history_corrected_R − A_history`, and the paired geometry contrast
`AC_history_corrected_R − AC_history_old_R`. Negative ΔRMSE favors the first
named candidate. Use station-equal mean RMSE and paired target-date bootstrap;
also report stations, seeds, MAE, MBE, lead-independent scope, and validation
sample counts. No experiment is expanded based on a favorable seed.

## Safety and outputs

- No test payload is read. Train/validation keys are asserted before loading
  AGRI or sidecar arrays.
- Preserve old experiment directories. All new scripts, inputs, output hashes,
  plots, logs and results go in this directory on the local repo and its matching
  server experiment folder.
- Report every unavailable input and keep the same common sample set; do not
  impute missing COT as clear sky.
- Save validation predictions per row and seed, reload checkpoints on CPU, and
  independently recompute FP64 metrics before reporting.
