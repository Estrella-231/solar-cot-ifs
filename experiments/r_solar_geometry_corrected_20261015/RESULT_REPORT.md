# Corrected solar geometry in the COT retrieval model (R)

**Run date:** 2026-10-15

**Split:** `hunan_monthly_block_2d_buffer1d_seed42_v1`

**PBS:** `214255.tc6000` (exit status 0; 652 s; node22; one A800-SXM4-80GB)

## Question and result

We tested whether correcting the equation-of-time unit conversion in the R
model's solar-geometry channels improves retrieval agreement with CPP COT. The
controlled comparison did **not** show a statistically clear overall gain on
the common validation support. The correction is physically justified, but
this run is insufficient to conclude that corrected geometry improves R.

The primary score is the equal-station mean of Sili and Zhujia station RMSEs.
Positive delta means corrected geometry has higher error:

| Comparison | Sili ΔRMSE | Zhujia ΔRMSE | Equal-station ΔRMSE | 95% date-cluster CI |
|---|---:|---:|---:|---:|
| Corrected retrain − old-formula retrain | +0.305 | −0.133 | **+0.086** | **[−0.282, +0.438]** |

There are 1,398 validation patches and 352,441 identical CPP-supported daytime
pixels across both arms, with 46 BJT dates per station. The paired bootstrap
resamples dates (2,000 replicates). Since the confidence interval spans zero
and the station effects have opposite signs, the data do not resolve an
overall benefit or harm.

## Validation metrics

All metrics below are evaluated on the same 352,441 pixels. COT units follow
the CPP reference product.

| Model | RMSE | MAE | Bias |
|---|---:|---:|---:|
| Archived original R checkpoint | 9.018 | 4.615 | +0.346 |
| Fresh seed-42 retrain, original geometry formula | **8.790** | 4.713 | −0.395 |
| Fresh seed-42 retrain, corrected geometry | 8.878 | **4.657** | −0.702 |

The strongest causal comparison is between the two fresh retrains. Relative to
that control, corrected geometry lowers pooled MAE by 0.056 but raises pooled
RMSE by 0.089. The average absolute error falls slightly while squared error
rises, so the changes are uneven and the larger residuals weigh more heavily;
the per-station and cloud-category results below show where the trade-off lies.
The archived checkpoint is context, not the controlled geometry comparison:
it has a different training history and normalization.

### By station

| Station | Old-formula RMSE | Corrected RMSE | Corrected − old |
|---|---:|---:|---:|
| Sili | 8.771 | 9.077 | +0.305 |
| Zhujia | 8.808 | 8.675 | −0.133 |

The correction helps at Zhujia and hurts at Sili in this single-seed run. This
opposite response is a reason not to replace the existing R globally based on
this result.

### By CPP reference COT category

| CPP reference category | Pixels | Old RMSE | Corrected RMSE | Old bias | Corrected bias |
|---|---:|---:|---:|---:|---:|
| Near-zero `[0, 0.1)` | 126,856 | 8.805 | 9.513 | +3.146 | +3.297 |
| Thin `[0.1, 5)` | 49,850 | 1.642 | 1.558 | +0.014 | −0.477 |
| Moderate `[5, 30)` | 125,359 | 5.756 | 5.426 | +0.718 | −0.112 |
| Thick `[30, 100]` | 50,376 | 16.131 | 15.745 | −12.484 | −12.465 |

These are descriptive strata; confidence intervals were not calculated per
stratum. Corrected geometry modestly lowers RMSE in the thin, moderate, and
thick bins, but increases near-zero RMSE. Thick-cloud underestimation remains
large: bias is about −12.5 in both retrains. Thus the geometry correction does
not fix the persistent thick-COT peak underestimation. Near-zero overestimation
also remains and is slightly larger in the corrected arm.

The corrected arm's validation Huber loss in `log1p(COT)` is lower (0.03481 vs
0.03539), while its physical-space RMSE is higher. These metrics optimize/weight
errors differently: the training objective is a robust loss after a logarithmic
transform, whereas COT RMSE squares errors in physical COT units and is more
sensitive to the remaining large errors.

## Controlled protocol and provenance

- Exact original R source pack: 15,414 train rows and 1,398 validation rows;
  the 13 unavailable first-pass train reads remained unavailable. No recovered
  pack was substituted.
- Both arms use the same COTUNet architecture, seed 42 and initialization state,
  row order, common training/validation pixel support, batch 1,284, AdamW
  (`lr=1e-3`, `weight_decay=1e-4`), masked Huber loss on `log1p(COT)`, and a
  maximum of 50 epochs with eight-epoch early stopping. Each arm's input
  normalization was fitted separately on its own train inputs over the same
  common valid pixels.
- Original geometry arm uses the producer formula. Corrected arm computes
  `equation_of_time = 4 * rad2deg(trigonometric_expression)` before deriving
  geometry channels from the fixed Hunan `grid_static` latitude/longitude/SAA.
- The common mask is the source CPP/input-valid mask intersected with the
  original R mask and corrected `day_mask > 0.5`, `cosSOZ > cos(80°)` support.
  It retains 3,886,399 train and 352,441 validation pixels. Relative to the
  original source masks, 35,425 train and 3,728 validation pixels were excluded.
- The archived original checkpoint was re-evaluated on the same common
  validation support using its original normalization. Its SHA-256 is
  `342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001`.
- Saved predictions were scored independently in FP64, checked for exact row,
  label, and mask alignment, and bootstrapped by BJT date. Best checkpoints were
  reloaded from disk before saved predictions were produced.
- No test payload was opened. The validation reference is CPP retrieval output,
  not independent COT truth.

## Interpretation and next decision

This experiment addresses the R retrieval model only. It does not show whether
the corrected R changes or improves downstream station GHI forecasts. The
geometry formula correction should be retained as an isolated, physically
corrected candidate, but the present metrics do not support declaring it a
better R. The most defensible next step is to carry both frozen R variants into
a separately named, same-sample paired GHI experiment only if the goal is to
test downstream value; rebuild the historical COT input bank from each R with
identical dates and masks, then hold the GHI head, training budget, and seed
fixed. Do not infer a GHI gain from this COT-to-CPP comparison.

## Files

- `PLAN.md`: predeclared experiment contract.
- `STATUS.md`: completion and execution receipt.
- `data/pack_audit.json`, `data/rows_supervised.csv`, and
  `data/geometry_support.csv`: source and support audit; train/validation only.
- `results/paired_v1/paired_score.json`: authoritative HPC FP64 score.
- `results/paired_v1/*_validation_predictions.npz`: saved validation outputs
  for independent rescoring (about 4.9 MB total).
- `results/paired_v1/{old,corrected}/artifact_hashes.json`: checkpoint and
  output hashes. The checkpoint weights remain on the HPC at
  `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/r_solar_geometry_corrected_20261015/results/paired_v1/`.

PBS log ended with `PAIRED_CORRECTED_SOLAR_GEOMETRY_R_COMPLETE`.
