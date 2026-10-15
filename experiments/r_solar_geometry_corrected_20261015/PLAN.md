# Corrected solar geometry R comparison — 2026-10-15

## Question

Does retraining the frozen monthly Hunan AGRI-to-COT model with the corrected
equation-of-time conversion improve CPP-supported validation retrieval versus
the existing R checkpoint, and does the effect persist when training data,
mask, initialization, and budget are paired?

## Fixed contract

- Split: `hunan_monthly_block_2d_buffer1d_seed42_v1`; train and validation only.
- Original checkpoint: `monthly_R_station16_COTUNet_seed42_v1`, SHA-256
  `342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001`.
- Same source pack as the original checkpoint: 15,414 supervised train and
  1,398 validation patches. The 13 first-pass train source-read failures stay
  unavailable; this experiment does not silently substitute the recovered
  pack or change the sample rows.
- Model: same COTUNet topology, 16×16 station patches, 13 AGRI channels plus
  `cosSOZ`, `cosRAA`, `day_mask`; fresh seed 42; AdamW `1e-3`, weight decay
  `1e-4`; finite-safe masked Huber loss on `log1p(COT)`; batch 1,284; up to 50
  epochs with the original eight-epoch early-stop rule.
- CPP COT in `[0,100]` with `CLP∈{0,1,2}` remains the retrieval reference.
  It is not independent truth. Test payloads remain closed.

## Paired design

1. Preserve the existing old R checkpoint and evaluate it on the paired common
   validation pixels using its original normalization and old geometry.
2. Recompute nominal-UTC solar zenith/azimuth from the corrected formula,
   `equation_of_time = 4 * rad2deg(trigonometric_expression)`. Recompute the
   three geometry channels from the fixed Hunan `grid_static` (`lat`, `lon`,
   `saa`). Do not change AGRI, CPP labels, split, station crops, or source-row
   availability.
3. For the controlled retraining pair, use only pixels eligible under both
   the original pack mask and corrected `day_mask>0.5, cosSOZ>cos80`. This
   keeps train and validation labels/support identical between arms. Refit
   each arm's normalization on its own input values over the same common
   train pixels.
4. Train one fresh seed-42 old-formula control and one fresh seed-42 corrected
   geometry model, with identical initialization hash, sample order, batch,
   optimizer, stopping rule, and validation selection procedure. The extra
   old-formula retrain separates the geometry change from seed/training
   variation; it is not a replacement for the archived original checkpoint.
5. Independently score saved predictions in FP64 on the common validation
   support. Primary summary: equal-station mean of the two station COT RMSEs,
   with paired BJT-date cluster bootstrap. Also report near-zero, thin,
   moderate, and thick reference strata, station breakdown, bias, MAE, support
   counts, checkpoint reload, source hashes, and actual training completion.

## Interpretation boundary

This is a validation comparison against CPP retrieval output on common
supported daytime pixels. A positive result shows better agreement with CPP
under this split; it does not establish independent COT truth or an improved
GHI forecast. No historical COT bank is regenerated in this step. Downstream
GHI impact requires a later separately authorized paired experiment using the
new frozen R and the same GHI contract.

## Execution status

Initial and final execution evidence are maintained in `STATUS.md`; final
metrics, cloud-category analysis, and limitations are recorded in
`RESULT_REPORT.md`. The paired run completed successfully, with no decisive
overall gain over the matched old-formula retrain. No manuscript files were
modified.
