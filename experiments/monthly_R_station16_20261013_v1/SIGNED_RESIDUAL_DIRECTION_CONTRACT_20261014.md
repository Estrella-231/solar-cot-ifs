# Signed GHI residual direction experiment

## Question

Does co-registered 3×3 COT help a model choose the sign and size of a local
GHI correction beyond a capacity-matched AGRI-only correction branch?

## Paired design

- Freeze each existing station/seed A checkpoint (seeds 42, 43, 44).
- Keep the monthly train/validation split, sample keys, labels, AGRI and solar
  geometry inputs, optimizer, epochs, batch size, loss, and validation
  checkpoint rule fixed.
- Compare `AGRI` and `AGRI_COT` signed residual heads. Both receive the same
  frozen A forecast, clear-sky GHI, and AGRI+geometry features. Their trainable parameter
  counts are identical. The auxiliary branch is 32→4 in both arms; AGRI-only
  uses the 32-channel AGRI representation, while the COT arm expands the
  normalized COT field over 32 channels before the same encoder.
- Initialize the residual readout to zero. Before training, both arms must
  reproduce their own frozen A forecast exactly. The learned residual is
  signed in W m⁻²; final GHI is clipped only at zero.
- Minimize MSE of final physical GHI divided by 1000, use AdamW (1e-3,
  weight decay 1e-4), cosine schedule, 30 epochs, batch size 2000, and select
  the checkpoint by full validation RMSE. Both arms reuse the same epoch
  permutations for each station/seed.

## Data and interpretation limits

Only the completed `hunan_monthly_block_2d_buffer1d_seed42_v1` train/validation
pack is read. The test payload remains closed. This is synchronous real-AGRI
to GHI retrieval, not SimVP forecast-AGRI forecasting.

Training residuals are computed against A predictions on the same training
rows used to fit A. This is a controlled diagnostic of a frozen-A correction,
not an out-of-fold production correction. Validation outcomes are the
generalization check. A positive result supports the value of COT for this
correction task only; it does not establish forecast-chain gains or a
physical cloud-shadow mechanism.

## Required checks

1. Reloaded A predictions match saved A validation predictions within 1.0
   W m⁻² for every station and seed. Native CUDA convolution is used to avoid
   the cluster cuDNN loader conflict; the observed first-64-row difference
   for Sili/seed42 was 0.0442 W m⁻² and the complete 2,351-row Zhujia/seed42
   difference was 0.4517 W m⁻². Exact step-zero identity is checked against
   the reloaded A in the same runtime. The relaxed tolerance is documented
   because the two convolution backends differ slightly in FP32 arithmetic.
2. The zero-initialized residual is exactly zero and matches A at step zero.
3. AGRI and AGRI_COT trainable parameter counts match.
4. Save validation predictions, signed residuals, validation keys, source
   hashes, per-epoch metrics, and test-closed provenance.
5. Recompute paired metrics independently from saved predictions and report
   station/seed results plus date-block uncertainty; do not pool samples as
   independent days.
