# Unified input feature signed residual experiment

## Question

Does the signed AGRI+COT residual learn a useful correction when train and
validation use one residual-owned encoder over the same normalized AGRI and
solar-geometry inputs, while the cross-fitted A models supply only a physical
GHI scalar?

## Fixed comparison

- Use the existing monthly train/validation pack, five contiguous blocked OOF
  A predictions, station/date keys, seeds 42/43/44, and full-A checkpoints.
- Compare exactly two capacity-matched arms: A + signed AGRI residual and
  A + signed AGRI + COT residual. Do not consume cached fold encoder features.
- Both residual models encode normalized 13-channel real AGRI plus two solar
  geometry channels through their own shared 3x3 spatial encoder. The AGRI
  control auxiliary branch receives the encoded local AGRI map; the COT arm
  receives normalized COT in its original 3x3 arrangement. Both auxiliary
  branches have the same 32-to-4 projection and readout dimensions.
- Train rows receive only their held-out fold A GHI prediction as `base`; the
  validation rows receive the existing full-training A GHI prediction. The
  residual is unconstrained in sign and added to this scalar base. The exact
  zero residual is a checkpoint-selection candidate.
- Hold optimizer, initialization seeds, row order, 30 epochs, batch size 2000,
  optimizer and learning-rate schedule equal to OOF v2. Validation selects
  checkpoints; no test payload is opened.

## Evaluation

Recompute per-seed and equal-station RMSE/MAE/bias on common validation rows;
paired BJT date-block bootstrap for AGRI-A, AGRI+COT-A, and AGRI+COT-AGRI.
Report signed adjustment fraction and right-sign fraction when A underpredicts
or overpredicts, plus the pre-registered bright/thick diagnostic
(`center COT >= 3`, clear-sky GHI >= 100 W/m2, observed kt >= 0.9). Ground truth
defines these post-hoc evaluation masks only; it never routes a model at
inference. Report best epochs, step-zero selections, and equal parameter
counts. This is same-time real-AGRI-to-GHI retrieval, not forecast-AGRI skill.

## Interpretation gate

If both arms improve consistently, the previous fold-hidden-feature mismatch
was a material implementation concern; this does not alone establish it as
the sole cause. If AGRI improves and COT does not, investigate cloud-height,
parallax, and solar-direction shadow alignment. If neither improves, compare
OOF and full-A scalar residual distributions by date/fold before further model
changes. Do not open test or claim operational forecast gain from this run.

## Status

PBS `214017.tc6000` completed on one NVIDIA A100-SXM4-40GB (node23 GPU 2),
exit status 0. The queue was empty at submission; pack and OOF provenance
gates passed. The 12 runs completed without error; equal parameter counts are
36,773 in each arm; no OOF hidden features or test payload were used.

Independent validation recomputation is in
`results/signed_residual_oof_v3_unified_analysis_20261014_v1/` and the detailed
interpretation is in `UNIFIED_FEATURE_RESIDUAL_RESULTS_20261014.md`. AGRI
residual improves over A; adding COT has a small negative point difference
against AGRI whose date-block interval crosses zero. This remains same-time
real-AGRI-to-GHI validation, not forecast-AGRI skill.
