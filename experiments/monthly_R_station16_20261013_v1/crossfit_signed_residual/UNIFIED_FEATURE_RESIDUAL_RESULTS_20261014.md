# Unified-input signed residual results

## Result

The unified-input residual learned nonzero corrections that improved validation
GHI RMSE over the frozen A baseline. The additional COT channel did not show a
reliable gain over the parameter-matched AGRI residual arm.

| Validation method | Equal-station RMSE (W/m²) | Change from A | Paired date-block 95% CI for change |
|---|---:|---:|---:|
| Frozen A | 123.979 | — | — |
| A + signed AGRI residual | 111.277 | −12.702 | [−17.635, −7.978] |
| A + signed AGRI+COT residual | 113.419 | −10.561 | [−16.624, −4.716] |
| AGRI+COT minus AGRI | — | +2.142 | [−0.688, +4.745] |

AGRI improved the point RMSE in all six station/seed pairs. The COT arm also
improved over A in all six pairs, but its point RMSE was higher than AGRI in
all six. The paired bootstrap interval for COT minus AGRI crosses zero, so the
data do not establish that COT makes the residual worse on average; they also
do not establish an incremental COT benefit. No run selected step-zero: all
12 arms selected a trained checkpoint. Both arms have 36,773 trainable
parameters, used the same 30 epochs and batch size 2000, and start from an
exact zero correction.

## Direction and thick-cloud cases

Direction masks are post-hoc diagnostics from validation truth and never route
inference. Across station/seed validation rows, AGRI's correction had the
right sign on 54.9% of A-underprediction rows and 77.9% of A-overprediction
rows. The COT arm reached 55.9% and 73.4%, respectively. Thus the learned
correction handles downward adjustment more reliably than upward recovery;
adding COT did not improve this directional diagnosis.

On the pre-registered bright/thick subset (center COT >= 3, clear-sky GHI >=
100 W/m², observed kt >= 0.9; 58 Sili and 36 Zhujia rows, repeated across
three seeds), average paired RMSE change over the six station/seed pairs was
−26.943 W/m² for AGRI and −15.344 W/m² for AGRI+COT. The right-sign fraction
was 66.5% for AGRI and 54.9% for AGRI+COT. AGRI improved this subset in all six
pairs; COT improved it in five of six, with the remaining pair essentially
flat/slightly worse. This diagnostic supports checking how COT is aligned to
the station's cloud shadow; it does not identify cloud height or parallax as
the proven cause.

## OOF-to-full-A shift

The new model consumes only each fold's scalar physical GHI prediction on
training rows; validation uses the full-data A scalar. The fold-by-fold audit
compares both A variants on the same training rows. Sili's OOF/full-A
prediction RMSE gap is generally about 18–26 W/m². Zhujia has a much larger
shift: fold 2 averages 185.39 W/m² OOF RMSE versus 123.15 W/m² from full A on
those exact rows, and their paired prediction RMSE gap is 84.44 W/m². Fold 3
has 124.25 versus 95.00 W/m² and a 49.90 W/m² prediction gap. This confirms a
material scalar-base distribution shift in the blocked OOF design. The
unified-input model still improves held-out validation, but this result cannot
isolate the latent-feature mismatch as the only reason v2 failed.

## Interpretation and next action

The architecture change addresses a real inconsistency: training no longer
uses unrelated fold-specific hidden bases while validation uses the full-A
hidden basis. The new residual encoder is also trainable and increases
capacity from the earlier v2 residual model, so the improvement over v2 cannot
be attributed to hidden-feature alignment alone. The fair COT comparison is
within v3, where both arms share the same architecture, training schedule,
initialization seed, and parameter count.

For the current validation contract, use A + AGRI residual as the stronger
candidate. Do not claim COT improves GHI. The next evidence-based step is a
bounded audit of whether a cloud-height source and geolocation support exist
for these train/validation rows, then project COT-bearing cloud locations
along the solar direction onto the station footprint and compare aligned COT
against the current unprojected COT in the same paired validation cases. If
cloud height is unavailable or unreliable, record that limitation before
designing a height proxy. Keep test closed.

## Provenance

- Split: `hunan_monthly_block_2d_buffer1d_seed42_v1`; pack manifest SHA-256:
  `27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4`.
- PBS: `214017.tc6000`, node23 GPU 2, NVIDIA A100-SXM4-40GB, `exit_status=0`.
- Date-block bootstrap: 10,000 paired replicates over 52 common BJT validation
  dates; seed averaging within station and equal station weighting.
- `test_payload_opened=false`; target truth is used only for losses and
  validation diagnostics.
- Saved predictions, signed corrections, per-seed metrics, direction groups,
  fold-shift table, and checkpoints are under
  `results/signed_residual_oof_v3_unified_analysis_20261014_v1/` and
  `results/signed_residual_oof_v3_unified_20261014_v1/`.
