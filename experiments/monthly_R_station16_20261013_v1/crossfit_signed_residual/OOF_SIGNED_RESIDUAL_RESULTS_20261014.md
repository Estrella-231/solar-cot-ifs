# Cross-fitted signed COT residual: validation results

## Question and scope

This experiment asks whether COT still helps a signed station-GHI correction
when the residual branch is trained against A predictions made on rows that
were excluded from fitting that fold's A model. It uses synchronous real AGRI
and retrieved COT, so it is a GHI retrieval diagnostic, not SimVP forecast-AGRI
evaluation. The monthly train/validation pack was used; the independent test
payload remained closed.

The paired design covers Sili and Zhujia, seeds 42/43/44, and 50,458 train plus
4,479 validation rows in the shared pack. For each station/seed, training dates
were split into five contiguous BJT blocks. Each fold A model trained for 30
fixed epochs on the other blocks, with the nearest available train date on
either side buffered out. Fold checkpoints were not selected with validation.
The two residual arms had equal 23,173 trainable parameters and identical
initialization, row order, optimizer, and training schedule. They received
fold-specific OOF A predictions and encoder features on train rows; validation
used the existing full-training A checkpoint.

## Main result

The v1 run selected only trained checkpoints, and every selected checkpoint
was epoch 0 (after one complete optimizer epoch). A review then found that the
untrained, exactly zero residual had been checked but omitted from checkpoint
selection. The v1 results are retained as run history. v2 changed only this
selection rule: the exact step-zero A identity was added as a candidate. If no
trained checkpoint beat it, v2 retained A exactly.

| Selected validation method | Equal-station mean RMSE (W/m²) | Change from A |
|---|---:|---:|
| Frozen A | 123.979 | — |
| A + signed AGRI residual (v2) | 123.845 | −0.134 |
| A + signed AGRI+COT residual (v2) | 123.748 | −0.231 |

Paired 10,000-replicate BJT date-block bootstrap, averaging seeds within each
station and then weighting stations equally:

- AGRI residual − A: **−0.134 W/m²**, 95% CI **[−0.325, 0.104]**.
- AGRI+COT residual − A: **−0.231 W/m²**, 95% CI **[−1.000, 0.418]**.
- AGRI+COT residual − AGRI residual: **−0.096 W/m²**, 95% CI **[−0.989, 0.643]**.

All intervals cross zero. The point differences are small, and the validation
does not establish a reliable COT gain. The original unguarded v1 estimates
were 124.668 W/m² for AGRI residual and 124.675 W/m² for AGRI+COT residual;
both were slightly worse than A.

In v2, 5/6 AGRI checkpoints and 3/6 AGRI+COT checkpoints selected the exact
step-zero A identity. Only three of the twelve residual branches selected a
trained checkpoint at all. AGRI+COT was better than AGRI in two Zhujia seeds,
but tied or worse in the other four station/seed pairs. This is too unstable
and too small to support a performance claim.

## Direction failure remains

On the post-hoc bright/thick diagnostic subset (retrieved center COT ≥ 3,
clear-sky GHI ≥ 100 W/m², observed kt ≥ 0.9), the validation set contains 58
Sili and 36 Zhujia rows. Labels are used only for diagnosis, never as model
inputs or routing gates. In the two Zhujia pairs where v2 selected a trained
AGRI+COT checkpoint, the selected correction was non-positive for every row
in that subset; subset RMSE rose from 252.8 to 266.8 W/m² and from 257.2 to
272.4 W/m². The one trained Sili COT checkpoint adjusted upward on only about
21% of the subset and also worsened its subset RMSE (273.2 to 275.9 W/m²).
The COT residual therefore still does not identify the upward correction
needed when A is too low on these bright, thick-cloud cases.

## What the cross-fit test diagnosed

The fold A errors vary strongly by date block. Sili OOF-train RMSE is about
125 W/m² across seeds; Zhujia is about 145 W/m², with one Zhujia fold near
186 W/m² while another is near 108 W/m². This is evidence of substantial
time-block heterogeneity. It is not a direct like-for-like comparison with
validation, because the OOF rows and validation dates are different cohorts.

The residual branch sees fold-specific A forecasts/features while learning,
then sees the full-training A forecast/features at validation. At Zhujia,
training loss continued falling as validation RMSE rose sharply (for example,
AGRI seed42 from about 113 at the first epoch to 169 by epoch 29). This
strongly suggests that the fold-to-full-A representation/error shift makes
the learned corrections fail to transfer, in addition to the unresolved
physical sign ambiguity. Consequently, the OOF run does not prove that
in-sample targets caused the earlier result; it shows that this particular
blocked cross-fit recipe is not a usable correction model.

The bright-thick failure is physically consistent with the inputs' limit:
local COT measures optical thickness but does not by itself determine whether
the station is under the cloud shadow, the cloud's height/parallax, or the
shadow's displacement relative to the sun. AGRI context and local geometry
were present, but this residual head was not supplied a validated cloud-height
and projected-shadow state. That is a plausible mechanism, not a proven causal
attribution from this experiment.

## Decision

Keep the frozen A forecast as the baseline and do not use either OOF residual
checkpoint as a forecast improvement. The experiment does not justify another
generic COT-only residual-head expansion. The next useful diagnosis is to
make fold A inputs more distribution-matched across dates (e.g. month-stratified
blocked folds) and independently audit fold-level date composition. Any later
physical correction should explicitly represent cloud-height/parallax and
solar-direction shadow alignment, then be compared against the same A and
capacity-matched AGRI control on fixed validation cases. This report is
validation-only and does not authorize opening test.

## Reproducibility

- Contract: `OOF_CONTRACT_20261014.md`; step-zero selection amendment:
  `STEP_ZERO_GATE_ADDENDUM_20261014.md`.
- Builder, residual trainers, analysis, and verifier are in this directory.
- Pack-manifest SHA-256: `27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4`.
- PBS v1 (`213969.tc6000`) and v2 (`213980.tc6000`) ran one A100-SXM4-40GB
  on node23 GPU 2. v2 has PBS `exit_status=0`; v1 log contains the completion
  sentinel. Both output manifests record `test_payload_opened=false`.
- Independent summary and per-seed/subgroup CSVs are stored under `results/`.
- Per-row validation keys, truth, predictions, and signed adjustments are in
  `results/validation_predictions_v1.npz` and `results/validation_predictions_v2.npz`;
  all 24 trained residual checkpoints are in
  `results/residual_checkpoints_v1_v2.tar.gz` (SHA-256
  `797be2a815aaabbffc75bd4167b47f46df238de1a373e0bd36d4078e6a2e248`).
