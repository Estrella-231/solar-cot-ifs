# New-monthly AGRI to COT retriever R

This is an independent reconstruction of the 16x16 Hunan station-patch COT
retriever for `hunan_monthly_block_2d_buffer1d_seed42_v1`. It is initialized
from scratch and trained only on train-date AGRI/CPP pairs. Validation dates
are for model selection; test payloads are never parsed or opened.

Inputs are the 13 frozen AGRI channels plus `cosSOZ`, `cosRAA`, and
`day_mask`. Targets are CPP COT with the already-audited mask: finite COT in
[0,100], CLP in {0,1,2}, all 13 AGRI channels valid, daytime, and
`cosSOZ > cos(80 deg)`. Valid zero-COT pixels remain supervised; missing CPP
or failed reads are recorded as unavailable, never converted to clear sky.

The network matches the historical two-downsample COT U-Net topology and
32-channel base width, with a fresh initialization and train-only normalization.
The target/loss follow the frozen monthly contract (`log1p(COT)` masked Huber).
CPP is a retrieval
reference, not independent truth. Any accuracy claim must be scoped to the
CPP-supported dates and pixels, whose monthly coverage is incomplete.

The full train/validation CPP-supported pack passed its split, geometry, shape,
mask, zero-COT, and array-hash audit. A fresh seed-42 monthly R was trained on
one A800 and independently rescored in FP64. Its validation metrics describe
CPP-supported daytime retrieval-reference pixels only, not independent truth.
Thirteen train timestamps had NetCDF/HDF read errors; these were retained as
unavailable source rows and were not converted to clear sky. See
`PROGRESS_20261013.md` for the run and validation record.

A separate COT bank was generated from the same observed train/validation
AGRI frames with this new monthly R. The bank contains 55,733 unique frames,
with zero source errors and no test rows; its manifest binds the bank to the
new R checkpoint. The previously completed bank used the older R and remains
a separate historical diagnostic.

The matched observed-AGRI GHI head experiment is complete for seeds 42/43/44:
two stations, A/AC/AP, 30 epochs per head. Independent FP64 validation scoring
and paired date-block bootstrap found a three-seed station-equal AC−A change of
−3.740 W m⁻² and AC−AP of −3.582 W m⁻², with both 95% intervals below zero.
This supports a validation-cohort COT-input gain beyond the AP capacity
control, but does not establish independent-test generalization or forecast
chain benefit. The frozen-head perturbation shows dependence on COT values and
sample alignment, with little response to one center-preserving permutation
of the eight local neighbors; it is not evidence that this head learned cloud
texture. See `results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md` and
`results/COT_INPUT_DEPENDENCE_VALIDATION_20261013.md`.

Full checkpoints and per-sample predictions are held in the server result
directory listed in `PROGRESS_20261013.md`; the repository stores aggregate
audits, provenance hashes, source code, and reports. No test payloads were
read.

The exact train/validation-only target manifest is archived at
`input/target_candidates_train_val.csv` (SHA-256 is recorded in
`PROGRESS_20261013.md`); it contains no test rows.
