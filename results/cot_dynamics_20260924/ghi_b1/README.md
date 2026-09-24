# COT dynamics B1: matched Hunan validation screen

This is a single-seed, train/validation-only engineering screen. The frozen
geometry24 Hunan SimVP supplies all 13 forecast AGRI channels; frozen R
supplies COT. A0 zeros COT, AH uses eight observed-history COT maps, and AI
adds 16 COT maps retrieved from SimVP forecast AGRI. All three use the same
GHI-head architecture, row keys, geometry, loss, optimizer, seed, and
validation-selection rule. O_diag keeps the SimVP forecast-image sidecar but
replaces future COT with R(real future AGRI); it is an **offline diagnostic,
not a deployable forecast or an independent physical COT observation**.

On 99,849 matched daylight validation rows, two-station equal-weight RMSE is
A0 **147.639**, AH **148.852**, AI **148.773**, and O_diag **130.109** W/m².
Thus AH−A0 is +1.213, AI−AH is −0.079, and O_diag−AI is −18.664 W/m².
Station RMSE (Sili/Zhujia) is A0 163.557/131.720, AH 164.323/133.380,
AI 164.431/133.116, and O_diag 140.834/119.384 W/m². Lower is better.
The small AI−AH difference changes sign after +150 min in the pooled
lead-wise readout; the separately trained heads and single seed prevent a
mechanistic attribution from that pattern alone. AI−O_diag pooled lead-wise
RMSE widens from **5.32 W/m² at +15 min** to **31.13 W/m² at +240 min**.

A paired bootstrap over the 72 **initialization dates** in the validation
period (2,000 fixed-seed replicates) gives a 95% percentile interval of
**[−2.301, +2.048] W/m²** for AI−AH and **[−23.062, −14.327] W/m²** for
O_diag−AI. The former is unresolved; the latter indicates a large offline
future-COT information gap in this validation period. These intervals do not
include checkpoint-selection optimism, training-seed variation, or
independent-test uncertainty; best checkpoints were selected on this same
validation period. CPP/R production provenance still limits physical
interpretation.

The 16×16 history-only motion pilot lost about half its valid upstream
station-neighborhood support by +240 min. A follow-up fixed-sample pilot moved
the **last observed AGRI** crop upstream according to historical C13 motion
and ran the same frozen R on each native 16×16 crop. It provided source
patches for 100% of 16 fixed validation sequences and 98.4% of 64 fixed
training sequences at all leads, but accuracy was mixed: at +240 min
center-5×5 MAE in `log(1+COT)` was 0.970 versus persistence 0.888 on the
validation pilot, and 0.837 versus 0.859 on the training pilot. This is a
causal, coverage-oriented AP diagnostic, not a validated replacement for E.
The CPU witness differs from the GPU-produced zero-shift historical R bank by
at most 0.0032 (val) / 0.0051 (train) `log(1+COT)`; the discrepancy is recorded
and no bit-identical cross-device output is claimed.

The complete per-lead/station metrics, row-matched predictions, head
completion records, and date-bootstrap audit are in this directory and
`../../../audits/cot_dynamics_20260924/`. Before training E_phys or E_task,
validate native-crop R on the source distribution and full train/validation
coverage, then keep AP and persistence as controls. An improvement in COT
coverage by itself does not establish improved COT or GHI accuracy.
