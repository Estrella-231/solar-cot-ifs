# AFNO COT paired frozen-GHI-head comparison (2026-09-23)

AFNO COT replaces only future forecast COT. For each station and head, the exact same frozen GHI-head checkpoint, history COT, train-only COT normalization, satellite image and solar metadata, causal mask, row keys, observed GHI, and clear-sky denominator were held fixed.

Negative ΔRMSE favors AFNO. Intervals are 2,000-replicate paired bootstrap intervals resampling BJT initialization days, with all leads for a day kept together.

| Station | Head | Rows | SimVP RMSE | AFNO RMSE | Δ (AFNO−SimVP) | 95% day-bootstrap CI |
|---|---|---:|---:|---:|---:|---:|
| Sili | AC_F | 49,806 | 166.401 | 166.307 | -0.094 | [-0.511, +0.371] |
| Sili | AC_ST | 49,806 | 165.364 | 165.401 | +0.037 | [-0.249, +0.333] |
| Zhujia | AC_F | 50,043 | 140.691 | 140.550 | -0.141 | [-0.343, +0.043] |
| Zhujia | AC_ST | 50,043 | 137.423 | 137.185 | -0.238 | [-0.772, +0.235] |

Pooled over both stations (each station retains its own trained head; resampling BJT days jointly across stations):
- AC_F: SimVP 154.053, AFNO 153.937, Δ -0.115 W m⁻²; 95% CI [-0.359, +0.153] (n=99,849, 72 days).
- AC_ST: SimVP 152.003, AFNO 151.916, Δ -0.087 W m⁻²; 95% CI [-0.352, +0.167] (n=99,849, 72 days).

All four intervals include zero. The direction is mixed at Sili (AC_F improves slightly; AC_ST is slightly worse) and AFNO is lower at Zhujia for both heads, but none of these validation-only differences is statistically resolved by this bootstrap.

Integrity: 99,849 validation rows total (49,806 Sili; 50,043 Zhujia), 72 initialization days; test data were not accessed. Replaying the original SimVP-COT head predictions differs by at most 0.00264 kt. The SimVP COT bank matched the frozen cache exactly (maximum absolute difference 0).

Scope: exploratory post-hoc input-swap diagnostic. The seed-42 GHI heads were trained on SimVP COT and selected using validation, so the result is neither an AFNO-trained GHI head nor independent test evidence. AFNO has a COT-distribution shift relative to the head's training input; a fair AFNO comparison for a paper claim requires retraining the same GHI-head architecture on AFNO COT using the same train/validation splits and then evaluating the held-out test once.

Raw result: `complete.json`; per-row predictions: `*_paired_predictions.npz`; summary data: `paired_summary.csv`.

Visualization workflow reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065.
