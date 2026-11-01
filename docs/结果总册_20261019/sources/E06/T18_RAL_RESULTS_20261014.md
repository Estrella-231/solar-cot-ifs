# T18 R-AL hidden-feature control: validation results

Date: 2026-10-14 BJT. Scope: frozen monthly split validation only; same 4,479 `(station,target_time_utc)` rows, same GHI truth/mask, same three seeds for A/AC/AP/AL. AL consumes the frozen new-monthly R `dec1` feature crop (32×3×3) with a 32→4 auxiliary branch parameter-matched to AP. Test payloads remain closed.

## Results

Station-equal RMSE first averages the three seeds within each station, then gives the two stations equal weight.

| Head | Sili RMSE | Zhujia RMSE | Station-equal RMSE |
|---|---:|---:|---:|
| A | 135.156 | 112.805 | 123.980 |
| AC | 132.368 | 108.113 | 120.240 |
| AP | 135.073 | 112.571 | 123.822 |
| AL | 132.665 | 106.950 | 119.807 |

Paired difference is candidate minus comparator; negative favors the first-named candidate.

| Contrast | Δ RMSE (W m⁻²) | 95% BJT-date-block CI |
|---|---:|---:|
| AL − A | −4.173 | [−7.137, −1.390] |
| AL − AC | −0.433 | [−1.476, +0.677] |
| AL − AP | −4.015 | [−6.914, −1.265] |
| AC − A | −3.740 | [−6.074, −1.647] |

AL reduces error against A and AP on this validation cohort. AL−AC is unresolved: the interval crosses zero, so this is not evidence of statistical equivalence, but it does not show an incremental AC advantage over the capacity-matched R-hidden representation. By station, AL is 0.297 W m⁻² worse than AC at Sili and 1.163 W m⁻² better at Zhujia, which cautions against an overall-only interpretation.

## Interpretation and limits

The COT input remains useful versus the plain AGRI head in the existing validation comparison. T18 shows that a frozen R hidden feature trained under the same CPP supervision can provide a similar downstream signal. Therefore the current evidence does not isolate explicit COT physical semantics as the unique cause of the GHI improvement. The hidden feature is neither an independent cloud observation nor a deployable alternative claim; both AC and AL inherit the same frozen R and its supervision. This is a validation-only representation control, not a forecast-GHI score and not a test generalization result.

The date-block bootstrap used 2,000 paired resamples, preserving all methods and seeds within each sampled BJT date; station date support was resampled separately (44 Sili dates, 50 Zhujia dates), then station means were averaged. Missing dates at one station are not imputed. This is a station-separate resampling sensitivity. A later paired-method reconciliation applied both this method and the preregistered shared 52-date union bootstrap to the same saved predictions: the shared-date AC−A interval is [−6.180, −1.597] W m⁻² and exactly matches the primary T14 analysis; the station-separate AC−A interval is [−6.074, −1.647] W m⁻² and exactly matches the original T18 recomputation. The point estimate and 4,479 paired rows are identical. Therefore, use the shared-date interval for the primary AC−A claim and retain the T18 interval only as sensitivity; the difference is due to resampling design, not a prediction or metric defect. See [bootstrap reconciliation](ci_method_reconciliation_20261014/T18_BOOTSTRAP_METHOD_RECONCILIATION_20261014.json).

## Reproducibility

- Six AL fits completed in PBS `213869.tc6000`, exit status 0; 30 epochs, batch 2,000, seeds 42/43/44, one A800. Training outputs: server `results/r_al_formal_t18_20261014_v1/`.
- Independent audit: [fp64_independent_audit.json](../../experiments/monthly_R_station16_20261013/results/r_al_formal_t18_20261014_v1/fp64_audit_v3/fp64_independent_audit.json)
- Paired saved outputs and keys: `paired_predictions_fp64.npz`, `paired_val_keys.csv` in that audit directory.
- Local second implementation recomputed all point estimates and bootstrap limits from the saved arrays with maximum absolute discrepancy 0.0: [local_independent_recompute.json](../../experiments/monthly_R_station16_20261013/results/r_al_formal_t18_20261014_v1/fp64_audit_v3/local_independent_recompute.json)
- Independent recomputation code: [verify_t18_r_al_saved_predictions.py](../../experiments/monthly_R_station16_20261013/verify_t18_r_al_saved_predictions.py)
- All prediction artifacts are bound to GHI pack manifest SHA-256 `27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4`; audit state is `INDEPENDENT_FP64_T18_AL_AUDIT_PASS`, local recomputation is `PASS_LOCAL_INDEPENDENT_FP64_RECOMPUTE`.
