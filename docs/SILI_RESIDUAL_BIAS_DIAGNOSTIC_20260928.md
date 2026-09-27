# Sili station residual-bias diagnostic (2026-09-28)

## Scope

This is a descriptive follow-up on the same fixed 512-sequence validation bank and the already selected epoch-1 transport-plus-thickness checkpoint. It compares the learned output with its pure C13-transport/step-0 candidate at the Sili station pixel (patch index 32,32), using the common CPP-valid and transport-in-domain mask. CPP is a retrieval reference, not independent in-situ cloud truth. The 512 windows overlap; the 5,664 valid station-lead records are not independent. No test data were opened.

The diagnostic bins records by **reference CPP COT** only to describe where the learned correction acts. These bins are not an inference-time routing rule.

## Findings

Across all 5,664 valid station leads, the learned correction changes COT by a mean **−0.825** relative to transport and lowers mean absolute error by 0.603. The all-lead RMSE difference (learned minus transport) has a 95% initialization-day block-bootstrap interval of **[−1.609, +0.133]**, so aggregate RMSE improvement is not resolved by this diagnostic.

The correction becomes strongly negative in the thickest reference regimes:

| CPP reference COT | n | Mean learned-minus-transport COT | Transport RMSE | Learned RMSE |
|---|---:|---:|---:|---:|
| 0–<5 | 3,530 | −0.36 | 4.654 | 3.187 |
| 5–<10 | 830 | −0.73 | 10.177 | 7.868 |
| 10–<20 | 809 | −1.95 | 10.829 | 8.844 |
| 20–<30 | 291 | +0.20 | 12.573 | 12.714 |
| 30–<50 | 158 | −5.20 | 21.117 | 23.410 |
| ≥50 | 46 | −9.60 | 35.725 | 42.429 |

On CPP≥30 cases (n=204), thick-cloud RMSE is **28.816 vs 25.163** for transport; the day-block bootstrap interval for the RMSE difference is **[+1.407, +5.591]**. Thick misses (prediction <10) increase from 47/204 to 64/204. The correction is negative on 64.6% of 30–<50 cases and 93.5% of ≥50 cases. Per-lead bins show the largest downward change at short leads, while the ≥50 bin remains negative in all four lead groups (counts are small: 7–15 per group).

The plotted paired pattern supports a **thick-cloud amplitude suppression** hypothesis for this selected residual model: it improves frequent low-COT errors and clear-reference false alarms, but tends to lower already underpredicted thick COT further. It does not identify the cause. The low-COT penalty, log-domain SmoothL1 objective, residual training/selection, limited history encoding, motion/transport errors, and CPP label behavior remain possible contributors.

## Interpretation and next action

This strengthens the case for a controlled loss ablation before changing the encoder: remove only the extra low-COT false-positive penalty, preserve the fixed cohort, seed, model, optimizer budget, all other weights, and validation criterion, and include the zero-residual transport candidate in epoch selection. Evaluate both station and 64×64 patch metrics by lead and CPP regime. Do not use the result to claim improved physical cloud thickness or GHI forecast skill.

The broader 64×64 patch/case audit is already PBS job `210981.tc6000`. A live `qstat -u slfu` check on 2026-09-28 showed 210935 and 210980 running and 210981 still queued (requested node22, one GPU); it has not started, and there is no audit output yet. Do not duplicate it. Check its status and artifacts before authorizing the loss ablation. Account queue count at that snapshot was three unfinished jobs, within the project ceiling.

## Reproducibility

- Analysis: `scripts/analyze_sili_residual_bias_val512_20260928.py`
- Input: compact validation tracks `experiments/audit_sili_pilot_snapshot_20260928/station_tracks_val512.npz`
- Source summary: `results/regional_station_cot_20260927/sili_station_val_audit_seed42_20260928_v2/summary.json`
- Machine-readable output: `results/regional_station_cot_20260927/sili_residual_bias_val512_20260928/residual_bias.json`
- Figure: `results/regional_station_cot_20260927/sili_residual_bias_val512_20260928/residual_bias_by_cot_and_lead.png` and `.pdf`
- Day-block bootstrap: 72 initialization BJT days, 2,000 replicates, seed 42. It keeps same-day overlapping windows together; it remains a descriptive uncertainty check, not proof of independent sampling.
