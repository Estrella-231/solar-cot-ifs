# Zhujia COT residual capacity control (2026-09-27)

## Question

Does forecast COT add validation value to a zero-initialized GHI residual corrector after controlling for the extra trainable capacity? This validation-only pilot compares AGRI-only residual against AGRI+COT residual under matched architecture, parameter count, initialization, cohort, seed, optimizer, and update budget. It does not test COT in general.

## Protocol and integrity

- Zhujia only; the pilot bank supplied 485 training sequences and 127 validation sequences. The paired validation set contains 1,624 station/lead rows across 70 initialization dates.
- Both arms start from the same frozen no-COT GHI anchor and the same randomly initialized residual branch (192,433 trainable parameters). The AGRI-only arm zeroes COT values but keeps the temporal validity mask; the COT arm receives causal forecast-COT maps.
- Both arms used seed 42, batch 512, learning rate `3e-4`, and exactly 600 optimizer updates. The residual output is initialized to zero, with `tanh` limit ±0.15 kt and delta penalty 0.05. Checkpoint selection is validation RMSE with strict improvement; step 0 is an eligible exact fallback.
- PBS 210971 passed the 3-update-per-arm GPU sanity run (exit 0). Formal PBS 210972 ran on node21/GPU0 and completed with exit 0. Its completion record confirms 600 updates per arm, the approved frozen-Sili hash, equal capacity, matching initialization, and `test_used=false`.
- Independent paired audit confirms identical 1,624 row IDs, truths, and leads; all rows belong to Zhujia. The initialization-date bootstrap used 5,000 resamples. No test set was read.

## Results

| Candidate selected on validation | Best step | RMSE (W m⁻²) | MAE (W m⁻²) | Bias (W m⁻²) |
|---|---:|---:|---:|---:|
| Frozen no-COT anchor | 0 | 127.2954 | 93.0143 | −11.7035 |
| AGRI-only residual | 276 | 126.3232 | 92.0802 | −18.7875 |
| AGRI+COT residual | 0 (fallback) | 127.2954 | 93.0143 | −11.7035 |

The selected COT arm is exactly the frozen anchor. Relative to the AGRI-only residual, its RMSE is higher by 0.9722 W m⁻². The paired initialization-date bootstrap 95% interval for (COT − AGRI) RMSE is [−1.4884, 3.3365] W m⁻², and 22.72% of resamples favor COT. Thus the point estimate favors AGRI-only, but the interval does not establish that either residual arm is superior. Lead-wise differences change sign and show no consistent COT gain.

During COT-arm training, training loss decreased, but validation RMSE stayed above the step-0 fallback and reached 149.68 W m⁻² at step 600. The AGRI-only arm's selected validation result was 0.9722 W m⁻² lower than the anchor; this is a small, single-seed point estimate and was not separately tested with a paired bootstrap against the anchor.

![Saved validation RMSE and training-loss curves for both residual arms](/F:/CODE_Classify/辐照度预报/solar-cot-ifs/results/zhujia_residual_capacity_control_20260927/training_curves.png)

*Figure 1. Actual saved training/validation curves. The dashed line is the frozen no-COT anchor and the selected COT step-0 result. Markers distinguish the two arms in addition to color. There is no uncertainty band because this is one seed; the RMSE axis spans 120–155 W m⁻² to include the full observed trajectory.*

## Claim gate and interpretation

**Claim supported: no.** A secondary result-to-claim review judged that this run does not support “forecast COT improves Zhujia GHI beyond the same-capacity AGRI residual corrector.” Supported wording is limited to: *in this single-seed, validation-only Zhujia capacity-control pilot, the COT residual selected its exact baseline fallback while AGRI-only selected a slightly lower validation RMSE; the paired interval includes zero.* Confidence is high only for this scoped no-go judgment, not for a universal claim that COT is ineffective or harmful.

The result separates extra residual capacity from a COT contribution: the slight point-estimate improvement appeared in the AGRI-only arm, while adding forecast COT did not improve selected validation predictions. The widening COT validation error despite falling training loss indicates poor generalization under this small pilot and current optimization/feature contract; it does not identify whether the limiting factor is COT fidelity, feature scaling, lead-dependent usefulness, or optimization.

## Next action

Keep the independent temporal test closed and do not change the station model. Before expanding this COT residual configuration, inspect forecast-COT fidelity, scaling, and valid-pixel coverage by lead and cloud regime; then, only if that chain is sound, repeat the frozen protocol on a materially larger train/validation bank across multiple seeds with a predeclared validation gate and paired lead/regime reporting. Open test only after stable validation gains pass that gate. If COT repeatedly selects step 0, retire this configuration rather than claiming a general COT null.

## Artifacts

The paired audit and compact training evidence are in `results/zhujia_residual_capacity_control_20260927/`. Checkpoints remain on the server and are identified by hashes in `complete.json`; this report and validation outputs do not imply deployment. Figures and case provenance are documented separately in `docs/ZHUJIA_COT_EVENT_FORENSICS_20260927.md`. The curve figure was generated with Matplotlib following the procedural guidance of Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065 (2026), https://doi.org/10.48550/arXiv.2609.00065.
