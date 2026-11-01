# Sili low-COT penalty ablation results (2026-09-28)

## Outcome

PBS `211063.tc6000` completed with `Exit_status=0` on node22, one NVIDIA A800 GPU. The fixed train/validation bank, seed 42, architecture, optimizer, batch size 8, sample order, 512 training sequences, and 512 validation sequences were shared by the two serial arms. Each ran eight epochs (512 optimizer updates). The sole intended change was whether the extra penalty on predictions above COT 10 where CPP-reference COT<5 was included. `test_used=false`; no GHI head was trained.

The change moved the learned model in the expected direction for thick COT, while increasing its low-COT false-positive rate. It did **not** make a learned checkpoint better than step-0 pure transport under the predeclared thick-COT selection metric, so both arms selected step 0.

| Candidate on fixed validation bank | Thick COT RMSE | All-regime RMSE | Thick misses (<10) | COT≥30 IoU | False-COT rate (CPP COT<5) |
|---|---:|---:|---:|---:|---:|
| Step-0 transport, shared baseline | 26.504 | 9.520 | 334,790 | 0.2411 | 0.05814 |
| Control, best learned epoch 1 | 28.279 | 9.056 | 371,192 | 0.1664 | 0.03107 |
| Penalty-off, best learned epoch 3 | 26.856 | 9.353 | 328,427 | 0.2307 | 0.03694 |

The penalty-off best learned epoch improved thick RMSE by 1.423 COT versus the control's best learned epoch and reduced severe thick misses by 42,765, while its all-regime RMSE was 0.297 higher and false-COT rate was 0.00587 higher. The penalty-off checkpoint still had thick RMSE 0.352 above step-0 transport, so it was not selected. At matched epoch 3, removing the penalty lowered thick RMSE by 1.962 and increased COT≥30 IoU by 0.0377, while all-regime RMSE rose by 0.329 and false-COT rate rose by 1.72 percentage points. These are pooled validation diagnostics, not independent estimates.

## Interpretation and limits

This single-seed result supports the hypothesis that the extra low-COT penalty contributes to suppressing thick COT in the learned residual. It is not the whole explanation: removing the term did not produce a model that beat pure transport on the selected thick-cloud objective. The learned models still trade lower overall error and fewer false COT values for worse thick-cloud amplitude. The selected output in both arms is exactly the same zero-residual/transport baseline.

The training job retained epoch-level pooled metrics and step-0 checkpoints, but did not retain unselected epoch-1/epoch-3 weights or per-pixel predictions. Therefore the planned per-lead and station comparison for the **unselected learned candidates**, and paired initialization-day bootstrap intervals for penalty-on versus penalty-off, cannot be reconstructed from this run. Do not describe the pooled counts (overlapping windows and pixel-leads) as independent samples or claim statistical significance. The selected step-0 outputs inherit the already completed paired station/patch audits; those do not establish a learned-loss effect. CPP is a satellite retrieval reference, not independent cloud truth.

## Decision

Do not adopt the penalty-off model in the GHI pipeline: validation selection retained pure transport in both arms. Treat the loss ablation as a directional diagnostic, not a passed model improvement. The next model experiment should address the remaining input limitation—the residual network sees only the last COT frame and its difference from the previous frame, not the full eight-frame COT history—and must preserve step 0 as an eligible candidate. Before a further run, define and implement artifact retention for candidate checkpoints/predictions so per-lead, station, regime, and initialization-day paired uncertainty are available from the same run.

## Provenance

- PBS log: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925/sili_lowcot_penalty_ablation_seed42_20260928_v1.pbs.log`; scheduler trace records `Exit_status=0`.
- Local archived outputs: `results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_seed42_20260928_v1/` (`training_complete.json`, per-arm `epochs.jsonl`, `step0.json`, and selected `best.pt`).
- Training source SHA-256: `f14db1c8ba78069c9af4d64835ac0715cf4077b4f949453bccbfcaff318d71a2`.
- Visualization: `results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_seed42_20260928_v1/ablation_validation_curves.png/.pdf`, generated from the saved epoch metrics by `scripts/plot_sili_lowcot_penalty_ablation_20260928.py`. Parsed epoch summary is in `analysis/ablation_analysis.json`.
- Test and GHI-head training remained closed.
