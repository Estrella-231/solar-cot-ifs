# Sili transport + thickness pilot result (2026-09-28)

## Job and scope

PBS `210963.tc6000` ran on node22 with GPU 7 (`NVIDIA A800-SXM4-80GB`). The log ends with `SILI_COT_DYNAMICS_PILOT_COMPLETE`; `training_complete.json`, `best.json`, `best.pt`, and all nine epoch records are present. The job is no longer in the active queue. PBS history no longer returns the job ID, so an explicit scheduler `Exit_status` was not available during this check. The completion sentinel and final report show normal script completion. Test was not used.

This is one exploratory seed on 512 training and 512 validation sequences, with CPP retrieval products as reference labels, not independent physical truth. Inputs were historical regional R-COT, seven-pair local C13 motion, frozen SimVP→R future COT, and lead. The model trained for 9 epochs and stopped after eight successive epochs without improving the selected validation score; best epoch was 1. Selection minimized thick-cloud RMSE. The validation contains 1,218,803 thick-cloud lead records; windows overlap and are not independent samples.

## Best validation checkpoint

| Method | All-valid COT RMSE | All-valid MAE | COT≥30 RMSE | COT≥30 misses predicted <10 | COT≥30 IoU | Clear false-COT rate |
|---|---:|---:|---:|---:|---:|---:|
| SimVP→R | 9.666 | 5.537 | 27.700 | 387,738 | 0.1806 | 0.0438 |
| Persistence | 9.609 | 5.288 | 27.078 | 346,485 | 0.2259 | 0.0567 |
| Seven-pair transport | 9.520 | 5.287 | **26.504** | **334,790** | **0.2411** | 0.0581 |
| Learned transport + thickness | **9.056** | **4.900** | 28.279 | 371,192 | 0.1664 | **0.0311** |

The learned model improves all-valid COT RMSE and MAE, and reduces the clear false-COT rate. However, its selected thick-cloud RMSE is worse than SimVP→R, persistence, and transport; its COT≥30 IoU is also below all three. It reduces thick-cloud underestimation counts relative to SimVP→R but does not match persistence or transport. The result therefore indicates a clear/thin-cloud versus thick-cloud tradeoff, not an unqualified improvement in cloud motion/evolution. Do not infer downstream GHI benefit from these COT metrics.

At epoch 7 the learned all-valid RMSE reached 8.934, but thick-cloud RMSE was 28.968, so it was not selected under the predeclared thick-cloud criterion. Best epoch 1 is consistent with early stopping; continued training optimized the aggregate training loss without improving the chosen thick-cloud validation score. Avoid selecting epoch 7 post hoc based on the more favorable all-valid RMSE.

## Artifacts and next gate

### Code-audit clarification (2026-09-28)

The local training source matches the currently deployed server copy (SHA-256 `b71b85659cf73b2f5f39cd0f3072ae94fd876345f8a30fda012c2dbdfb40c3b5`). The table's "All-valid" means **64×64 local-patch CPP-valid pixels intersected with transport source support**, pooled over 16 leads. It is not the full 256×256 region or all local pixels. Likewise, "clear false-COT rate" means prediction COT≥10 conditional on CPP COT<5; this reference bin includes optically thin cloud and is not an independent clear-sky classification.

Although eight historical COT frames are loaded, `make_input` feeds only the last COT frame and its difference from the penultimate frame to the learned network. The motion input summarizes seven historical C13 pairs, but this does not constitute learning from the full eight-frame COT history. `warp_batch` transports the final local 64×64 COT frame with one constant translation vector; it cannot import cloud from outside that crop. Training uses the CPP-valid mask, whereas evaluation additionally excludes transport-out-of-domain pixels; the network does not receive the source-support mask as a separate input.

The output residual is zero-initialized, but step 0 is not evaluated or saved as a selection candidate (`best=inf` before epoch 1). A zero-residual checkpoint should be evaluated explicitly, including the effect of output clipping, before interpreting the selected learned checkpoint against transport. The saved epoch-1 thick-cloud score is worse than the independently evaluated transport baseline.

The log-domain SmoothL1 objective (thick-cloud weight 3, COT<5 weight 2) also includes a physical-domain squared penalty for predictions above 10 on CPP COT<5 pixels. This creates an objective tradeoff that may suppress thick-cloud amplitude, but the contribution of each term and signed conditional bias must be measured before assigning causality. Recommended order: replay best and step-0 predictions with per-lead/site/support masks and physical cases; then isolate the low-COT penalty in a fixed-cohort loss ablation; only then test full-eight-frame history and upstream regional transport as separate changes.

### Sili station-pixel validation audit (2026-09-28)

The separate station-pixel audit uses the same fixed 512-sequence validation bank and the same CPP-valid ∩ transport-support mask for every method. It has 5,664 valid station/lead records; 204 have CPP COT≥30. The learned checkpoint is the same best epoch 1 selected on this bank, so these are additional diagnostics on the checkpoint-selection validation set, not an independent test. The step-0 output at the station pixel equals pure transport: no scored transport value exceeded the model's COT=100 output cap.

| CPP COT regime | n | Transport RMSE | Learned RMSE | Learned minus transport RMSE, 95% paired day-bootstrap CI |
|---|---:|---:|---:|---:|
| all valid station leads | 5,664 | 8.738 | **8.049** | -0.689 [-1.589, 0.106] |
| 0–5 | 3,530 | 4.654 | **3.187** | -1.466 [-2.447, -0.607] |
| 5–10 | 830 | 10.177 | **7.868** | -2.309 [-5.705, 0.888] |
| 10–30 | 1,100 | 11.317 | **10.015** | -1.302 [-3.095, 0.687] |
| 30–60 | 186 | **22.993** | 26.051 | +3.058 [+0.781, +4.977] |
| 60–100 | 18 | **41.385** | 48.971 | +7.586 [+4.149, +11.999] |

Across all CPP≥30 records, learned correction reduces station COT by a mean 6.20 and median 5.72 relative to transport; 71.1% of corrections are negative. Thick-cloud bias changes from -18.42 to -24.62, while the rate of predictions below COT 10 rises from 23.0% to 31.4%. The thick-bin date bootstrap spans only 26 initialization dates; the 60–100 bin occurs on 3 dates and has only 18 records, so those intervals are diagnostic rather than decisive. The overall RMSE interval includes zero, while clear-reference improvement is the main source of the aggregate station gain.

This narrows the next hypothesis: the learned residual is too negative on CPP-thick station cases, while helping frequent low-COT cases. The extra low-COT penalty is a plausible contributor, but not yet a proven cause. Before training a new head, prepare a full local64 per-lead/support audit and representative spatial COT maps; then run a fixed-cohort paired loss ablation that removes only the extra false-COT penalty, retaining the same seed, sampling, architecture, optimizer budget, and step-0 transport candidate. Report station and patch metrics together. Do not select a new run based on the small 60–100 subset alone.

Small result artifacts are copied to `results/regional_station_cot_20260927/sili_transport_thickness_pilot/` (`epochs.jsonl`, best checkpoint/metrics, and completion report). The regional bank and complete forecast artifacts remain on the server. The COT GHI workflow status is updated in `docs/REGIONAL_COT_GHI_NEXT_20260927.md`.

Next, audit the saved per-lead and per-regime validation metrics and representative physical cases from this fixed best checkpoint. Keep test closed and do not launch multi-seed or GHI-head training until this tradeoff is understood and the same selection criterion is retained. CPP provenance/physical QA remains an acceptance limitation.

### Full local-64 patch audit submitted (2026-09-28)

A corrected evaluation now scores all 16 leads and all pixels on the 64×64 station patch, stratified by CPP COT regime, and renders four preselected station cases (growth, decay, persistent-thick, clear-reference) with CPP, SimVP→R, persistence, transport/step-0, learned residual, and learned-minus-CPP maps. It uses the existing validation bank and epoch-1 checkpoint; the test split is not copied or opened. The validation bank is staged once to node-local scratch to avoid repeated reads from the shared filesystem. A review caught and fixed the prior draft's support-mask indexing, which had selected only the first lead; that draft was not used for the submitted run.

The script and PBS wrapper compiled under the `swc` Python environment and passed `bash -n`; local and server audit-script SHA-256 is `5cfd6cef73e3fcfcc5f75472898b5b023b62d0af725eaa39057ac2e72e9d186c`. PBS `210981.tc6000` was submitted to `node22` with one GPU and a 1-hour limit. At submission it was queued; the account had two running jobs, within the three-unfinished-job ceiling. No metrics or figures are claimed until the completion marker and artifacts are verified.

After this audit, the next experiment remains a one-factor loss ablation: remove only the extra low-COT false-positive penalty, retain the same fixed 512/512 time cohorts, architecture, seed, optimizer schedule, weighting, and validation selection rule, and add the zero-residual/step-0 transport candidate to epoch selection. Compare full-patch and station metrics by lead and COT regime. Do not proceed to full-history encoding, a wider upstream domain, GHI-head retraining, or the test set until this validation diagnosis is complete.

### Residual magnitude follow-up (2026-09-28)

A paired station-pixel diagnostic further stratified the selected epoch-1 correction by reference CPP COT and lead. The learned residual averaged −0.83 COT relative to pure transport overall, but the correction reached −5.20 for CPP 30–<50 and −9.60 for CPP≥50. For all CPP≥30 station leads, thick RMSE rose from 25.16 to 28.82; the 72-day block-bootstrap interval for the difference was [+1.41,+5.59]. The aggregate all-lead RMSE interval crossed zero. This supports a thick-cloud amplitude-suppression symptom in this validation checkpoint, not a causal attribution to the loss or evidence of physical/GHI benefit. CPP remains a retrieval reference and overlapping windows remain dependent. Full details, per-lead figure, script, and JSON are in [the residual-bias diagnostic](SILI_RESIDUAL_BIAS_DIAGNOSTIC_20260928.md).

The planned 64×64 patch/case audit is PBS `210981.tc6000`. Live queue inspection on 2026-09-28 found it still queued on node22; no output is available and it was not resubmitted. Finish that audit before running the one-factor loss ablation: remove only the extra low-COT false-positive penalty, keep the cohort/model/seed/budget/other weights fixed, include step-0 transport in selection, and score station plus full patch by lead/regime. Keep test closed.
