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

### Residual magnitude follow-up (2026-09-28)

A paired station-pixel diagnostic further stratified the selected epoch-1 correction by reference CPP COT and lead. The learned residual averaged −0.83 COT relative to pure transport overall, but the correction reached −5.20 for CPP 30–<50 and −9.60 for CPP≥50. For all CPP≥30 station leads, thick RMSE rose from 25.16 to 28.82; the 72-day block-bootstrap interval for the difference was [+1.41,+5.59]. The aggregate all-lead RMSE interval crossed zero. This supports a thick-cloud amplitude-suppression symptom in this validation checkpoint, not a causal attribution to the loss or evidence of physical/GHI benefit. CPP remains a retrieval reference and overlapping windows remain dependent. Full details, per-lead figure, script, and JSON are in [the residual-bias diagnostic](SILI_RESIDUAL_BIAS_DIAGNOSTIC_20260928.md).


### Full local-64 patch audit (2026-09-28)

The read-only validation audit completed as PBS `210984.tc6000` after two earlier attempts exposed target/mask and forecast-lead slicing bugs. Those failed attempts produced no accepted metrics. The corrected script asserts `[512,16,64,64]` for every target, mask, support field, and prediction before scoring. It evaluated the fixed epoch-1 checkpoint and the exact validation bank; `test_used=false`. On this bank, 24,355,048 CPP-valid pixel-leads intersected with 21,153,251 common transport-supported pixel-leads (86.9%). The shared mask was applied identically to all models for metrics and the corrected case maps.

The 64×64 patch reproduces the pooled pilot ranking: all-regime RMSE is 9.666 for SimVP→R, 9.609 persistence, 9.520 clipped transport/step 0, and 9.056 learned residual. For CPP COT≥30, RMSE is 27.700, 27.078, 26.504, and 28.279 respectively; COT≥30 IoU is 0.1806, 0.2259, 0.2411, and 0.1664. The learned model's aggregate improvement therefore comes mainly from lower COT regimes and does not translate to thick-cloud skill. In the 30–60 bin, learned RMSE is 25.50 versus 23.85 for transport; in 60–100 it is 49.26 versus 46.44. These pixel-lead counts are clustered and should not be read as independent sample size.

All 16 lead scores show the same tradeoff: learned all-regime RMSE is favorable over long leads, while its COT≥30 RMSE is worse than transport at most leads and approaches it only near the end. Shared-support coverage declines from 96.0% at +15 min to 78.6% at +240 min. The zero-residual output exactly equals clipped transport on the shared support. Four preset validation cases visualize cloud growth (+135 min), decay (+60 min), persistent thick cloud (+180 min), and a clear-reference case (+15 min). In the growth/thick examples, the learned field is visibly weaker than the CPP retrieval around the marked station; interpret this as a model-versus-retrieval discrepancy, not cloud truth. Unsupported pixels are gray in every panel.

Artifacts: `results/regional_station_cot_20260927/sili_patch_case_audit_20260928_v3/summary.json`, `case_manifest.json`, `case_maps_common_support.png/.pdf`, `lead_regime.png/.pdf`, and `case_maps.npz`. The raw audit's unmasked-preview `case_maps.png` is retained for provenance; use the `_common_support` plot for interpretation. Reproduction scripts are `scripts/audit_sili_patch_and_cases_20260928.py`, `scripts/render_sili_patch_cases_common_support_20260928.py`, and `scripts/plot_sili_patch_lead_regime_20260928.py`.

At the time this audit snapshot was written, the single-factor loss-ablation protocol in `docs/SILI_LOSS_ABLATION_PROTOCOL_20260928.md` was complete and training had not yet been submitted. The later authorized run and its evidence-bounded result are recorded in `docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md`. Test remains closed.

### Full local-64 validation audit completed (2026-09-28, supersedes submission snapshot above)

The completed artifact is `experiments/regional_station_cot_20260925/sili_patch_case_audit_seed42_20260928_v3/` on the server, mirrored locally at `results/regional_station_cot_20260927/sili_patch_case_audit_20260928_v3/`. Its state is `COMPLETE_SILI_LOCAL64_VALIDATION_AUDIT`, `test_used=false`, best checkpoint epoch 1. The successful PBS log contains `SILI_PATCH_CASE_AUDIT_COMPLETE`; a later duplicate submission stopped at the guard because this output directory already existed. Earlier attempts exposed tensor indexing/shape errors and are not result sources. The completed code hash recorded in the success log and local source is `0bad5f15552b918f88699e58f4f4b2a912be120b77f6fcf10bcfd6310c3e9a3a`.

The analysis pools 21,153,251 CPP-valid ∩ transport-supported patch pixels across overlapping validation windows and all 16 leads. Transport support declines from 96.0% at +15 min to 78.6% at +240 min. Learned residual improves the all-COT RMSE from 9.520 to 9.056, but this aggregate masks a regime tradeoff:

| CPP COT regime | n | Transport RMSE | Learned RMSE | Transport bias | Learned bias |
|---|---:|---:|---:|---:|---:|
| 0–5 | 11,359,870 | 4.766 | 3.029 | +1.418 | +0.793 |
| 5–10 | 3,719,564 | 7.847 | 6.339 | +0.835 | +0.113 |
| 10–30 | 4,855,014 | 10.873 | 10.212 | −2.522 | −3.692 |
| 30–60 | 1,116,315 | 23.853 | 25.502 | −16.754 | −20.972 |
| 60–100 | 102,488 | 46.441 | 49.260 | −39.973 | −46.307 |
| ≥30 | 1,218,803 | 26.504 | 28.279 | −18.707 | −23.102 |

For COT≥30, IoU at threshold 30 falls from 0.356 to 0.203. The head helps the frequent COT<10 regimes, while making thick-cloud underestimation worse; this matches the station-pixel diagnosis and supports testing the low-COT penalty as a hypothesis, not treating it as a proven cause. At +15 to +75 min, learned all-COT RMSE is worse than transport; it is lower from +90 onward, while thick-cloud RMSE remains worse through +165, is marginally lower at +180 to +225, then slightly worse at +240. These validation-only pooled counts are not independent due to overlapping sequences, and CPP remains a retrieval reference, not independent cloud truth; none of these COT metrics establishes a GHI gain.

The original unmasked case plot showed bright values at the right boundary where transport had no supported upstream source. Those pixels were excluded from all reported paired metrics. A corrected local visualization masks every method to CPP-valid ∩ transport-supported pixels (gray marks excluded pixels): `results/regional_station_cot_20260927/sili_patch_case_audit_20260928_v3/case_maps_common_support.png` and `.pdf`, reproducible with `scripts/plot_sili_local64_cases_masked_20260928.py`. The four cases remain illustrative validation selections, not independent proof.

The next action from this patch audit was the one-factor loss ablation, later completed as PBS `211063.tc6000`; see `docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md` for the outcome and artifact limits.

### Paired penalty ablation completed (2026-09-28)

PBS `211063.tc6000` completed with `Exit_status=0` on node22 using one A800 GPU. The two serial seed-42 arms each ran 8 epochs/512 updates on the fixed train/validation bank, differing only in the extra COT<5 false-COT penalty. Both selected the step-0 transport candidate because no learned checkpoint beat its thick-cloud RMSE of 26.504. The best learned checkpoint was epoch 1/control at 28.279 and epoch 3/penalty-off at 26.856; removing the term improved thick RMSE directionally but did not pass the frozen selection gate. Full pooled metrics and missing uncertainty/learned-candidate artifact limitations are documented in `docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md`. Test and GHI-head training remain closed.

The completed patch audit's paired results are mirrored in `results/regional_station_cot_20260927/sili_patch_case_audit_20260928_v3/`. To make repeated examples fully auditable, local sidecar `scripts/plot_sili_local64_cases_masked_20260928.py` regenerates the displayed maps from the saved case arrays and masks excluded transport-source pixels.

### Low-COT penalty ablation completed (2026-09-28)

PBS `211063.tc6000` completed both serial arms on one node22 GPU. The log ends with `SILI_LOW_COT_PENALTY_ABLATION_COMPLETE`; both arm-level and aggregate completion JSONs are present, `test_used=false`. The same local-staged 512/512 bank, seed-42 initialization, shuffled minibatch order, architecture, optimizer, weighting, 40-epoch cap, and eight-epoch patience were used. The only loss change was removal of the additional physical-domain penalty on predictions above 10 for CPP COT<5. Both arms made step-0 transport an explicit candidate selected by thick-cloud RMSE.

| Candidate | Extra low-COT penalty | Epoch | Thick COT RMSE | All-valid COT RMSE | Thick COT IoU |
|---|---|---:|---:|---:|---:|
| Step-0 transport | n/a | 0 | **26.504** | 9.520 | 0.2411 |
| Best learned epoch under penalty-on arm | on | 1 | 28.279 | 9.056 | 0.1664 |
| Best learned epoch under penalty-off arm | off | 3 | 26.856 | 9.353 | 0.2307 |

Under the fixed selection rule, both runs select step-0 transport because no learned epoch beats its thick-cloud RMSE of 26.504. Removing the extra low-COT penalty improves the best learned thick-cloud RMSE by 1.423 and IoU by 0.064 relative to the penalty-on learned candidate, so the penalty plausibly contributed to suppression. It does not explain the full failure: the penalty-off learned candidate still fails to beat transport on thick-cloud RMSE or IoU. This is one seed and one validation cohort; it motivates looking at the residual objective/representation and event timing, not a final physical conclusion. The selection retains the unchanged baseline, so there is no reason to deploy either learned correction from this pilot.

Validation curves are at `results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_seed42_20260928_v1/ablation_validation_curves.png` and `.pdf`; regenerate with `scripts/plot_sili_lowcot_penalty_ablation_20260928.py`. Outputs and per-epoch records are copied under the same local results directory. Test remains closed; no GHI head was trained or evaluated.
