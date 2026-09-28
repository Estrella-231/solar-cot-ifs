# Sili full-history COT pilot results (2026-09-28)

## Decision

The full-history input improves the best learned thick-COT patch score relative to the short-history learned candidate, but it does **not** pass the predeclared gate against pure transport. Both arms select step-0 transport. Keep pure transport as the current COT trajectory baseline; do not send either learned correction to the GHI head and do not open the test split.

## Run and support

PBS `211083.tc6000` completed on one A800 with `Exit_status=0`. The run used one common 512-sequence training bank and 512-sequence validation bank, seed 42, and a profiled batch size of 512 (39.60 GiB peak allocated). The two arms ended at 256 optimizer updates under the fixed early-stop rule. The maximum absolute difference between the mapped initial functions was zero. The arms differ only in COT history input: the short-history representation versus all eight historical COT maps. The remaining head, initialization mapping, optimizer, loss, data order, validation cohort, and step-0 transport candidate were held fixed.

Evaluation uses CPP-valid pixels intersected with seven-pair C13 transport in-domain support, with the same support for all methods. CPP is a retrieval reference, not independent cloud truth. The 512 validation windows overlap. All intervals below resample the 72 BJT initialization days (5,000 bootstrap draws, seed 20260928); because model selection and these intervals reuse the fixed validation cohort, intervals are descriptive and post-selection, not confirmatory generalization evidence.

## Validation results

| Candidate | Patch all-pixel RMSE | Patch COT≥30 RMSE | Patch thick miss rate | Station all RMSE | Station COT≥30 RMSE | Station thick miss rate |
|---|---:|---:|---:|---:|---:|---:|
| Step-0 pure transport | 9.520 | **26.504** | 0.2747 | 8.738 | **25.163** | 0.2304 |
| Best unselected short-history checkpoint (update 192) | 9.380 | 27.313 | 0.2883 | 8.115 | 27.743 | 0.3039 |
| Best unselected full-history checkpoint (update 48) | **9.300** | 26.938 | 0.2662 | 8.101 | 27.148 | 0.2549 |

Lower RMSE/miss rate is better. The full-history checkpoint improves patch thick RMSE by 0.375 versus the short-history checkpoint; its paired day-bootstrap 95% interval for full minus short is `[-0.594, -0.172]`. This is a useful representation signal, but both checkpoints were selected on the same validation set, so it is exploratory. Against transport, full-history patch thick RMSE is still +0.434; the paired 95% interval is `[-0.199, 1.096]`, spanning zero. At the station, full-history thick RMSE is +1.986 versus transport, with interval `[+0.280, +3.420]`, indicating deterioration on this validation cohort. The learned candidate lowers its clear-pixel false-COT rate (0.0419 versus 0.0496 for transport) while missing more of the station thick cases (0.2549 versus 0.2304), another regime trade-off.

The best learned checkpoints were not the selected outputs: the frozen selection rule chose `step0_zero_residual_transport` in both arms. Thus the modest all-pixel RMSE improvement of the unselected full-history checkpoint is not sufficient to advance this model. The result suggests that temporal context is informative but the present residual objective/parameterization fails to convert it into reliable station thick-cloud recovery. It does not establish whether transport, COT retrieval, or the station labels are the sole cause.

## Scope and next action

This is a validation-only COT-reference experiment. It did not evaluate GHI skill, use independent cloud truth, or establish a result on unseen dates. Do not interpret improved COT RMSE as a GHI gain.

Close the current pointwise residual-head branch for now and retain step-0 transport as the baseline. Before another learned thickness run, use the already saved validation sequence bank to diagnose event-level thick-cloud growth, decay, and passage at the station: verify whether C13 transport placed the event correctly and whether the remaining error is primarily amplitude evolution, timing, or retrieval mismatch. Any follow-up should be one-factor, keep the test split closed, and require improvement on station thick events without a supported clear-sky regression before a GHI-head comparison.

## Reproduction artifacts

- Protocol: `docs/SILI_FULL_COT_HISTORY_PILOT_PROTOCOL_20260928.md`
- Trainer and smoke: `scripts/train_sili_full_history_pilot_20260928.py`, `scripts/smoke_sili_full_history_pilot_20260928.py`
- PBS launcher: `scripts/run_sili_full_history_pilot_20260928.pbs`
- Analysis: `scripts/analyze_sili_full_history_pilot_20260928.py`
- Raw run, epoch records, step-0 metrics, per-day sufficient statistics, batch profile, PBS log, analysis JSON, and figure: `results/regional_station_cot_20260927/sili_full_history_pilot_seed42_20260928_v1/`
- Test use: `false`; no GHI head was trained or evaluated.
