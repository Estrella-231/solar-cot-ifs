# Sili capacity-matched COT history control results (2026-09-28)

## Decision

The matched comparison does not show a reliable thick-COT benefit from the six older history maps for this residual head. The full-history model improves patch thick-COT RMSE by only 0.067 relative to the repeated-last control, with a 72-day paired interval spanning zero. Station thick-COT RMSE is effectively unchanged. Both arms select step-0 pure transport, and both learned candidates have worse thick-COT RMSE than transport. Close this head's history-input ablation and move to event-level error diagnosis before another model change.

## Experiment and completion

The run completed normally with `SILI_FULL_VS_REPEAT_LAST_COMPLETE` in the PBS log and `COMPLETE_SILI_FULL_VS_REPEAT_LAST_HISTORY_PILOT` in the saved summary. PBS later returned `Unknown Job Id`, so no scheduler `Exit_status` was recovered. The run used one node22 NVIDIA A800 80GB, batch 512, and 256 updates per arm. The arms used the same 15-channel Conv3d architecture and parameter count, optimizer, objective, initialization function (maximum absolute output difference 0), sample order, fixed 512/512 bank, masks, and evaluation keys. The only intended input difference was eight actual historical COT maps versus eight repeats of the final map; both also received the same final-frame difference channel. Test was not loaded.

Evaluation used CPP-valid pixels intersected with C13 seven-pair transport source support. CPP is a retrieval reference, not independent cloud truth. The validation windows overlap. Paired intervals resample 72 BJT initialization days (5,000 draws, seed 20260928); checkpoint selection and intervals reuse the validation cohort, so intervals are descriptive and post-selection.

## Validation results

| Candidate | Patch all-COT RMSE | Patch COT≥30 RMSE | Station all-COT RMSE | Station COT≥30 RMSE | Selected |
|---|---:|---:|---:|---:|---|
| Step-0 pure transport | 9.520 | **26.504** | 8.738 | **25.163** | Yes, both arms |
| Best learned repeated-last (update 160) | 9.440 | 27.209 | **8.192** | 27.606 | No |
| Best learned full-history (update 96) | 9.603 | 27.143 | 8.325 | 27.609 | No |

The point estimate for full-history minus repeated-last thick RMSE is `-0.067` on the patch and `+0.004` at the station. The corresponding paired 95% intervals are `[-0.161, +0.029]` and `[-0.316, +0.293]`. Neither supports a reliable full-history advantage. The full-history candidate also remains above transport by `+0.638` patch thick RMSE and `+2.447` station thick RMSE; the station difference interval is `[+0.573, +3.977]`. Its patch interval versus transport is `[-0.032, +1.336]` and spans zero. Repeated-last is above transport by `+0.705` patch and `+2.443` station thick RMSE, with intervals `[+0.054, +1.386]` and `[+0.561, +4.057]`.

Lower COT errors in frequent thin/clear regimes still make some all-regime or station aggregate scores look better, while the thick-COT objective selects transport. This is another measured regime trade-off, not a GHI result. The validation evidence suggests that simply supplying older maps does not fix the current residual head's thick-cloud suppression; it does not prove that temporal information is generally useless.

## Next decision

Do not launch another history-channel or same-head loss sweep from this result. Use the saved validation cases to group overlapping windows into distinct cloud events and separate three error modes: displacement/arrival-time error, thickness-amplitude evolution, and CPP retrieval mismatch. First verify whether transported COT puts each event at the correct station and time. A new thickness model is justified only if correctly located events still show systematic amplitude error. Keep test closed and require station thick-event improvement without a supported clear-sky regression before training a GHI head.

## Artifacts

- Protocol: `docs/SILI_FULL_VS_REPEAT_LAST_HISTORY_PROTOCOL_20260928.md`
- Trainer/launcher: `scripts/train_sili_history_repeat_control_20260928.py`, `scripts/run_sili_history_repeat_control_20260928.pbs`
- Figure renderer: `scripts/plot_sili_full_vs_repeat_last_history_20260928.py`
- Run outputs, both checkpoints, batch profile, full validation summary, epoch metrics, and day-bootstrap sufficient statistics: `results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/`
- Log marker: `SILI_FULL_VS_REPEAT_LAST_COMPLETE`; `test_used=false`; no GHI head was trained.
