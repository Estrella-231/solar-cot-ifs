# Sili transport plus thickness pilot — validation result

## Question and data boundary

This pilot asks whether a learned COT residual can add cloud-thickness change to history-derived motion and the frozen SimVP→R COT forecast. It uses a fixed 512-sequence training and 512-sequence validation bank, with 16 forecast leads from +15 to +240 minutes. The selected cohort follows the existing GHI cohort manifest and is not filtered by future CPP availability. All results below use the validation split. `test_used=false` is confirmed by both bank and run manifests.

CPP COT is the retrieval reference used as a training and validation label; it is not independent in-situ cloud truth. Adjacent 15-minute windows overlap, so event and pixel-lead counts are descriptive and not independent samples. The results do not establish GHI benefit.

Inputs to the learned head are the frozen forecast COT, local32 subpixel motion estimated from seven historical C13 pairs, transported final-history R COT, the final two historical R-COT frames as level and trend, and lead. Although the bank contains eight historical COT frames, the head does not encode the full eight-frame COT sequence. Call this the **two-frame-trend residual pilot**.

## Training and selection

The pilot completed 9 epochs and stopped after eight consecutive epochs without a new validation thick-pixel RMSE minimum. The saved checkpoint is epoch 1, selected by the lowest validation RMSE over all thick CPP pixels in the local 64×64 crop. This criterion is separate from the station-pixel audit below. The frozen COT checkpoint SHA-256 is `27B91971376F7582B10BB834234B748C4C718448920E22DACC41DDA1E88EB2B0`.

The selection criterion found no epoch that beat the pure-transport baseline on the thick-pixel metric. The final audit therefore retains transport as the reference method; the learned checkpoint is an exploratory result only.

## What the validation shows

On the full local 64×64 field, learned residuals reduce overall error and clear-reference false clouds, but harm thick-cloud scores:

| Method | All-pixel MAE | All-pixel RMSE | Thick-pixel RMSE | Thick misses (prediction <10; CPP ≥30) | Clear false ≥10 rate | COT≥30 IoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen SimVP→R | 5.537 | 9.666 | 27.700 | 387,738 | 4.38% | 0.1806 |
| Persistence | 5.288 | 9.609 | 27.078 | 346,485 | 5.67% | 0.2259 |
| C13 transport | 5.287 | 9.520 | **26.504** | **334,790** | 5.81% | **0.2411** |
| Transport + residual | **4.900** | **9.056** | 28.279 | 371,192 | **3.11%** | 0.1664 |

At the Sili station pixel, there are 5,664 valid sequence-lead records, including 204 thick CPP records and 3,530 clear CPP records. The station result has the same trade-off:

| Method | Station MAE | Station RMSE | Thick RMSE | Thick misses / 204 | Clear false ≥10 rate | COT≥30 IoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen SimVP→R | 4.772 | 8.604 | 28.363 | 67 / 32.8% | 4.90% | 0.0990 |
| Persistence | 4.543 | 8.458 | 26.921 | 55 / 27.0% | 4.67% | 0.1838 |
| C13 transport | 4.686 | 8.738 | **25.163** | **47 / 23.0%** | 4.96% | **0.1978** |
| Transport + residual | **4.083** | **8.049** | 28.816 | 64 / 31.4% | **3.60%** | 0.0866 |

Relative to transport, the learned head lowers full-field MAE/RMSE by 7.3%/4.9%, but raises thick-pixel RMSE by 6.7% and thick misses by 10.9%; clear false alarms fall 46.6% and COT≥30 IoU falls 31.0%. At the station it lowers MAE/RMSE by 12.9%/7.9%, but raises thick RMSE by 14.5%, thick misses by 36.2%, and lowers COT≥30 IoU by 56.2%; clear false alarms fall 27.4%. Its mean station bias is −0.52 COT versus +0.30 for transport, consistent with a correction that often pulls the predicted field downward.

Among 54 overlapping validation windows whose station CPP peak reaches 30, the learned prediction averages COT 18.7 at the CPP peak; transport averages 21.9. The predicted peak occurs 26.7 minutes early on average for the learned head and 24.4 minutes late for transport. These window-level means are descriptive because the windows overlap.

Lead-wise station scores also cross. At +240 minutes, learned RMSE is 9.25 versus 11.13 for transport, while thick RMSE is 33.09 versus 30.85 and the thick-miss rate is 5/13 for both. At +60 minutes, learned thick RMSE is 31.98 versus transport 21.51, and learned misses 2/10 while transport misses 0/10. There is no consistent thick-cloud advantage.

The event figure makes the intensity failure visible. On 2025-08-05 the CPP station series rises to 80.2 COT while the learned series stays much lower. On 2025-07-09 the CPP retrieval peaks at 68.7 and decays; the learned head tracks the decay but remains low. For the persistent thick case on 2025-08-14, it likewise fails to reach the reference peak. The clear-reference case is included as a counterexample for false cloud output. These are validation examples selected by reference-pattern rules, not independent observations.

## Decision and next action

Do not use this checkpoint to rebuild GHI training inputs or claim that the residual improves cloud thickness. The current gain in average COT error comes with worse thick-cloud intensity and overlap. The learned head appears to suppress false clouds while also shrinking valid thick cloud peaks.

The next step is a targeted loss and residual audit before changing the architecture: compare training versus validation thick-pixel residuals by lead and CPP magnitude, quantify the clear-penalty contribution and residual sign/magnitude, and verify that station-pixel weighting matches the scientific target. Then define an acceptance gate that requires beating pure transport on thick peak error/misses without worsening clear false alarms. Keep test closed. Only a candidate that passes that validation gate should justify a separately named full-history COT encoder or a downstream GHI retraining.

## Reproducibility files

- Station summary and per-lead/peak tables are alongside this report in `sili_station_val_audit_seed42_20260928_v2/`.
- Figures: `station_event_tracks.pdf` and `station_lead_metrics.pdf` (PNG previews are included).
- Per-sample track arrays and CSV remain on the compute server at `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925/sili_station_val_audit_seed42_20260928_v2`; they are not part of this Git commit.
- PBS station audit `210977.tc6000` completed with `exit_status=0` and `SILI_STATION_VAL_AUDIT_COMPLETE`.
- Audit jobs `210974` and `210975` exposed validation date-map and peak-mask indexing bugs before producing results. `210976` exited without entering user code, with no user-code error in its isolated log. Corrected audit job `210977` completed with `exit_status=0` and a separate log. These audit-script corrections did not modify or retrain the pilot model.
