# G1S station-patch COT forecast pilot — result record

**State:** training and monthly validation complete; predeclared COT gate not passed; paired GHI training not started.

**Scope:** the shared Hubei 16×16 station-neighborhood patch around Niushou, Wangzhai, and Mizhuang. This is not the full 256×256 Hubei region and is not an independent-test result.

## Question and input path

This pilot tested whether a learned time-evolution model can forecast station-center COT from eight historical AGRI-to-frozen-R COT maps. The model received those historical COT maps, their validity masks, and deterministic solar-geometry fields; known future geometry was available. Future AGRI, CPP, COT labels, and GHI were not model inputs. The targets are CPP-retrieved COT and must be interpreted as a satellite retrieval reference, not independent physical truth. The R train predictions used to build the history bank are in-domain, not OOF.

The forecast covers 16 steps from +15 to +240 minutes. Seed42 was randomly initialized. Checkpoint selection used a contiguous date-grouped holdout from train dates with a one-day purge; monthly validation was not used to choose the epoch. There were 5,793 fit windows and 1,338 inner-holdout windows. The selected epoch was 2 at global step 362 (internal COT RMSE 9.137, compared with 10.424 at epoch1); training stopped after epoch10 at global step1,810 under patience8. The selected checkpoint is `evidence/gpu_training_attempt9_seed42/best.pt`.

## GPU profile and choice

All profile arms used global batch32 and the same cache, with five warmup steps followed by 30 timed updates. Throughput was 2,041.58 samples/s on one A800, 1,845.11 on two, and 1,572.77 on four. Two and four GPUs were respectively 9.6% and 23.0% slower than one GPU. The model is small and the measured DDP communication/scheduling overhead outweighed parallel compute, so the fastest stable configuration—one A800 on Zhejiang Normal University node22, physical GPU0—was used for the full run. This choice follows measured wall-clock throughput; it does not imply multi-GPU is generally slower.

## Monthly validation result

The score uses the identical station-center pixels for the learned forecast and persistence: CPP target valid and initialization-time historical COT valid. It includes 3,220 init/lead pairs over 16 BJT target dates and 9,644 valid station-center records. The windows overlap in time, so these records are not independent weather events.

| Validation view | Learned COT forecast | Historical COT persistence | Interpretation |
|---|---:|---:|---|
| Equal-station/equal-lead RMSE | 13.368 | 15.379 | Point estimate favors the learned model by 2.012 units (13.08%) |
| 95% date-block interval for forecast minus persistence RMSE | — | — | [−7.282, +3.052]; crosses zero, so improvement is not established |
| Niushou RMSE | 13.191 | 14.848 | Point estimate favors forecast |
| Wangzhai RMSE | 13.781 | 15.617 | Point estimate favors forecast |
| Mizhuang RMSE | 13.130 | 15.672 | Point estimate favors forecast |

The aggregate gain is not uniform across COT thickness. Thickness groups are defined using CPP retrieval values:

| CPP-retrieved target group | Forecast RMSE | Persistence RMSE | Main behavior |
|---|---:|---:|---|
| Near-zero, COT < 1 | 8.897 | 15.301 | Forecast lower |
| Thin, 1–10 | 4.828 | 6.810 | Forecast lower |
| Moderate, 10–30 | 10.811 | 17.878 | Forecast lower |
| Thick, COT ≥ 30 | 23.053 | 17.566 | Forecast higher; forecast bias −21.781 vs persistence +7.257 |

For the CPP-retrieved thick threshold, forecast POD/FAR/CSI are 0.310/0.423/0.253, while persistence is 0.873/0.433/0.524. The clearest failure is therefore missed or attenuated thick retrievals, even though the overall point RMSE is lower. These are validation diagnostics against CPP, not independent truth.

At the time this pilot score was produced, no exact same-R forecast-AGRI comparator was included. A subsequent validation-only context comparison is now complete on the 1,824-key common intersection; see `../same_r_comparator_audit/SAME_R_CONTEXT_DIAGNOSTIC_20261022.md`. It passes attempt12 MSE-selected SimVP predicted AGRI through the exact frozen G1S R. Future-COT / persistence / attempt12 AGRI→R equal-station/equal-lead RMSE is 13.871 / 16.097 / 16.567. The paired date-block intervals for the apparent future-COT improvements cross zero, and the comparator is contextual rather than the matched attempt13 MSE arm. On CPP-retrieved COT≥30 cases, future-COT RMSE is 23.415 versus 18.551 for persistence and 11.420 for AGRI→same-R. The follow-up therefore does not pass the COT gate and provides no GHI evidence.

## Decision

The full-support point estimate beats persistence, but its date-block interval includes no gain and the CPP-retrieved thick group regresses. The later same-R contextual comparison also has intervals crossing zero and shows a large thick-group regression; it does not replace the original full-support score or become an attempt13 paired SimVP comparison. Therefore the predeclared G1S gate is **NOT PASSED**. Do not train paired GHI heads, expand seeds, or claim that COT improves GHI forecasting from this pilot. Full-region work separately remains blocked on restoring the read-only FD-201 CPP-source route; current route details and the source-field requirements are in `../CPP_SOURCE_INTERFACE_AUDIT_20261022.md`.

## Reproduction and provenance

- Local evidence bundle: `evidence/gpu_training_attempt9_seed42/` (checkpoint, validation predictions, epoch log, completion receipt, score JSON, and 1/2/4-GPU profile JSONs).
- Local evaluator: `evaluate_station_patch_cot_validation.py`; input arrays and keys are checked before scoring, predictions are converted from scaled log-COT into physical units, and date-block intervals use 5,000 paired resamples.
- Checkpoint SHA-256: `d144e1c917366c0f9244619dc3dd096b5ba0c18634cd88f5cfbc9d36ab555282`.
- Validation-prediction SHA-256: `d45438e53ad3b50192485b338e81d9c856d2f83f428963bbc1db221ea573e845`.
- Validation-score SHA-256: `68881e16849c84c5acce11044f19100d5ee7d8a7d2462ec17abc41735f40e54a`.
- Selected worker receipt SHA-256: `e3cc18d4607de5fcf4120be2f404ca09ba243023ce4ce04c248fb633e293e72a`.
- Frozen R checkpoint SHA-256: `8ad4dbd1328325b778308ca4f8a6398dc476043ad41230ed094b9ae4855877c9`.
- Test and buffer payload reads: 0 and 0.
