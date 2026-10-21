# Hubei regional COT transport and evolution candidate

Status: full-region source gate remains blocked. The separate 16x16 station-patch pilot completed seed-42 training and monthly validation. Its point RMSE beats historical persistence overall, but the date-block interval crosses zero, the predeclared COT gate is not passed, and retrieved-thick cases regress substantially. Do not proceed to paired GHI training from this bank. See `station_patch_pilot_v1/RESULT_REPORT_20261022.md`.
Date opened: 2026-10-22
Scope: the independently trained Hubei three-station study only. Do not reuse Hunan weights, grids, CPP banks, or outcomes. Keep the current monthly train/validation split and keep all test/buffer payloads closed.

## Scientific question

Can a COT field with enough upstream spatial context, followed by causal cloud transport and learned growth/decay, produce more accurate future station-neighborhood COT than the current local SimVP-to-R chain and historical persistence? If it does, does that future COT improve a paired GHI forecast when the AGRI forecast, geometry, head, keys, labels, and training budget are held fixed?

This is a testable candidate explanation for the weak forecast-COT result. The evidence does not yet establish that the AGRI base-map extrapolation is the sole limitation. The attempt13 auxiliary loss failed its COT skill gate, and its AGRI guard had a slightly worse point estimate with an interval crossing zero; do not present either as proof of a physical mechanism.

## Frozen evidence and data contract

- Attempt13 forecast COT has 1,164 common CPP-reference init/target/lead keys across 16 BJT target dates. Equal-station/equal-lead RMSE differences are +0.739 COT units versus matched AGRI-MSE (95% date-block interval [-3.163, +4.476]) and +1.839 versus historical COT persistence ([-1.307, +3.779]); positive is worse. It passes neither comparator gate.
- Its validation AGRI guard has 407 exact common windows and a normalized full-field RMSE change of +1.068% for COT-aux versus MSE, with a 22-date paired interval [-0.220%, +2.241%]. This is not a stable degradation finding.
- Forecast inputs remain the corrected Hubei 13 AGRI channels plus the approved solar-geometry channels; history is 8 frames, output is +15 through +240 min in 16 steps. Use the Hubei 256x256 crop, Hubei grid and station coordinates. Never substitute Hunan geometry or the 16x16 station crop for the regional input.
- Historical source evidence is stronger than a missing-path check: on 2026-10-20, the FD-201 pipeline read the original CPP NetCDF root `/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP` and extracted all 3,479 corrected-UTC train/validation keys (3,178 train, 301 val) into the existing station-local 16x16 sidecar. This proves those files were readable then, but the exported bank is not a regional 256x256 cache.
- On 2026-10-22, the currently reachable ZJNU login `zjnu-hpc` (`tc6000`) did not contain the original `/home/Data_Pool_3` source roots; the corrected sidecar and train/validation AGRI staging are present. The current desktop SSH configuration does not resolve an `fd201` alias, so the historical FD-201 route is not available from this shell. This is a route/transfer blocker, not evidence that CPP data are absent. Hunan regional caches are not valid Hubei substitutes.
- A train/validation-only 22,061-row Hubei AGRI key index is now local (20,206 train, 1,855 val; SHA-256 `8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183`). Candidate exact-cadence 8-history-plus-16-future windows before the last known CPP BJT time (2026-03-02 18:00) number 9,673 train and 1,022 validation, but these are AGRI candidates only; CPP file/QC/grid support has not been checked for those keys.
- The existing 3,479-row sidecar itself yields only 764 full-support overlapping train windows and 59 validation windows; the validation windows begin on only three init dates (15 hourly-stride and four non-overlapping 6-hour windows). This is too sparse for a defensible four-hour model-selection result. The exact audit is `CPP_LOCAL_SEQUENCE_SUPPORT_AUDIT.md`; no training was done from this feasibility check.

## Staged work and gates

### G0 — Close attempt13 and freeze baselines

Completed by the linked attempt13 audit. Do not retune its COT auxiliary weight on the same validation dates. Preserve its checkpoints and negative result read-only.

### G1 — Restore and inventory Hubei regional CPP sources

Use only the existing train/validation timestamp manifest. Locate the original CPP files and record product/version, source filename, UTC timestamp, quality fields, COT/CLP array dimensions, latitude/longitude axes, fill values, and checksums. Verify every CPP timestamp against BJT = UTC + 8 h and map the 256x256 Hubei grid with the exact Hubei static coordinates. Do not interpolate, fill missing CPP as clear sky, or read test/buffer files.

Produce a per-date and per-frame inventory: source present, decodable, valid-pixel fraction over the 256x256 region, and coverage in the three station neighborhoods. Stop if the source cannot be read or time/grid/QC provenance cannot be reconstructed. The immediate next action is to recover a read-only route to this source and run this inventory; no GPU is needed.

Use the new AGRI train/validation key index as the candidate list, not only the older 3,479-row R manifest. First compute exact CPP path existence after BJT-to-UTC conversion; then inspect quality masks and grid mapping on all eligible sequence frames. The 9,673/1,022 AGRI sequence counts above are only an upper bound until this raw-source audit passes. A separate station-patch forecaster is not to be trained on the current 59-window/three-date validation support.

### G1S — Bounded station-neighborhood COT forecasting pilot

The full-region G1 route remains blocked on the FD-201 source route. To make progress without relabeling the existing 16x16 product as regional data, run a separately named station-neighborhood pilot using the frozen, corrected-UTC Hubei R-COT bank and CPP retrieval sidecar. Its spatial scope is exactly the 16x16 patch containing Niushou, Wangzhai, and Mizhuang; it cannot establish full-region COT forecasting skill.

The metadata-only join audit `COT_BANK_CPP_OVERLAP_AUDIT.md` passes: 22,061 train/validation R-bank rows match the AGRI index exactly; all 3,479 CPP sidecar rows match a bank key and split; and 47,932 train / 4,692 validation init-lead pairs have eight causal history frames and a full-support CPP manifest row. Validation covers 16 target dates and has 277-298 candidate pairs per lead before additional daylight/solar masks are applied. These are overlapping forecast cases, not independent weather events, and CPP remains a retrieval reference.

Build a train/validation-only cache from this bank. The history input is eight actual-AGRI-to-frozen-R COT maps, their validity masks, and three deterministic geometry fields; known future geometry may be used, but future AGRI, CPP, COT labels, GHI, and target masks are never model inputs. Reuse `RetrievedCOTForecaster` with random initialization and partial per-lead CPP supervision. Mark R outputs above its [0,100] training target range invalid rather than clipping. The current R train bank is not OOF; preserve this provenance, count affected history pixels, and interpret the work as a validation pilot. Do not call the training-input R predictions independent of CPP training labels.

The cache audit is a training gate: after physical masks, validation must retain at least 12 distinct target dates and at least 200 init/target pairs at each lead; otherwise stop and report the reduced support. Select stopping/checkpoint only through a date-grouped internal holdout carved from train, with a one-day purge around held-out dates. Freeze that rule before fitting; do not use the monthly validation dates to select architecture, loss weights, or epoch. Score once on common supported validation pixels against COT persistence and the forecast-AGRI-to-R result only if its R/source lineage can be matched exactly; otherwise identify it as a nonmatched contextual comparator. Report physical-unit RMSE/MAE/bias, date-block paired intervals, per-lead and per-site results, and thick-cloud behavior. Any later GHI application must use the same forecast AGRI, geometry, sample keys, head, labels, and training schedule in paired no-COT/COT arms.

This G1S route is an exploratory station-patch branch. It does not release G1/G2/G3 full-region work or independent-test claims. Its monthly validation result is complete: the overall point estimate beats persistence, but uncertainty includes no improvement and the CPP-retrieved thick stratum is worse. The forecast-AGRI-to-R comparator was not included in the saved scoring report. The GHI gate is therefore NOT PASSED; do not start paired GHI training or expand seeds.

The same-R context comparator is now complete on the exact 1,824-key common intersection, documented in `same_r_comparator_audit/SAME_R_CONTEXT_DIAGNOSTIC_20261022.md`. It uses the attempt12 MSE-selected SimVP validation AGRI bank passed through the exact G1S R, not the matched attempt13 MSE-control checkpoint. The future-COT RMSE point estimate is lower than both persistence and this contextual comparator, but both date-block intervals cross zero, MAE does not beat persistence, and CPP-retrieved thick-cloud RMSE materially regresses. Therefore this closes the requested contextual comparison but does **not** pass the G1S COT gate. Do not train paired GHI heads or expand seeds.

A separately named persistence-residual patch model (`persistence_residual_attempt13`) finished on 2026-10-22. Its 3,220-key full validation RMSE is 11.991 versus persistence 15.379 (22.03% lower point estimate), but the paired date-block 95% interval for forecast-minus-persistence is [-7.165,+1.060], and CPP-thick RMSE remains worse (19.784 versus 17.566). On the existing 1,824-key subset it improves the previous learned model from RMSE 13.871 to 12.947, but the paired interval crosses zero. The attempt is exploratory only; it does not pass the COT gate and provides no GHI result. Full metrics, code, checkpoints, logs and hashes are recorded in `persistence_residual_attempt13/RESULT_REPORT_20261022.md` and its `evidence/` directory. For this small patch, a fixed-global-batch profile measured 1/2/4 A800 throughput at 2,041.58/1,845.11/1,572.77 samples/s, so one GPU was fastest. Re-profile larger regional models when the raw CPP source gate is restored; do not infer that multi-GPU is generally slower.

### G2 — Build one immutable regional train/validation cache

After G1 passes, extract each unique allowed AGRI/CPP timestamp once. Preserve original CPP masks and retrieval metadata. Match the corrected full-region AGRI and solar geometry by exact timestamp. Store train and validation separately, fit normalization on train only, and publish a manifest with row keys, missing-source list, coverage, and hashes. Transfer once to node-local storage before training; do not repeatedly stream the bank from a shared drive.

### G3 — Train and evaluate a Hubei regional retrieval model

Train a new Hubei-only R_region from random initialization on supported train frames; do not widen the old 16x16 R by pretending its training support is regional. Freeze architecture, normalization, split, and stopping rule before fitting. Validate on the same timestamps and CPP valid pixels. Report physical-unit COT MAE/RMSE/bias and support at full-region, each station pixel, 3x3, and 16x16 views; CPP remains a retrieval reference. If source support is too sparse or station/thickness strata are absent, stop and report the limitation rather than balancing by selecting favorable dates.

### G4 — Compare future-COT routes before any GHI-head retraining

Using one frozen R_region and identical keys/common CPP support, compare:

1. Historical COT persistence.
2. Causal advection of the last historical COT field, with motion estimated only from AGRI history.
3. The current forecast-AGRI-to-R route using the new regional R.
4. A transport-plus-source candidate: advect in log1p(COT), then predict a bounded spatial residual for cloud growth/decay from historical COT, AGRI history, solar geometry, lead, and valid-source masks.

Use identical target times and missing-pixel rules. Freeze motion method on train dates only. Evaluate by station, lead, continuous 1-hour horizons and full 0-4 h, physical COT error, bias, cloud arrival time, thick-cloud detection, and date-block paired intervals. No future AGRI or CPP may be model input. CPP and R(AGRI) outputs are reference targets/diagnostics, not independent physical truth.

The E candidate proceeds to the GHI test only if it beats both persistence and forecast-AGRI-to-R on the predeclared equal-station/equal-lead COT score, the date-block interval supports improvement, and no station shows a material consistent regression. Do not select checkpoints or tune on the GHI results.

### G5 — Paired GHI application

Only after G4 passes, generate causal train/validation forecast-COT banks. Compare a no-COT GHI head and a future-COT GHI head with the same frozen AGRI forecast, geometry, row keys, labels, H architecture, initialization per seed, sample order, global batch, optimizer, update count, and selection rule. All three Hubei stations remain included. Save every validation prediction and independently rescore in FP64 with equal-station summaries and BJT target-date paired uncertainty. If gains are absent or any site regresses consistently, stop expansion and report it.

## Multi-GPU execution policy

For long GPU jobs, first inspect the live queue and allocated-node availability, then profile 2/4 GPUs early (and include 1 GPU as a speed baseline when practical). Use the fastest stable configuration; multi-GPU is preferred when it improves wall-clock throughput while preserving global batch, optimizer updates, LR schedule, and masked-loss numerator/denominator. Node22 is the Zhejiang Normal University college server. For this particular 16x16 station-patch pilot, identical global-batch=32 profiles measured 1 GPU at 2,041.58 samples/s, 2 GPUs at 1,845.11, and 4 GPUs at 1,572.77, so one GPU was the fastest measured choice; this result is model-specific, not a general claim against multi-GPU. Preserve complete checkpoints before any resume or migration.

Immediately before and after every qsub, inspect the full slfu queue and keep at most three unfinished jobs across the workspace. Do not cancel other users' work. Query assigned GPUs and nvidia-smi inside the allocation; never infer that node22 GPUs are free from an empty account queue. Use Luna for routine monitoring after a task is actually running. Source extraction and manifest audits remain CPU tasks.

## Stop rules

- No accessible, full-region Hubei CPP source: remain at G1; do not train a regional R/E model using Hunan caches or a fabricated 16x16-to-256x256 field.
- G1/G2 fails time, grid, QC, or split checks: preserve the failure and fix the source contract before GPU use.
- Regional COT or transport candidate fails the COT gate: do not retrain GHI heads with that forecast-COT bank.
- COT improves but paired GHI does not: report COT skill separately; do not claim a GHI forecast gain or blame the backbone without a controlled attribution.
- Test and buffer remain closed throughout this plan.

## Current next step

1. G1S training, hardware profile, validation metrics, provenance, and GHI no-go are recorded in `station_patch_pilot_v1/RESULT_REPORT_20261022.md` and the tracker; the same-R context follow-up is recorded separately.
2. The local sidecar/interface audit is complete: `CPP_SOURCE_INTERFACE_AUDIT_20261022.md` records the retained fields and the minimum regional inventory. Fresh probes using configured aliases `FD-201` and fallback `FD-107` both had their SSH connections closed before any command ran. This remains a route blocker, not evidence CPP is absent. Retry the read-only route later; do not substitute Hunan products.
3. No new R/E training or GHI retraining is justified while the source gate and G1S COT gate are closed. Once a session is available, perform the frozen train/validation inventory before any regional cache or training. Do not fabricate regional fields or cloud-height corrections.
