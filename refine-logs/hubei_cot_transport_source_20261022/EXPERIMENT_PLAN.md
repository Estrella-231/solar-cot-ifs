# Hubei regional COT transport and evolution candidate

Status: source-coverage gate in progress; no new model training authorized before it passes.
Date opened: 2026-10-22
Scope: the independently trained Hubei three-station study only. Do not reuse Hunan weights, grids, CPP banks, or outcomes. Keep the current monthly train/validation split and keep all test/buffer payloads closed.

## Scientific question

Can a COT field with enough upstream spatial context, followed by causal cloud transport and learned growth/decay, produce more accurate future station-neighborhood COT than the current local SimVP-to-R chain and historical persistence? If it does, does that future COT improve a paired GHI forecast when the AGRI forecast, geometry, head, keys, labels, and training budget are held fixed?

This is a testable candidate explanation for the weak forecast-COT result. The evidence does not yet establish that the AGRI base-map extrapolation is the sole limitation. The attempt13 auxiliary loss failed its COT skill gate, and its AGRI guard had a slightly worse point estimate with an interval crossing zero; do not present either as proof of a physical mechanism.

## Frozen evidence and data contract

- Attempt13 forecast COT has 1,164 common CPP-reference init/target/lead keys across 16 BJT target dates. Equal-station/equal-lead RMSE differences are +0.739 COT units versus matched AGRI-MSE (95% date-block interval [-3.163, +4.476]) and +1.839 versus historical COT persistence ([-1.307, +3.779]); positive is worse. It passes neither comparator gate.
- Its validation AGRI guard has 407 exact common windows and a normalized full-field RMSE change of +1.068% for COT-aux versus MSE, with a 22-date paired interval [-0.220%, +2.241%]. This is not a stable degradation finding.
- Forecast inputs remain the corrected Hubei 13 AGRI channels plus the approved solar-geometry channels; history is 8 frames, output is +15 through +240 min in 16 steps. Use the Hubei 256x256 crop, Hubei grid and station coordinates. Never substitute Hunan geometry or the 16x16 station crop for the regional input.
- The existing corrected Hubei CPP sidecar contains station-local 16x16 targets only. Its extraction metadata lists 3,479 train/validation timestamps (3,178 train and 301 validation) and 16x16 valid-pixel counts; this does not establish a full 256x256 CPP bank.
- The extraction metadata names the original CPP root as `/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP`. On the currently reachable ZJNU login, that path and the older Data_Pool input path are not present. The configured FD-107/201/172/248 SSH probes currently close connections. A direct connection to 10.65.129.39 without its configured host alias failed public-key authentication; the configured zjnu-hpc alias does log in, but the original CPP path is absent there. This is an access/source-location blocker, not evidence that the original CPP data are absent. Existing Hunan regional caches are not valid Hubei substitutes.

## Staged work and gates

### G0 — Close attempt13 and freeze baselines

Completed by the linked attempt13 audit. Do not retune its COT auxiliary weight on the same validation dates. Preserve its checkpoints and negative result read-only.

### G1 — Restore and inventory Hubei regional CPP sources

Use only the existing train/validation timestamp manifest. Locate the original CPP files and record product/version, source filename, UTC timestamp, quality fields, COT/CLP array dimensions, latitude/longitude axes, fill values, and checksums. Verify every CPP timestamp against BJT = UTC + 8 h and map the 256x256 Hubei grid with the exact Hubei static coordinates. Do not interpolate, fill missing CPP as clear sky, or read test/buffer files.

Produce a per-date and per-frame inventory: source present, decodable, valid-pixel fraction over the 256x256 region, and coverage in the three station neighborhoods. Stop if the source cannot be read or time/grid/QC provenance cannot be reconstructed. The immediate next action is to recover a read-only route to this source and run this inventory; no GPU is needed.

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

The model work is potentially multi-hour. Once the regional-data gates pass, use a single PBS job requesting 4 A800s on node22 if available; node22 is the Zhejiang Normal University college server. Profile 1/2/4-worker throughput briefly first and select the fastest stable configuration. Previous attempt13 migration measured about 2.07x epoch-speedup on four A800s versus one, so four cards are a justified starting candidate, not an unconditional allocation. Keep global batch, total updates, LR schedule, and masked-loss numerator/denominator identical across worker counts. Preserve complete checkpoints before any resume or migration.

Immediately before and after every qsub, inspect the full slfu queue and keep at most three unfinished jobs across the workspace. Do not cancel other users' work. Query assigned GPUs and nvidia-smi inside the allocation; never infer that node22 GPUs are free from an empty account queue. Use Luna for routine monitoring after a task is actually running. Source extraction and manifest audits remain CPU tasks.

## Stop rules

- No accessible, full-region Hubei CPP source: remain at G1; do not train a regional R/E model using Hunan caches or a fabricated 16x16-to-256x256 field.
- G1/G2 fails time, grid, QC, or split checks: preserve the failure and fix the source contract before GPU use.
- Regional COT or transport candidate fails the COT gate: do not retrain GHI heads with that forecast-COT bank.
- COT improves but paired GHI does not: report COT skill separately; do not claim a GHI forecast gain or blame the backbone without a controlled attribution.
- Test and buffer remain closed throughout this plan.

## Current next step

Recover access to the original Hubei CPP source and complete G1. Until then, the useful work is provenance/interface review and cache-builder preparation; no GPU job should be submitted.
