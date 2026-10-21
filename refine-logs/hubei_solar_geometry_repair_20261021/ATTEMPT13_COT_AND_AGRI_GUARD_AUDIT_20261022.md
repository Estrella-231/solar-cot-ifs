# Attempt13 COT skill and AGRI guard audit

Date: 2026-10-22
Scope: Hubei train/validation only; independent validation audit, not an independent test.

## Result

The attempt13 COT-aux arm does not pass the predeclared COT skill gate. On the common CPP-reference support (1,164 init/target/lead keys across 16 BJT target dates), equal-station/equal-lead COT RMSE is higher for COT-aux than for the matched AGRI-MSE model by 0.739 COT units (5,000-replicate date-block bootstrap 95% interval: -3.163 to +4.476). It is also higher than historical COT persistence by 1.839 units (95% interval: -1.307 to +3.779). Both intervals cross zero; all three station point estimates are worse than both comparators. CPP is a retrieval reference, not independent physical truth. Do not tune the same validation cohort or route this forecast-COT output into a new GHI-head claim.

The independent AGRI guard uses the same 407 validation windows, exact row keys/order, targets, future geometry, and normalization for the two arms. Full-field normalized-space RMSE is 0.490145 for AGRI-MSE and 0.495381 for COT-aux, a paired difference of +0.005235 (+1.068% relative; positive means worse). The paired BJT initialization-date interval over 22 dates is [-0.000971, +0.011075], or [-0.220%, +2.241%] relative, so the point estimate is slightly worse but the interval does not establish a stable AGRI degradation.

The physical AGRI table keeps each of the 13 channels and 16 leads separate. It contains 208 channel/lead comparisons and is exploratory; no multiplicity-adjusted claim is made. The audit also emitted a pooled local 3x3 summary that mixes channels with different physical scales. That pooled number is invalid and is intentionally excluded from interpretation.

## Audit scope and limits

The independent FP64 audit re-aggregated saved per-sequence full-field error sufficient statistics and checked saved local-patch arrays and exact bank alignment. Full-field raw prediction arrays were not saved, so this is not an independent re-inference/rescoring of raw full-field predictions. Test and buffer payload reads are both zero. These results are validation diagnostics and cannot be promoted to independent-test or deployment claims.

The paired COT scoring receipt records 1,164 common keys, 16 target dates, and zero test/buffer reads. Its sign convention is auxiliary minus comparator, so positive RMSE differences mean worse COT forecasts.

## Reproduction artifacts

All paths below are relative to this directory:

- `attempt13_agri_guard_datefix_audit_v3/audit_attempt13_agri_guard_fp64_v5.py`
- `attempt13_agri_guard_datefix_audit_v3/server_outputs/agri_guard_fp64.json`
- `attempt13_agri_guard_datefix_audit_v3/server_outputs/forecast_cot_summary_fp64.json`
- `attempt13_agri_guard_datefix_audit_v3/server_outputs/agri_paired_delta_by_lead.csv`
- `attempt13_agri_guard_datefix_audit_v3/server_outputs/full_field_agri_paired_delta_by_channel_lead.csv`

Audit PBS: `215510.tc6000`, normal queue, CPU-only on node4, exit status 0. The final account-wide `qstat -u slfu` check returned no listed jobs. No GPU training was submitted for this audit.

SHA-256:

- Audit script: `A342FA7BC58A4E13400060887ABB8EE4518FEB18E2D6E803745B171BA9F4E894`
- AGRI result JSON: `C41CD6449B86057C98AC599B8523273E22B61315941FB024684E93E438AA0D05`
- COT score summary: `59526FCC73D245EED6CD58964DE46A7166E2E7839F468FF96AA4128A89FCBE76`
- AGRI per-lead delta CSV: `492C4F8F4BE1055EEE077481AC150621BCEBC7A881BC208922E44FD525A230EA`
- AGRI full-field channel/lead delta CSV: `B0425E346723ED87C09A3E610C44A211A113F5DEDC118F6EA4C1F99EF237A6D0`

## Decision

Attempt13 is closed as a validation COT-skill failure: it did not beat either matched AGRI-MSE or historical persistence. The evidence does not prove that COT is useless, nor that the AGRI extrapolator is the sole cause. The next candidate must first establish a Hubei regional COT source and input-support contract; the existing 16x16 station R product cannot stand in for a regional upstream cloud field.
