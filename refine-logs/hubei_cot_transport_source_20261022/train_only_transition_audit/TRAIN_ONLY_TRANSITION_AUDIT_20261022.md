# Train-only station-center COT transition audit

**Scope:** Hubei 16×16 station-patch cache, training split only. This is a diagnostic of historical persistence against future CPP retrieval values, not a forecast-model score, independent COT truth, or GHI result.

## Result

The audit used 41,933 training init/lead pairs from 152 target dates (2025-08-01 to 2026-02-27), yielding 98,371 supported station-center records across Niushou, Wangzhai, and Mizhuang. The current COT value at initialization came from the cached frozen-R AGRI-to-COT bank; future targets are CPP retrieval COT. The training R bank is in-sample, and CPP is a retrieval reference, so both the source mismatch and retrieval uncertainty limit physical interpretation.

| Forecast lead group | Persistence RMSE (COT) | Persistence MAE | Mean target minus history COT |
|---|---:|---:|---:|
| +15 to +60 min | 9.885 | 5.054 | -2.010 |
| +75 to +120 min | 11.416 | 6.105 | -1.788 |
| +135 to +180 min | 12.664 | 6.937 | -1.525 |
| +195 to +240 min | 13.580 | 7.745 | -1.359 |

The persistence RMSE rises with lead at all three station pixels: Niushou 9.37→13.89, Wangzhai 9.61→14.30, and Mizhuang 9.75→13.97 from +15 to +240 minutes. The mean future CPP value is lower than the initialization-time R value by 1.705 COT units overall. This could reflect cloud evolution, CPP/R source differences, or both; these data cannot distinguish them.

Using a threshold of 30 COT on the single station pixel, transitions across all overlapping train pairs were:

| Threshold transition | Records | Fraction |
|---|---:|---:|
| Below 30 at history, ≥30 in future CPP (`thick_arrival_or_growth`) | 4,234 | 4.30% |
| ≥30 at history and future CPP (`thick_retained`) | 10,479 | 10.65% |
| ≥30 at history, below 30 in future CPP (`thick_decay`) | 5,852 | 5.95% |
| Below 30 at both | 77,806 | 79.09% |

`thick_arrival_or_growth` means only a threshold crossing at the station-center pixel. It does not separate a moving cloud arrival from local thickening. The forecast windows overlap heavily, so these record fractions are not independent-event frequencies.

Persistence errors are concentrated at higher initial R-COT values. In the history bin ≥30, equal-site persistence RMSE is 21.29 COT and the future CPP mean is 13.05 units below history on average. In the history bin <1, equal-site RMSE is 3.61. These patterns identify a difficult high-COT regime for a local persistence baseline, but do not prove that a learned transport model can correct it.

## Access and calculation scope

- Read split/key metadata from the bank and pair CSV indexes, then accessed numeric model/label arrays only at train init/target indexes. Validation model/label array values read: 0; test and buffer payload values read: 0.
- Support is the intersection of valid initialization-time history COT and valid future CPP COT at each station-center pixel. Leads are +15 through +240 minutes.
- Values were converted from cached normalized `log1p(COT)/log(101)` to physical COT in FP64, then RMSE/MAE/bias and threshold transitions were recomputed.
- Reproduce with `audit_train_cot_transitions.py`; the saved machine-readable output is `evidence/train_only_transition_audit_attempt1.json`.
- Local SHA-256: audit script `45303833823ACD3F2B516EECA5B7584F21B25586703EF5A1806A97EFA8D655BB`; result JSON `A7B91CAA93716D9323C3D776A4054C251C066088006ED7C2A7D9DA9AC6592517`.

## Consequence for the next step

This train-only audit supports continued investigation of high-COT persistence errors, but it does not reopen training on the exposed monthly validation cohort. The separate 16×16 learned COT candidates still fail their predeclared skill gate, so no paired GHI head is justified. Full-region work remains gated on recovering readable Hubei CPP source files and passing their time/grid/QC inventory. If that source becomes available, profile the new regional workload on 1/2/4 GPUs and use the fastest stable allocation while preserving the global batch and update contract. The current audit was CPU-only and did not use GPUs.
