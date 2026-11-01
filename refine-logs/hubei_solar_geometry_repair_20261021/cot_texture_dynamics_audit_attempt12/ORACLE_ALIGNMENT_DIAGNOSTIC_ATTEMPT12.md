# Attempt12 COT spatial-alignment upper-bound diagnostic

This is a validation-only attribution diagnostic, not an operational score. The reference is the same frozen R applied to observed future AGRI; that future image is unavailable at initialization. CPP only defines the common valid-pixel mask and is not independent COT truth.

Matched init-target-lead pairs: 1164; target BJT dates: 16; test/buffer reads: 0/0.
Before this diagnostic, an independent full-patch re-score reproduced all 48 archived forecast/persistence/offline-future RMSE and support-count rows (16 exact leads × 3 methods). The R-output array lacks a source-manifest digest; its local SHA and this score-reproduction check are recorded in the JSON receipt.
All methods use the same CPP-valid central 12x12 pixels; shifts are bounded by ±2 pixels per axis and no wraparound is used.
The per-case shift/scale sees the future R proxy, so it is an unattainable oracle. It only indicates whether position or magnitude correction could plausibly recover error.

## Pooled validation-proxy results

| Method | RMSE vs future R proxy | Paired date-block 95% CI of RMSE difference vs forecast |
|---|---:|---:|
| Forecast COT | 5.6240 | reference |
| History persistence | 5.9754 | [-0.6800, +1.4260] |
| Oracle spatial shift | 5.1538 | [-0.6572, -0.2398] |
| Oracle amplitude scale | 2.3670 | [-4.8450, -1.5113] |
| Oracle shift + scale | 2.1822 | [-5.0491, -1.6926] |

## By exact lead

See `oracle_alignment_by_lead_attempt12.csv` for exact +15…+240 min rows and date-cluster intervals; `oracle_alignment_pairs_attempt12.csv` contains each paired case, its selected oracle shift/scale, and per-case errors.

## Claim boundary

A lower oracle-shift error would show that spatial registration has recoverable headroom; a lower oracle-scale error would show amplitude-calibration headroom. Neither proves a deployable method. The future AGRI→R target and CPP mask are retrieval-chain references, not independent physical COT truth. A new learned transport/source-sink model still requires train-only development, untouched validation selection rules, and a fresh paired evaluation against forecast and persistence.

Reproduce with `python oracle_alignment_attempt12.py`. Input SHA-256 values, exact target dates, masks, bootstrap settings, and the pair-level output hash are saved in `oracle_alignment_attempt12.json`.
