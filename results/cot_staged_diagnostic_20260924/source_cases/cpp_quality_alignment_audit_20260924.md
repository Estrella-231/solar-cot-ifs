# CPP quality and alignment audit (2026-09-24)

## Four false-cloud cases: raw CPP check

Each extracted raw NetCDF crop was compared pixel-for-pixel with the corresponding aligned sidecar. All four COT crops matched their sidecars exactly, and all 16×16 latitude/longitude grids matched exactly. At the four reported station pixels, source CPP has COT=0 and CLP=0 (clear); there are no zero-COT pixels classified as water/ice in these 16×16 crops. The crops have no missing CLP entries. This supports “valid clear-class zero” for these four examples, not a general guarantee that every historical zero is valid.

| Station / UTC | pixel (r,c) | COT | CLP | CPP grid lat,lon | zero COT count / clear |
|---|---:|---:|---:|---|---:|
| zhujia 2025-07-06T08:15:00Z | (0, 9) | 0.000 | 0 (clear) | 29.8800, 112.4800 | 234 / 234 |
| sili 2025-07-14T05:15:00Z | (0, 9) | 0.000 | 0 (clear) | 26.2800, 112.4000 | 1 / 1 |
| zhujia 2025-07-06T06:30:00Z | (8, 11) | 0.000 | 0 (clear) | 29.5600, 112.5600 | 207 / 207 |
| sili 2025-07-07T05:15:00Z | (0, 6) | 0.000 | 0 (clear) | 26.2800, 112.2800 | 90 / 90 |

The supplied CPP V2 Portable archive was inspected (SHA-256 `dc7959b7bc7ef467239a261072f8789d416df9e9a7bf19c8d16403e3d113eb7b`). Its `Code/Quality_Control.py` defines CLP=0 as clear and treats zero COT as valid in clear pixels; its newer `Retrieval_CPP.py` writes a QA field whose 1 means previous-hour replacement. The four historical 2025 NetCDF files inspected here contain no QA field, so that newer flag contract cannot be retroactively used to certify their lineage.

## Full validation-pack zero accounting

The validation pack has 1,664,013 research-mask-valid pixels. Of these, 584,013 (35.10%) are stored as valid zero-COT targets. Another 9,203 zeros are merely stored placeholders outside the mask and are not supervised. By station: Sili 186,798/825,721 (22.62%); Zhujia 397,215/838,292 (47.38%). This is full-pack label accounting under the research mask; it does not establish official product QA for every validation time.

## Full validation-set raw CPP source audit

All 6,536 validation station windows across 3,290 timestamped CPP NetCDF files were read. Twelve files initially returned HDF read errors under the parallel scan; all twelve succeeded on sequential retry, so the final unreadable count is zero. All 6,536 COT/CLP windows were finite, and all 6,536 station coordinate windows exactly matched the corresponding corrected sidecar grid. Under the pack research mask, raw CPP COT matched the saved training label exactly for every valid pixel (zero mismatches).

| Check | Result |
|---|---:|
| COT=0 pixels in the full station-window collection | 587,328 |
| COT=0 with CLP=clear | 587,328 (100%) |
| COT=0 with CLP=water / ice / missing | 0 / 0 / 0 |
| Mask-valid zero-COT labels | 584,013 / 1,664,013 (35.10%) |
| Mask-valid zero COT with non-clear/missing CLP | 0 |
| Valid raw COT vs packed label mismatch | 0 |
| Exact station-grid coordinate matches | 6,536 / 6,536 |
| Historical NetCDF files containing a QA variable | 0 / 3,290 |

This verifies that the pack preserved valid clear-class zero labels and that invalid/out-of-mask zero placeholders are excluded from supervision. It does **not** prove that every historical clear-class result was a native retrieval: the historical NetCDFs have no QA flag, while the portable V2 code's QA field is a later contract. The supplied producer code says CLP=0 means clear and assigns COT=0 there, but without historical run logs/checkpoint binding, a previous-hour fallback or upstream silent failure cannot be ruled out solely from these files.

## Validation-set offset sensitivity

The fixed common support contains 934,809 sample-pixels over 6,536 validation samples (mean 143.0 pixels/sample). All 25 offsets use exactly this same support. Offsets are applied to CPP references while predictions stay fixed.

| dy | dx | RMSE (COT units) | Δ vs 0,0 |
|---:|---:|---:|---:|
| -2 | -2 | 5.1041 | +1.0623 |
| -2 | -1 | 4.7955 | +0.7537 |
| -2 | +0 | 4.6372 | +0.5954 |
| -2 | +1 | 4.6564 | +0.6146 |
| -2 | +2 | 4.8373 | +0.7955 |
| -1 | -2 | 4.6659 | +0.6241 |
| -1 | -1 | 4.3395 | +0.2977 |
| -1 | +0 | 4.1933 | +0.1515 |
| -1 | +1 | 4.2616 | +0.2198 |
| -1 | +2 | 4.5156 | +0.4738 |
| +0 | -2 | 4.4551 | +0.4134 |
| +0 | -1 | 4.1491 | +0.1073 |
| +0 | +0 | 4.0418 | +0.0000 |
| +0 | +1 | 4.1625 | +0.1207 |
| +0 | +2 | 4.4702 | +0.4284 |
| +1 | -2 | 4.5350 | +0.4932 |
| +1 | -1 | 4.2945 | +0.2527 |
| +1 | +0 | 4.2465 | +0.2047 |
| +1 | +1 | 4.4037 | +0.3619 |
| +1 | +2 | 4.7230 | +0.6812 |
| +2 | -2 | 4.8757 | +0.8339 |
| +2 | -1 | 4.7197 | +0.6779 |
| +2 | +0 | 4.7274 | +0.6857 |
| +2 | +1 | 4.8982 | +0.8564 |
| +2 | +2 | 5.1976 | +1.1558 |

Zero-offset RMSE: **4.0418**. Best aggregate offset: **dy=+0, dx=+0**, RMSE 4.0418 (0.00% lower on the same cohort). This is a sensitivity clue only.

### Aggregate by station

| Station | Samples | Best (dy,dx) | RMSE at best | RMSE at 0,0 |
|---|---:|---:|---:|---:|
| sili | 3246 | (0, 0) | 4.4812 | 4.4812 |
| zhujia | 3290 | (-1, 0) | 3.5428 | 3.5564 |

### Best offset by station and UTC date

| Station | UTC date | Samples | Best (dy,dx) | RMSE at best | RMSE at 0,0 |
|---|---|---:|---:|---:|---:|
| sili | 2025-06-30 | 6 | (-1, -2) | 0.9623 | 1.4603 |
| sili | 2025-07-01 | 49 | (-1, -1) | 2.6316 | 2.8629 |
| sili | 2025-07-02 | 49 | (-1, 1) | 1.8073 | 1.9122 |
| sili | 2025-07-03 | 49 | (1, 0) | 0.7410 | 0.7568 |
| sili | 2025-07-04 | 46 | (0, 0) | 1.6796 | 1.6796 |
| sili | 2025-07-05 | 48 | (0, 1) | 3.3332 | 3.5503 |
| sili | 2025-07-06 | 48 | (0, 1) | 2.6509 | 2.7728 |
| sili | 2025-07-07 | 48 | (1, -1) | 4.2518 | 4.8051 |
| sili | 2025-07-08 | 48 | (0, 0) | 4.2135 | 4.2135 |
| sili | 2025-07-09 | 48 | (-1, 0) | 9.3991 | 9.5500 |
| sili | 2025-07-10 | 48 | (0, 0) | 3.2280 | 3.2280 |
| sili | 2025-07-11 | 47 | (1, -1) | 3.4918 | 3.7088 |
| sili | 2025-07-12 | 42 | (0, 2) | 6.7881 | 7.6136 |
| sili | 2025-07-13 | 47 | (0, 0) | 4.8543 | 4.8543 |
| sili | 2025-07-14 | 45 | (2, 0) | 7.2324 | 8.4164 |
| sili | 2025-07-15 | 45 | (0, 0) | 2.8169 | 2.8169 |
| sili | 2025-07-16 | 45 | (2, 0) | 0.8372 | 0.9396 |
| sili | 2025-07-17 | 45 | (0, 0) | 1.2923 | 1.2923 |
| sili | 2025-07-18 | 47 | (0, 0) | 4.7314 | 4.7314 |
| sili | 2025-07-19 | 47 | (0, 0) | 2.5114 | 2.5114 |
| sili | 2025-07-20 | 47 | (0, 1) | 2.9623 | 3.0487 |
| sili | 2025-07-21 | 47 | (0, 0) | 5.0517 | 5.0517 |
| sili | 2025-07-22 | 47 | (-2, 1) | 5.3418 | 5.6807 |
| sili | 2025-07-23 | 47 | (-2, -1) | 4.0462 | 4.7171 |
| sili | 2025-07-24 | 47 | (0, 0) | 1.9249 | 1.9249 |
| sili | 2025-07-25 | 47 | (1, 0) | 1.7398 | 1.7919 |
| sili | 2025-07-26 | 47 | (0, 1) | 4.5137 | 5.0654 |
| sili | 2025-07-27 | 47 | (0, 0) | 2.7559 | 2.7559 |
| sili | 2025-07-28 | 47 | (0, 2) | 7.2267 | 7.5430 |
| sili | 2025-07-29 | 47 | (-1, 0) | 4.6517 | 5.1292 |
| sili | 2025-07-30 | 47 | (0, -1) | 6.1116 | 6.3764 |
| sili | 2025-07-31 | 47 | (0, 0) | 4.0168 | 4.0168 |
| sili | 2025-08-01 | 47 | (0, 0) | 2.6118 | 2.6118 |
| sili | 2025-08-02 | 47 | (1, 0) | 1.7138 | 1.9361 |
| sili | 2025-08-03 | 47 | (0, 2) | 3.3344 | 3.9301 |
| sili | 2025-08-04 | 47 | (0, 0) | 2.3700 | 2.3700 |
| sili | 2025-08-05 | 47 | (1, 2) | 7.7150 | 8.0711 |
| sili | 2025-08-06 | 46 | (2, 1) | 6.7654 | 7.0797 |
| sili | 2025-08-07 | 47 | (2, 2) | 9.3316 | 10.4553 |
| sili | 2025-08-08 | 45 | (0, 0) | 1.8331 | 1.8331 |
| sili | 2025-08-09 | 47 | (0, 1) | 1.1437 | 1.1795 |
| sili | 2025-08-10 | 47 | (0, 0) | 0.8456 | 0.8456 |
| sili | 2025-08-11 | 44 | (1, 1) | 1.3444 | 1.3691 |
| sili | 2025-08-12 | 44 | (1, 0) | 1.5067 | 1.5344 |
| sili | 2025-08-13 | 45 | (0, 0) | 3.8381 | 3.8381 |
| sili | 2025-08-14 | 45 | (0, -1) | 9.2152 | 9.3608 |
| sili | 2025-08-15 | 40 | (-2, -2) | 3.3773 | 4.9106 |
| sili | 2025-08-16 | 45 | (0, -2) | 3.5152 | 4.2718 |
| sili | 2025-08-17 | 45 | (0, 1) | 2.0215 | 2.0950 |
| sili | 2025-08-18 | 44 | (-2, 0) | 4.1039 | 5.2940 |
| sili | 2025-08-19 | 45 | (2, 1) | 3.7626 | 3.9824 |
| sili | 2025-08-20 | 45 | (-2, 1) | 5.5915 | 5.9238 |
| sili | 2025-08-21 | 44 | (-1, 0) | 9.1528 | 9.1784 |
| sili | 2025-08-22 | 45 | (0, 1) | 2.0218 | 2.1822 |
| sili | 2025-08-23 | 45 | (0, 0) | 0.9108 | 0.9108 |
| sili | 2025-08-24 | 42 | (0, 1) | 2.4996 | 2.5396 |
| sili | 2025-08-25 | 45 | (2, 0) | 0.6066 | 0.6924 |
| sili | 2025-08-26 | 45 | (2, -1) | 0.5599 | 0.8592 |
| sili | 2025-08-27 | 45 | (1, 1) | 3.7345 | 4.6099 |
| sili | 2025-08-28 | 45 | (0, 0) | 3.1484 | 3.1484 |
| sili | 2025-08-29 | 45 | (0, 2) | 2.6048 | 3.4844 |
| sili | 2025-08-30 | 43 | (0, -2) | 2.3029 | 3.0971 |
| sili | 2025-08-31 | 45 | (0, 1) | 3.3326 | 3.3943 |
| sili | 2025-09-01 | 31 | (0, 0) | 7.0831 | 7.0831 |
| sili | 2025-09-02 | 45 | (0, 0) | 1.5517 | 1.5517 |
| sili | 2025-09-03 | 45 | (2, -1) | 1.0147 | 1.8281 |
| sili | 2025-09-04 | 42 | (1, -1) | 0.9901 | 1.0381 |
| sili | 2025-09-05 | 44 | (0, 1) | 2.1097 | 2.1519 |
| sili | 2025-09-06 | 37 | (-2, 2) | 2.9917 | 4.1726 |
| sili | 2025-09-07 | 38 | (0, 0) | 1.9096 | 1.9096 |
| sili | 2025-09-08 | 21 | (0, 0) | 5.6982 | 5.6982 |
| sili | 2025-09-09 | 42 | (1, -2) | 4.9998 | 5.3620 |
| sili | 2025-09-10 | 41 | (-1, 1) | 1.4916 | 1.5957 |
| sili | 2025-09-22 | 1 | (-2, -1) | 4.7007 | 5.2699 |
| zhujia | 2025-06-30 | 6 | (2, 1) | 0.0268 | 0.0301 |
| zhujia | 2025-07-01 | 49 | (0, 2) | 0.0918 | 0.1890 |
| zhujia | 2025-07-02 | 49 | (0, 2) | 0.0848 | 0.0866 |
| zhujia | 2025-07-03 | 49 | (-2, -2) | 0.0302 | 0.0302 |
| zhujia | 2025-07-04 | 47 | (-2, -2) | 0.0326 | 0.0326 |
| zhujia | 2025-07-05 | 49 | (-2, -2) | 0.0316 | 0.0358 |
| zhujia | 2025-07-06 | 49 | (1, -1) | 2.3929 | 3.3267 |
| zhujia | 2025-07-07 | 49 | (0, 2) | 0.2736 | 0.3442 |
| zhujia | 2025-07-08 | 49 | (0, 1) | 2.2382 | 2.3806 |
| zhujia | 2025-07-09 | 49 | (0, -2) | 1.4837 | 2.0798 |
| zhujia | 2025-07-10 | 49 | (0, 0) | 0.8524 | 0.8524 |
| zhujia | 2025-07-11 | 49 | (-1, -1) | 1.2392 | 1.3445 |
| zhujia | 2025-07-12 | 44 | (0, 0) | 1.4612 | 1.4612 |
| zhujia | 2025-07-13 | 49 | (-1, -2) | 3.0999 | 5.3793 |
| zhujia | 2025-07-14 | 47 | (0, 0) | 0.7517 | 0.7517 |
| zhujia | 2025-07-15 | 47 | (0, 2) | 0.4316 | 0.5109 |
| zhujia | 2025-07-16 | 47 | (0, 1) | 0.4286 | 0.4712 |
| zhujia | 2025-07-17 | 47 | (1, -1) | 2.6024 | 2.7009 |
| zhujia | 2025-07-18 | 49 | (0, -1) | 4.4098 | 4.9187 |
| zhujia | 2025-07-19 | 49 | (0, 1) | 0.9316 | 1.0057 |
| zhujia | 2025-07-20 | 49 | (-1, 0) | 2.1883 | 2.5555 |
| zhujia | 2025-07-21 | 49 | (-1, 0) | 3.1039 | 3.3148 |
| zhujia | 2025-07-22 | 49 | (-1, -1) | 5.3192 | 5.4517 |
| zhujia | 2025-07-23 | 48 | (2, 1) | 3.9604 | 4.2369 |
| zhujia | 2025-07-24 | 48 | (1, 0) | 2.9606 | 3.0230 |
| zhujia | 2025-07-25 | 47 | (0, 1) | 1.8235 | 1.9074 |
| zhujia | 2025-07-26 | 47 | (1, 1) | 3.2581 | 3.4892 |
| zhujia | 2025-07-27 | 47 | (0, 0) | 5.3531 | 5.3531 |
| zhujia | 2025-07-28 | 47 | (1, -1) | 2.4928 | 2.6234 |
| zhujia | 2025-07-29 | 47 | (-2, 0) | 0.5877 | 0.6966 |
| zhujia | 2025-07-30 | 47 | (-2, 0) | 2.3474 | 3.0833 |
| zhujia | 2025-07-31 | 47 | (0, 1) | 0.2822 | 0.2859 |
| zhujia | 2025-08-01 | 47 | (0, 0) | 0.4596 | 0.4596 |
| zhujia | 2025-08-02 | 47 | (0, 1) | 0.1200 | 0.1243 |
| zhujia | 2025-08-03 | 47 | (0, 1) | 0.1569 | 0.1953 |
| zhujia | 2025-08-04 | 47 | (1, -1) | 1.3321 | 1.3554 |
| zhujia | 2025-08-05 | 47 | (0, -1) | 1.0683 | 1.1069 |
| zhujia | 2025-08-06 | 46 | (-1, 2) | 6.2855 | 6.9514 |
| zhujia | 2025-08-07 | 47 | (-1, -1) | 5.4697 | 5.5985 |
| zhujia | 2025-08-08 | 45 | (0, 0) | 1.0399 | 1.0399 |
| zhujia | 2025-08-09 | 47 | (0, 0) | 0.8563 | 0.8563 |
| zhujia | 2025-08-10 | 47 | (0, -1) | 5.6991 | 5.7708 |
| zhujia | 2025-08-11 | 45 | (-2, -2) | 7.7715 | 8.7314 |
| zhujia | 2025-08-12 | 45 | (-2, -2) | 8.0954 | 9.3509 |
| zhujia | 2025-08-13 | 46 | (1, -1) | 8.7134 | 8.9719 |
| zhujia | 2025-08-14 | 47 | (0, 0) | 4.2197 | 4.2197 |
| zhujia | 2025-08-15 | 42 | (1, 0) | 0.5602 | 0.6015 |
| zhujia | 2025-08-16 | 47 | (-1, 0) | 0.0392 | 0.0500 |
| zhujia | 2025-08-17 | 46 | (-2, -2) | 0.0288 | 0.0288 |
| zhujia | 2025-08-18 | 44 | (-2, -2) | 0.0344 | 0.0386 |
| zhujia | 2025-08-19 | 45 | (0, 0) | 0.1567 | 0.1567 |
| zhujia | 2025-08-20 | 45 | (0, -1) | 1.0127 | 1.0335 |
| zhujia | 2025-08-21 | 44 | (-1, 0) | 1.4829 | 1.6219 |
| zhujia | 2025-08-22 | 45 | (-2, 1) | 1.1368 | 1.6593 |
| zhujia | 2025-08-23 | 45 | (-1, 0) | 3.0243 | 3.1321 |
| zhujia | 2025-08-24 | 42 | (0, 0) | 1.2981 | 1.2981 |
| zhujia | 2025-08-25 | 45 | (0, -1) | 0.2929 | 0.3283 |
| zhujia | 2025-08-26 | 45 | (-1, 0) | 0.6052 | 0.6538 |
| zhujia | 2025-08-27 | 45 | (1, -2) | 0.7682 | 0.7975 |
| zhujia | 2025-08-28 | 45 | (0, -2) | 0.5816 | 0.9014 |
| zhujia | 2025-08-29 | 45 | (-1, 1) | 0.8943 | 0.9485 |
| zhujia | 2025-08-30 | 43 | (-2, 0) | 3.8948 | 4.1319 |
| zhujia | 2025-08-31 | 45 | (0, 2) | 3.9369 | 4.6220 |
| zhujia | 2025-09-01 | 31 | (0, -1) | 3.9328 | 4.2894 |
| zhujia | 2025-09-02 | 45 | (-2, 0) | 6.0338 | 7.3547 |
| zhujia | 2025-09-03 | 45 | (2, 2) | 0.9362 | 1.0271 |
| zhujia | 2025-09-04 | 43 | (-1, 0) | 1.2492 | 1.4004 |
| zhujia | 2025-09-05 | 44 | (-1, 0) | 0.6625 | 0.6786 |
| zhujia | 2025-09-06 | 37 | (-1, 1) | 1.9076 | 1.9305 |
| zhujia | 2025-09-07 | 38 | (0, 1) | 1.5571 | 1.6600 |
| zhujia | 2025-09-08 | 21 | (2, -1) | 3.5143 | 3.8212 |
| zhujia | 2025-09-09 | 42 | (0, 1) | 6.2783 | 6.3551 |
| zhujia | 2025-09-10 | 41 | (-2, -2) | 9.4870 | 11.3524 |
| zhujia | 2025-09-22 | 1 | (-1, -2) | 5.0771 | 5.8608 |

## Interpretation and limits

- The four false-cloud pixels are not CPP fill values: the source stores finite COT=0 with CLP=0, which the attached producer QC code defines as clear sky. Their high model cloud probability is therefore a classifier/product-disagreement issue at those points, not a packaging conversion of missing COT to zero.
- The historical files have no independent QA layer. The portable V2 package has a later QA/fallback contract, so it cannot establish the exact lineage of these 2025 products. We cannot rule out all retrieval failures from COT/CLP alone.
- Original AGRI NPZs were permission-denied on the current remote account. The pack/sidecar retain timestamp, original SHA-256, and geolocation records; the source CPP crop aligns exactly with sidecar and matches the corrected grid coordinates, but raw AGRI values were not reopened here.
- The ±1–2 px experiment compares a frozen model to shifted CPP retrieval targets on a common, fixed cohort. A lower error under a shift does not identify the correct geolocation and must not be used to move labels. Any stronger conclusion needs independent geolocation controls (AGRI navigation arrays / original source patch) and cross-date / cross-station agreement.
