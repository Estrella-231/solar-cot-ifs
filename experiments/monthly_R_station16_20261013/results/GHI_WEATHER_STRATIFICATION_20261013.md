# CPP/CLP weather-stratified validation of matched GHI heads

Date: 2026-10-13 BJT
Scope: validation only; no test payload opened.

## Cohort and method

This diagnostic joins the frozen GHI validation pack to CPP CLP by the exact `(station, target_time_utc)` key. It uses only the CPP/CLP intersection and does not alter the primary GHI cohort: 1,291 of 4,479 GHI validation rows had a keyed CLP record; 1,284 had complete CLP support over the centered 3×3 station footprint, with seven rows left unclassified. The remaining 3,188 GHI rows lack a matched CLP record and remain in the all-weather primary result.

Weather class is based on cloud fraction within the same centered 3×3 footprint used by the GHI head: clear at cloud fraction ≤20%, cloudy at ≥80%, and mixed in between. CPP CLP classes 1 and 2 count as cloud; class 0 counts as clear. No global COT average or percentile is used. Metrics are station-equal RMSE differences, averaged over the three matched seeds. The 2,000-resample BJT-date bootstrap preserves paired methods, stations, and seeds. CPP/CLP is a co-produced retrieval classification, not an independent cloud truth.

## Results

| CLP stratum | Sili / Zhujia / total rows | Three-seed mean AC−A (W m⁻²) | 95% date-bootstrap CI | Three-seed mean AC−AP (W m⁻²) | 95% date-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Clear | 119 / 251 / 370 | −3.082 | [−7.327, 0.545] | −2.823 | [−6.871, 0.597] |
| Mixed | 26 / 48 / 74 | −1.658 | [−12.990, 4.622] | −1.651 | [−13.521, 4.499] |
| Cloudy | 465 / 375 / 840 | −3.730 | [−7.813, −0.289] | −3.559 | [−7.831, −0.119] |

Within the cloudy subset, AC beats both A and AP in each seed at the station-equal level. Its 95% date-bootstrap intervals are below zero for both paired contrasts. Clear-sky point estimates also favor AC, but both intervals cross zero. The mixed subset is small and has wide intervals; its direction varies across seeds. The larger overall paired gain therefore appears most consistently in the densely cloudy CLP stratum in this diagnostic support.

## Limits

These results are conditional on the CPP/CLP-supported subset, only two stations, and validation-selected checkpoints. The mixed group has only 74 rows, including 26 at Sili. Cloud labels come from CPP CLP and may share retrieval errors with COT. The analysis identifies a validation association; it does not establish causal physical benefit or independent-test generalization. The primary all-weather result remains the full 4,479-row cohort and must not be replaced by this restricted analysis.

## Reproduction

- Script: `../stratify_ghi_heads_by_cpp_clp.py`.
- Machine-readable output: `ghi_heads_cpp_clp_weather_validation.json`.
- The output records source and prediction SHA-256 values, keyed support, bootstrap seed, and `test_payload_opened=false`.
