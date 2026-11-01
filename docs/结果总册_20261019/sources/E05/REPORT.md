# Hunan validation: high-sun irradiance and daily irradiation

Date: 2026-10-17. This is a validation-only paired analysis of saved three-seed predictions under `hunan_monthly_block_2d_buffer1d_seed42_v1`. It opens no test payload and starts no training.

## Question and support

The synchronous GHI results are also evaluated in a high-solar-altitude regime and as integrated daily solar irradiation. The high-sun stratum is station-centre solar altitude of at least 45 degrees, derived from the frozen physical `cosSOZ` input. Scoring uses common BJT dates from both Hunan stations; there are 1,527 unique station-target rows across 35 dates. For each seed, each station's RMSE is computed first; the reported point estimate is the arithmetic mean across the two stations and three seeds. The paired date-block interval resamples the 35 BJT dates 2,000 times.

Daily irradiation is `0.25 × sum(15-min interval-mean GHI) / 1000`, in kWh m⁻². Expected daylight targets are the canonical 15-min target grid with solar elevation above zero, calculated from station coordinates and UTC target time. A day is retained only when every expected daylight target is present at both stations; no missing values are interpolated. This yields 30 common complete dates. Within each seed and station, daily absolute error is computed first; the station/date comparison averages equally over stations and seeds. Its confidence interval resamples BJT dates, preserving the paired methods and both stations.

## Results

| Validation quantity | AGRI baseline | AGRI + COT | Paired difference (AC − A), 95% date-block interval |
|---|---:|---:|---:|
| GHI RMSE, solar altitude ≥45° (W m⁻²) | 154.111 | 148.645 | −5.465 [−8.990, −2.152] |
| Daily irradiation MAE, complete days (kWh m⁻²) | 0.5491 | 0.5032 | −0.0459 [−0.0861, −0.0067] |
| Daily irradiation bias, complete days (kWh m⁻²) | −0.0164 | −0.0561 | — |

The capacity control remains close to the AGRI baseline in these quantities: high-sun RMSE 153.854 W m⁻² and complete-day irradiation MAE 0.5479 kWh m⁻².

## Interpretation and limits

Within the Hunan validation cohort, COT is associated with lower GHI error in the ≥45° solar-altitude subgroup and lower absolute error in integrated daily irradiation. The daily bias becomes more negative, so the smaller absolute error does not mean every aspect of the energy estimate improves. Daily irradiation is a radiometric resource quantity, not measured PV generation or a percentage change in power output. These results are validation evidence conditional on selected checkpoints; Hubei is not included because its label source has a different sampling contract. They do not replace the independent test.

## Reproduction

Run `python paper/analyses/solar_resource_validation_20261017.py` from the repository root. Machine-readable outputs are `solar_resource_validation_audit.json`, `stratified_validation_metrics.csv`, `daily_coverage_audit.csv`, `daily_irradiation_errors.csv`, and `high_sun_summary.csv`. The JSON records the hashes of the source pack and all prediction arrays.
