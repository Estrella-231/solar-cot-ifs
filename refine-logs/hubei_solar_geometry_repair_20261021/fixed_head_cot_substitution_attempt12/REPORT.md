# Attempt12 fixed-GHI-head COT replacement diagnostic

## Question

The corrected-solar, three-station paired forecast experiment found almost no aggregate GHI improvement from forecast COT. This validation-only counterfactual asks whether the trained FAC head can use a different COT input when AGRI, solar geometry, target GHI, checkpoint, and all other preprocessing stay fixed.

## Procedure and scope

For each Hubei station, the seed-42 FAC checkpoint was loaded without retraining. Its normalized AGRI and geometry inputs were unchanged. Only the COT input was replaced among three sources:

1. **Forecast COT:** corrected SimVP-predicted future AGRI passed through the frozen corrected R. This is the deployable forecast-chain input and the baseline for this diagnostic.
2. **History-persistence COT:** observed AGRI at the forecast initialization time passed through the same frozen R and held constant at all leads.
3. **Offline future-AGRI COT:** observed future AGRI at each target time passed through the same frozen R. This is an oracle-style attribution input; it is unavailable at forecast issuance and is not a forecast score.

All three sources were scored on exact `(station, init_time_bjt, target_time_bjt, lead_minutes)` joins. A row was retained only when both initialization and target timestamps existed in the saved R validation bank, so every arm had the same rows within a station. The resulting aligned support is small: 992/5,305 Niushou rows (18.7%), 874/4,883 Wangzhai rows (17.9%), and 909/5,087 Mizhuang rows (17.9%). The all-station date-cluster interval uses the 15 target dates represented at all three stations. This support restriction makes the counterfactual exploratory; it cannot be treated as the complete validation result.

Only training and validation assets were read. Test and buffer payload reads were zero. The offline future-AGRI arm is explicitly non-deployable. CPP was not used as independent truth in this GHI-head substitution.

## Results

Equal-station RMSE on the aligned common support:

| Horizon | Forecast COT | History-persistence COT | Offline future-AGRI COT | Offline vs forecast change |
|---|---:|---:|---:|---:|
| All leads | 122.58 | 137.12 | 117.75 | **3.94% lower** |
| 1 h | 127.02 | 130.48 | 123.10 | 3.08% lower |
| 2 h | 124.66 | 136.99 | 120.91 | 3.01% lower |
| 3 h | 122.14 | 141.11 | 117.46 | 3.83% lower |
| 4 h | 115.44 | 139.73 | 108.07 | 6.39% lower |

RMSE is in W m⁻². The 95% target-date-cluster interval for the offline-future-minus-forecast RMSE difference is `[-10.63, -1.39] W m⁻²` over all leads on this restricted support. The corresponding history-persistence interval is `[-5.69, +44.42] W m⁻²`; it crosses zero. The 2 h offline-future interval also just crosses zero (`[-12.76, +0.02] W m⁻²`); the 1 h, 3 h, and 4 h intervals are below zero on this selected support.

Station-specific all-lead RMSE:

| Station | Forecast COT | History-persistence COT | Offline future-AGRI COT | Offline vs forecast change |
|---|---:|---:|---:|---:|
| Niushou | 103.56 | 129.90 | 103.83 | 0.26% higher |
| Wangzhai | 121.28 | 130.17 | 111.71 | 7.89% lower |
| Mizhuang | 142.90 | 151.28 | 137.70 | 3.64% lower |

The baseline seed-42 FAC predictions were recomputed from their checkpoint on CPU FP32 and compared against the saved A800 autocast-FP16 predictions. Maximum absolute differences were 0.186, 0.237, and 0.193 W m⁻² for Niushou, Wangzhai, and Mizhuang; RMS differences were 0.040, 0.083, and 0.060 W m⁻². These are small numerical differences relative to GHI forecast errors and are recorded as a CPU/GPU precision difference, not a data-key mismatch.

An independent FP64 audit recomputed all 45 station/horizon/source metric rows and the date-cluster intervals from the saved 2,775 paired prediction rows. It passed with exact key uniqueness and no test payload reads.

## Interpretation

The frozen GHI head is responsive to its COT input: substituting historical persistence changes performance, and the offline future-AGRI COT lowers GHI RMSE by about 3.9% on this aligned subset. That makes “the head completely ignores COT” an unlikely explanation for the near-zero full-validation FA/FAC difference. The result is consistent with forecast-COT quality limiting downstream benefit, especially at longer leads, but does not prove that explanation: the offline arm uses observed future AGRI, the support covers only about 18% of validation rows, and the diagnostic uses one GHI seed. It is not a deployable improvement or an independent-test result.

The full validation paired FA/FAC result remains the primary forecast comparison: 95.703 versus 95.609 W m⁻² equal-station/equal-seed RMSE, a 0.098% point reduction with date-block intervals spanning zero. The replacement diagnostic must not be substituted for that result.

## Reproduction and provenance

- Main script: [`fixed_head_cot_substitution_attempt12.py`](../fixed_head_cot_substitution_attempt12.py)
- Independent FP64 audit script: [`audit_fixed_head_cot_substitution_attempt12.py`](../audit_fixed_head_cot_substitution_attempt12.py)
- Saved common-support predictions: [`paired_predictions_common_validation.csv`](paired_predictions_common_validation.csv)
- Metric table: [`metrics_by_station_horizon.csv`](metrics_by_station_horizon.csv)
- Machine-readable run receipt: [`summary.json`](summary.json)
- Independent recomputation: [`independent_fp64_audit.json`](independent_fp64_audit.json)

Attempt12 server root: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021`.

Key SHA-256 values:

| Asset | SHA-256 |
|---|---|
| Corrected full R checkpoint | `d8f25a34ac1bf6fdd9005d2ea9d52d02dd3c2c24e4758e6cd74061121d6b5d0b` |
| Niushou seed-42 FAC checkpoint | `c5f84627ad7090ae45349cd49de1845d999027fac6a0e46fac3af5bb320f2033` |
| Wangzhai seed-42 FAC checkpoint | `40873a4c601a56d291307880461d8202c858846773d5bb0f04f15b9c25a5d019` |
| Mizhuang seed-42 FAC checkpoint | `3908d3e05304070ccbb2bdedf7416ae459f5d520141a0b7bf77daf4374d33510` |
| Train-only GHI normalization contract | `4ef1814a1d88107741de2b212e9d186d7950e4a26d21cb2a5466476db7476ec5` |
| GHI trainer used for model definition | `3b2319b5c72f3abb00f003f8c228d3cdd6bef1ab39fb72724ae7e8fd7bec53ab` |
| R validation keys | `87b7e95549074f53d99ad66735726f1b67cfcfbeca99164d15104c32ec7412f1` |
| R validation COT predictions | `c4ed659d776bab20c5fd8f3fb3798196c6d5e1ae2c9fd5c5e7f975c697fcf1bc` |
| R validation mask | `ee2b8ea789de0f876110801648d8aba640e69545af7bddf2d2d0561483b2d7a3` |
| Forecast-COT GHI input array | `94dec72dabf1f68032cce07fe87914038401f4bad3a4ac263ada7e24751b6968` |

