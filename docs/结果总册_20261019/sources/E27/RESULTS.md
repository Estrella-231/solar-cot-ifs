# Forecast-COT temporal-change audit (2026-10-17)

## Question

Does forecast AGRI preserve the 15-minute COT evolution seen when the same frozen retriever is applied to observed future AGRI, and do the loss-fine-tuned candidates change that response? A sparse CPP-supported comparison is included as a retrieval-reference check. This is a validation-only, read-only analysis; no test payloads or training jobs were used.

## Contract and method

- Cohort: 3,140 validation sequences; station-centred 3×3 pixels; 16 leads at 15-minute intervals. Adjacent changes are differenced within a sequence.
- Forecast banks: original SimVP, equal-budget MSE, temporal-change, and spatial-gradient candidates. All source IDs, target timestamps, and dataset indices were checked for exact equality. Current CPP values and masks were checked byte-for-byte across the R07 candidate banks.
- Physical COT values were restored from `log1p(COT)` once with `expm1`.
- Full-cohort response proxy: R(observed future AGRI), finite at both adjacent leads. This is an offline response of the same retriever, not independent COT truth.
- CPP reference: both adjacent CPP leads valid in the station-centred 3×3. Date-cluster bootstrap resamples BJT target dates; overlapping forecast windows remain separate cases.

## Results

On the full response-proxy support (52 BJT dates, 423,900 pixel transitions per station), the original SimVP→R COT-change RMSE was 9.955 at Sili and 9.180 at Zhujia, compared with 11.692 and 11.364 for last-frame persistence. Paired improvements over persistence were 1.737 (95% date-block interval 1.212–2.396) and 2.184 (1.494–3.076). Forecast-to-R change was associated with the R(observed future AGRI) response (Pearson r=0.533/0.593), but through-origin slopes were only 0.334/0.392, indicating amplitude attenuation against this proxy.

Temporal-change fine-tuning changed proxy change RMSE by −0.046 at Sili (95% interval −0.085 to −0.013) and −0.013 at Zhujia (−0.047 to +0.018) relative to original SimVP. MSE and spatial-gradient fine-tuning did not resolve a paired improvement at either station. These proxy findings do not reverse the primary CPP-referenced result: none of the candidates improves station-and-lead macro COT RMSE on CPP support.

When CPP is valid at both adjacent leads, only seven BJT dates per station remain (37,017 transitions at Sili; 37,656 at Zhujia). Original SimVP→R change had weak association with CPP change (r=0.133/0.019; slopes 0.046/0.0066). R(observed future AGRI) change had higher association (r=0.650/0.608; slopes 0.492/0.575). The direction is consistent with a loss of temporal COT response in the forecast-image chain, but seven dates and a non-independent retrieval reference do not identify forecast error as the sole cause; CPP retrieval variability and R response remain confounded.

## Interpretation and paper consequence

The defensible conclusion is limited: SimVP→R retains more temporal variation than persistence when judged against the same retriever's response to observed future AGRI, but the response is attenuated. A temporal-change loss gives a small proxy-only reduction at Sili, not a repeatable two-station result and not an improvement against CPP-referenced aggregate skill. No matched forecast-input GHI gain follows from this diagnostic. Manuscript text reports these distinctions and labels the proxy as mechanistic evidence only.

The next scientific step is not another unconstrained loss sweep. First obtain or construct a broader independently supported adjacent-time cloud/COT reference and test the observed-future-AGRI→R versus forecast-AGRI→R discrepancy on that common support. Until then, preserve the fixed-SimVP paper contract and do not treat proxy response as physical COT accuracy. Forecast-input GHI training remains governed by the existing B4 gate.

## Provenance

- Script: `analyze_cot_temporal_change.py`; SHA-256 `9bba6805e3629771d1983535d99c9efec10858edba802e4f2165df5f96675af1`.
- Original SimVP bank SHA-256: `0acc9e4095b8d401e609b50905f7252b6c897c93698cd1e3a774e32ec84ecaaf`.
- R07 MSE/source-reference bank SHA-256: `e1cafb8232b1a1aa83bf57a83dbf98a6342fa4a6b7c68eda7451fab920b257d4`.
- Temporal-change bank SHA-256: `82fc8d054bb14bd4d3faa1d90f67de39f7d9154f4d239587aa6cd5f61b9f97c6`.
- Spatial-gradient bank SHA-256: `a8f444929cb2db559a5e4d4d113e96db810de5fdee4e13ea12b6dab2aa4cc89f`.
- Summary and machine-readable metrics are in this directory. No PBS job was submitted; test payload reads remain zero.
