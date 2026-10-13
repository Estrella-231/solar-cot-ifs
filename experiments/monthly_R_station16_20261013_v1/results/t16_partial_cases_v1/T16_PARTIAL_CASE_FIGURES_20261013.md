# T16 partial error-blind physical case figures

Date: 2026-10-13 (BJT)

Status: partial validation diagnostic; 4 of the preregistered 16 cases are available.

## What is shown

The four panels correspond to the frozen JJA cases in `t16_partial_case_manifest_v1.json`: two Sili and two Zhujia sequences. Each panel samples the 16×16 station patch at +15, +60, +120, and +240 min and shows:

- observed AGRI C03/C02/C01 as an RGB false-colour proxy, with each channel clipped to the fixed reflectance interval [0, 1] and gamma 1/2.2; this is an offline diagnostic of the observed future scene, not a deployable forecast input or a natural-colour photograph;
- CPP reference COT and frozen-R retrieved COT in physical units, with the same fixed 0–60 colour scale and the common CPP-valid mask; R output is transformed once with `expm1` from its `log1p(COT)` output;
- solar zenith angle reconstructed from `cosSOZ`, masked by `day_mask`, on a fixed 0–90 degree scale;
- the 16-lead observed 15-minute GHI target and A/AC same-time real-AGRI-head predictions for seeds 42/43/44, plus the station-centre CPP/R COT trajectories.

The cyan/white cross marks the station-centre pixel (8, 8). Image panels use nearest-neighbour display on the native 16×16 grid. The GHI comparison is the frozen-head retrieval experiment with real AGRI at each target time; it is not a forecast-AGRI or future-COT forecast-chain result. Per-case RMSE is descriptive only.

## What the four cases suggest—and do not establish

The selected Sili change case (2025-07-09 13:15 BJT) has a clear thick-cloud COT rise and decay in both CPP and R station-pixel trajectories, while the three-seed mean GHI head trajectories are much smoother than observed GHI during the abrupt drop around +75 to +135 min. This is consistent with unresolved timing/intensity error in the GHI mapping, but the four cases cannot identify its cause.

The paired per-case three-seed-mean RMSE differences (AC−A; negative favours AC) are +4.557, −0.871, +8.938, and +9.917 W m⁻² for Sili change, Sili hash, Zhujia change, and Zhujia hash, respectively. Thus the examples are mixed and do not visually support a universal per-event gain. They must not replace the preregistered three-seed aggregate, the weather-stratified score, or a future forecast-chain test. CPP is the retrieval reference used here, not independent truth. The sample is restricted to JJA because the present validation archive lacks complete reference sequences for the other station-season strata.

## Provenance and reproducibility

- Case selection: `experiments/monthly_R_station16_20261013_v1/results/t16_partial_case_manifest_v1.json`; frozen before reading GHI values, predictions, or errors. Rule: `refine-logs/cot_value_first_20261013/windows/window_3/T16_EVENT_SELECTION_RULE.md`.
- Figure source: `experiments/monthly_R_station16_20261013/plot_t16_partial_cases.py`.
- Figure/input/checkpoint/row hashes, selected keys, and the `test_payload_opened=false` flag: `t16_case_figure_audit.json`.
- Derived AGRI/geometry arrays are retained locally for regeneration but are not added to the public repository; the audit records their SHA-256 and their source hashes.
- The validation set only was used after the error-blind manifest had been frozen. No test payload was opened.

## Visualization workflow reference

The figure workflow followed the scientific-visualization skill guidance on fixed, disclosed scales, visible missing-data masks, and rendered-image inspection. Related methods paper: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065, v2. https://doi.org/10.48550/arXiv.2609.00065. This is a visualization-workflow reference, not a meteorological source.
