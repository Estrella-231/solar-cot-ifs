# Spatial COT B2 controls — 2026-09-17

## Scope

This is a seed-42 validation-only control after the `AC_map` activation and alignment diagnostic. It uses the same Hunan train/validation cohort, fixed 14-channel `SpatialHead`, 10 epochs, batch 2048, AdamW, and GHI loss as the exploratory B1 pilot. No test rows, IFS inputs, or multi-seed results were used.

The controls are:

- `AC_shuffle`: keep each sample's COT values but apply the deterministic non-center cyclic spatial permutation used by the diagnostic.
- `A_proxy`: replace the 16×16 COT map by its spatial mean broadcast over the map, preserving sample-level amplitude while removing spatial variation.

## Results

| group | best epoch | Sili RMSE (W/m²) | Zhujia RMSE (W/m²) | equal-station RMSE (W/m²) |
|---|---:|---:|---:|---:|
| AC_map (B1) | 7 | 165.0922 | 132.2524 | **148.6723** |
| AC_shuffle | 7 | 165.0939 | 132.2829 | 148.6884 |
| A_proxy | 7 | 165.3506 | 132.6464 | 148.9985 |
| A_sp (B1) | 7 | 164.0630 | 133.7177 | 148.8904 |

Relative to `AC_map`, shuffling increases equal-station RMSE by only 0.0161 W/m² (0.011%), and replacing the map by a spatial mean increases it by 0.3262 W/m² (0.219%). The no-training diagnostic already showed that zeroing COT changes predictions (RMSE 152.237 W/m²), while spatial permutation changes them only marginally. The training controls agree: the current head can use COT amplitude, but no reproducible spatial-order benefit is visible.

## Decision

The spatial-map contribution is not supported as a paper claim. Do not start seed43/44, IFS fusion, or the independent test on this `AC_map` route. Keep these controls as an engineering diagnostic and redirect the next pilot toward a clearly defined amplitude-conditioned COT representation or a producer/provenance audit before spending long GPU runs. The result remains exploratory because the historical CPP producer chain is unresolved.

## Remote evidence

- PBS control job: `209426.tc6000` (completed)
- Remote run: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/spatial_b2_controls_seed42_20260917_v1`
- State: `COMPLETE_EXPLORATORY_B2_CONTROLS`
- Base audit SHA-256: `3de1c9d622bc5838ef11db2347c675d4b7db9b8e1106ce7ac5a478215e7ee08c`
- Sidecar audit SHA-256: `e98bc3e91871ef34c2c8cb498c960e3c231cd750f923574fa86a15d5578f5edf`
- Control script SHA-256: `571ae77544f139f825b8c1eca760597a70a4604101377b80c5975a8b011543ab`
