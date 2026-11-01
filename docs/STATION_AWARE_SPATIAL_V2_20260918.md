# Station-aware aligned AGRI+COT v2 — 2026-09-18

## Verified station geometry

The frozen Hunan S contract and the independent station-grid audit agree that the nearest grid pixel is patch coordinate `[8, 8]` for both stations:

| station | 16×16 bounds in Hunan grid `[r0:r1,c0:c1]` | station patch coordinate | distance to nearest grid pixel |
|---|---|---:|---:|
| Sili | `[155:171, 140:156]` | `[8,8]` | 1.811 km |
| Zhujia | `[65:81, 142:158]` | `[8,8]` | 1.218 km |

The spatial input is therefore station-centred under the frozen grid contract. This verifies the coordinate used for the station-aware aggregation; it does not close the separate CPP historical producer provenance gap.

## New head contract

`station_aware_spatial_v2` is a separately named head and does not overwrite H14 or its results.

1. The 13 aligned forecast-AGRI channels and the aligned normalized `log1p(COT)` map are concatenated to `[14,16,16]` and enter the same first `Conv2d(14→16)`. Thus AGRI and COT are processed at each corresponding pixel before any pooling.
2. After the two convolutional layers, the predictor concatenates three joint spatial summaries: encoded feature at `[8,8]`, mean encoded feature in the `[6:11,6:11]` 5×5 station neighbourhood, and full 16×16 mean encoded feature.
3. `A_station` and `AC_station` use exactly the same 18,521-parameter head; only the COT plane is respectively zero or supplied. Both retain geometry, lead and station embeddings, while the old four-statistic COT slots are zero.

## B0 acceptance

PBS `209540.tc6000` completed with `PASS_B0_STATION_AWARE_SPATIAL_GPU_SMOKE`; no test rows were read. COT first-layer gradient L1 was `0.05155`, and adding `0.25` to the COT plane changed station predictions (mean absolute change `3.80e-4` in predicted kt). The full GPU-resident pack profile selected batch 16,384 from the tested 2,048/8,192/16,384 candidates, with 888,416 rows/s and 9.99 GiB peak allocation.

## Validation-only seed-42 pilot

PBS `209543.tc6000` completed against the fixed 99,849 validation rows. It used 10 epochs, batch 16,384, AdamW (`lr=3e-4`, `weight_decay=1e-4`), and the unchanged GHI loss. Test and IFS were not used.

| group | best epoch | Sili RMSE (W/m²) | Zhujia RMSE (W/m²) | equal-station RMSE (W/m²) |
|---|---:|---:|---:|---:|
| A_station | 9 | 168.7611 | 133.6108 | **151.1860** |
| AC_station | 4 | 169.5363 | 132.9672 | 151.2517 |

`AC_station − A_station = +0.0658 W/m²` (+0.044%) at the equal-station primary endpoint. COT improves Zhujia by 0.6437 W/m² but degrades Sili by 0.7752 W/m². This run used automatic batch 16,384, which reduced the 10-epoch optimizer-update count by eightfold versus the pre-registered B1 batch-2,048 contract. It is therefore an implementation/throughput screen, not a valid positive-or-negative representation verdict; it must not be expanded to additional seeds, IFS, or test.

The corrected batch-2,048 rerun was submitted as PBS `209558.tc6000`, but the scheduler aborted it before Python started (`Gold is abnormal`, exit status `-2`, zero CPU/GPU time). No corrected result exists yet. Resume only after the PBS accounting service accepts a fresh job; do not treat the batch-16,384 result as evidence that COT degrades.

## Evidence

- Remote B0: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/audits/station_aware_spatial_b0_20260918_v1`
- Remote B1: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/station_aware_spatial_b1_seed42_20260918_v1`
- B1 base audit SHA-256: `3de1c9d622bc5838ef11db2347c675d4b7db9b8e1106ce7ac5a478215e7ee08c`
- B1 COT sidecar audit SHA-256: `e98bc3e91871ef34c2c8cb498c960e3c231cd750f923574fa86a15d5578f5edf`
- Training script SHA-256: `0a6943d371ab693dd3552e22c373ef875e156c68172c2835e6af7e126fad71ce`
