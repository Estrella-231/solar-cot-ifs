# Four false-cloud cases: pixelwise and temporal forensic review

Date: 2026-09-24. Scope is limited to four previously selected **validation** errors and their adjacent `−15/0/+15 min` validation frames. No test rows were read. “Final COT” means the staged classifier plus conditional-regression retrieval evaluated on that frame's real AGRI input; it is not a future-COT forecast.

## What the matched images show

The composite is built from the actual preprocessed AGRI 20-channel source, cropped to each station's 16×16 patch. Pseudo-RGB is `R=C03, G=C02, B=C01`, with a fixed validation-wide 1st–99th percentile stretch per channel. The thermal panel is source C13 in kelvin. CPP COT and retrieved COT share one 0–60 display range; values above 60 are clipped. All panels use the same row/column coordinates. Red × marks the false-cloud pixel; white ★ marks a correctly classified pixel from the same frame, selected to match solar angle and its local 5×5 CPP cloud context.

![Four central validation frames with correctly classified controls](forensic_figures/four_cases_central_frame_overview.png)

The controls are all CPP ice-cloud pixels (CLP=2) and the staged model classifies them as cloudy. The four controls have `P(cloud)=0.948–0.990`; their retrieved COT is 8.12–45.87. Meanwhile each marked error pixel is CPP clear (COT=0, CLP=0), but `P(cloud)=0.986–0.987` and retrieved COT is 32.39–36.94. Thus the local maps show a sharp pixel-class disagreement inside a cloud-rich scene, rather than a general failure to detect the surrounding cloud mass.

| Case | False pixel | cosSOZ difference to control | 5×5 fraction with CPP COT≥10 (false / control) | False: CPP COT / CLP / P(cloud) / final COT | Correct control: pixel / CPP CLP / CPP COT / P(cloud) / final COT |
|---|---:|---:|---:|---:|---:|
| 1 | (0,9) | 0.00087 | 0.733 / 0.750 | 0 / clear / 0.987 / 36.94 | (1,7) / ice / 33.18 / 0.979 / 24.37 |
| 2 | (0,9) | 0.00011 | 0.933 / 0.933 | 0 / clear / 0.987 / 36.57 | (0,8) / ice / 35.20 / 0.990 / 45.87 |
| 3 | (8,11) | 0.00096 | 0.480 / 0.480 | 0 / clear / 0.986 / 35.28 | (11,8) / ice / 34.92 / 0.948 / 8.12 |
| 4 | (0,6) | 0.00096 | 0.800 / 0.800 | 0 / clear / 0.986 / 32.39 | (8,0) / ice / 30.52 / 0.981 / 27.42 |

The scene controls match the false pixels' solar cosine to within 0.001 and have the same or nearly the same nearby-cloud fraction. Three controls are close spatial neighbors; case 4's closest context match is farther across the same 16×16 patch. Their local 5×5 mean COT values are also similar (false/control: 23.45/21.22, 36.51/40.37, 23.65/19.21, and 24.70/23.96). This makes them useful same-scene controls, while preserving the fact that their center-pixel CPP phase is ice rather than clear.

## What the adjacent times add

| Case | False pixel CPP COT at −15 / 0 / +15 min | `P(cloud)` at −15 / 0 / +15 min | Retrieved COT at −15 / 0 / +15 min |
|---|---|---|---|
| 1 | 0.00 / 0.00 / 36.33 | 0.985 / 0.987 / 0.986 | 29.70 / 36.94 / 32.64 |
| 2 | 22.98 / 0.00 / 32.81 | 0.972 / 0.987 / 0.990 | 15.18 / 36.57 / 50.98 |
| 3 | 49.98 / 0.00 / 0.00 | 0.986 / 0.986 / 0.983 | 35.87 / 35.28 / 26.78 |
| 4 | 0.00 / 0.00 / 19.13 | 0.972 / 0.986 / 0.987 | 15.54 / 32.39 / 35.42 |

These retrieval outputs are frame-wise: the classifier and COT regressor do not consume temporal history. Their persistent high probability therefore reflects similar spatial radiance/cloud context in each frame, not learned temporal memory. The reference COT at the marked pixel changes across a 15-minute step in all four selected cases, while model probability stays high. At every zero-COT time, the validation-wide source audit confirms CLP=clear. This is consistent with a CPP clear/cloud transition at the pixel and with the retriever spreading neighboring cloud evidence across the local boundary; it does not by itself decide whether the transition is a CPP retrieval/classification fluctuation, a geolocation/time-registration issue, or a real short-lived cloud edge.

The available original NetCDF CLP crops cover the four center times. For adjacent frames, only zero-COT pixels are labeled clear; nonzero-COT pixels are shaded as phase-unknown in the temporal pages because adjacent water/ice CLP arrays were not recovered. Thus the maps do **not** claim to show adjacent cloud-phase transitions. The four central false-pixel CLP values and the matched-control CLP values are directly read from the original CPP NetCDF crops. Historical source files have no independent QA flag, and CPP is an AGRI-derived retrieval reference rather than an independent ground-truth COT observation.

![Case 1 temporal sequence](forensic_figures/case_1_temporal_forensics.png)

Each case has a full sequence page in the PDF and its own PNG in `forensic_figures/`. They show the pseudo-RGB, C13 temperature, CPP COT, available CPP phase, cloud probability, and final retrieved COT at all three timestamps with one shared physical scale per quantity.

## Solar-geometry chain

I fetched the original preprocessed 20-channel AGRI files for the 12 selected frames from the accessible ZJNU server, cropped the documented station patches, and reconstructed all 16 packed inputs. Across all 12 frames and 16×16 pixels:

- The 13 imagery channels match the packed channels exactly (maximum absolute difference 0).
- `cosSOZ = cos(deg2rad(SOZ))` matches exactly (maximum difference 0).
- `cosRAA = cos(deg2rad(SOA − SAA))` matches exactly (maximum difference 0).
- `day_mask = source_day_mask > 0.5` has zero mismatched pixels.

The model standardizes `cosSOZ` and `cosRAA` using train-only statistics: means/stds are `0.63983/0.24325` and `0.29319/0.46211`. `day_mask` has mean 0 and std 1, so it remains binary. The calculation uses degree-valued source angles and converts to radians before cosine. I found no angle order, degree/radian, or normalization error in these 12 real inputs.

As a secondary sanity check, an independent NOAA nominal-UTC approximation differs from the saved cosSOZ by MAE 0.00699 (RMSE 0.00807). A uniform time shift of about +4.5 to +6 minutes reduces that residual, but the saved input already matches the source AGRI pixel-angle field exactly. The offset is therefore not a correction and is not evidence that the model input is wrong; it reflects an unresolved difference between the nominal-time approximation and the product's per-pixel angle convention.

![Saved AGRI solar input and independent nominal-time comparison](forensic_figures/solar_geometry_input_vs_noaa.png)

## Interpretation and remaining checks

This evidence supports a narrow diagnosis: the staged model confidently labels cloud-rich texture as cloud, including in neighboring true ice-cloud pixels, but it also assigns similar confidence to the four CPP-clear edge pixels. The strongest current hypothesis is insufficient pixel-level discrimination at cloud boundaries, potentially compounded by temporally inconsistent CPP labels. The solar-input construction is ruled out for these selected samples. These four examples do not establish the cause across the full validation set or prove CPP is physically wrong.

To close the remaining uncertainty, recover the original CPP CLP crops for the eight adjacent timestamps, then compare the corresponding full-resolution AGRI/CPP coordinates and scan-time conventions. The stored map, control selection, point timelines, source-angle verification, and solar audit are reproducible with:

- `scripts/select_cot_matched_cases_20260924.py`
- `scripts/verify_original_agri_solar_inputs_20260924.py`
- `scripts/plot_cot_false_cloud_forensics_20260924.py`
- `scripts/audit_cot_solar_geometry_20260924.py`

Machine-readable outputs include `false_cloud_matched_controls_20260924.csv`, `forensic_figures/four_case_pixel_timelines.csv`, `original_agri_solar_input_audit_20260924.json`, and `solar_geometry_audit_20260924.json` in this directory.
