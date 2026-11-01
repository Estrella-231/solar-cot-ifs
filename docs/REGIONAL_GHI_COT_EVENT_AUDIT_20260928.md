# R256 GHI pilot: paired COT-to-GHI event audit (2026-09-28)

## Scope and matching

This is a validation-only audit of the completed regional GHI pilot. It reuses the pilot's saved no-COT, forecast-COT and oracle-COT predictions, with no retraining and no test access. The forecast and oracle sidecars use the retrained frozen R256 checkpoint (SHA-256 `74ddd7b2e78c29716260c9a790e0824659f2976d95b4598c3511c9816e77ad25`). Oracle COT is actual future AGRI passed through this same R256; it is non-deployable and is not independent COT truth.

Three previously selected Sili cloud events were matched by sequence ID to the GHI pilot's fixed validation bank: 2025-08-05 09:15 BJT (dissipation), 2025-08-14 11:30 BJT (moving/thickening cloud), and 2025-07-09 12:15 BJT (missed growth). Each has 16 target leads (+15…+240 min) at both Sili and Zhujia. The three GHI prediction files have identical row IDs, station labels, leads and observed 15-minute GHI targets; each event-site pair therefore has 16 exact paired rows. All 16 station-pixel CPP references are valid in these selected cases.

The events were selected for diagnostic cloud behavior. They are not a random or representative sample. COT errors are measured against CPP retrieval references, whose physical QA remains unresolved.

## Event-level paired results

GHI delta is candidate RMSE minus no-COT RMSE, so negative values favor the COT arm. COT columns are station-pixel RMSE against CPP, respectively for SimVP-forecast AGRI→R and actual-future AGRI→R.

| Event | Station | No-COT GHI RMSE | Forecast-COT GHI RMSE (Δ) | Oracle-COT GHI RMSE (Δ) | Forecast / oracle COT RMSE vs CPP |
|---|---|---:|---:|---:|---:|
| Dissipation, Aug 5 | Sili | 324.3 | 299.2 (−25.1) | 293.1 (−31.1) | 10.8 / 2.2 |
| Dissipation, Aug 5 | Zhujia | 79.2 | 85.9 (+6.6) | 75.2 (−4.0) | 0.1 / 0.1 |
| Moving/thickening, Aug 14 | Sili | 119.1 | 157.3 (+38.2) | 156.7 (+37.7) | 22.3 / 4.5 |
| Moving/thickening, Aug 14 | Zhujia | 214.1 | 190.8 (−23.4) | 177.4 (−36.7) | 13.3 / 0.9 |
| Missed growth, Jul 9 | Sili | 334.1 | 319.4 (−14.7) | 303.6 (−30.5) | 30.6 / 4.0 |
| Missed growth, Jul 9 | Zhujia | 113.4 | 93.8 (−19.6) | 95.6 (−17.8) | 7.6 / 2.2 |

These are event-specific RMSEs over 16 dependent leads, not significance tests. The source table also records MAE, bias, sample count and paired row-id bounds.

## What the comparison says

R256-oracle COT is much closer to CPP than forecast COT at Sili in the two thick-cloud events: RMSE falls from 22.3 to 4.5 on Aug 14 and from 30.6 to 4.0 on Jul 9. This points to future AGRI/SimVP cloud evolution as a real upstream limitation in these cases. At Zhujia, oracle COT is also closer to CPP in the Aug 14 and Jul 9 cases, while Aug 5 is nearly clear for both COT inputs.

Better COT does not guarantee better GHI. On Aug 14 at Sili, oracle COT improves the CPP comparison by 17.8 COT RMSE, yet its GHI RMSE remains 37.7 W m⁻² worse than no-COT and is almost identical to forecast-COT. The GHI head fails to turn the much better thick-cloud trajectory into a better irradiance forecast in this event. The trajectory plot shows the observed sharp GHI collapse while all three heads stay substantially too high. This is direct evidence of a downstream fusion/response limitation for that event, though it does not prove that the head alone caused the error.

The same Aug 14 sequence behaves differently at Zhujia: the site remains mostly clear and later COT decays; forecast COT improves GHI RMSE by 23.4 W m⁻² and oracle by 36.7 W m⁻². The two stations sit under different cloud trajectories and have different irradiance variability, so one joint head does not receive the same useful COT signal at both locations. On Jul 9, Sili's thick-cloud growth is missed by forecast COT and recovered by oracle COT; oracle gives a larger GHI gain than forecast COT. At Zhujia, the COT event arrives only near the end of the window, and both COT arms improve GHI similarly.

Across only these three selected cases, the mean of event-specific GHI RMSEs is Sili 259.1 (no COT), 258.6 (forecast COT), 251.2 (oracle); Zhujia 135.6, 123.5 and 116.1 W m⁻². This descriptive aggregation hides the Aug 14 Sili failure and must not be reported as a cohort result. In the full 128-sequence pilot, the station-level forecast-COT delta versus no-COT was −7.087 W m⁻² at Sili and +5.946 at Zhujia; oracle deltas were −12.093 and +3.326. The opposing site response largely cancels in the two-station average.

## Bottleneck assessment

The evidence supports **both** an upstream forecast-COT quality problem and a downstream GHI-use problem, with their relative importance varying by site and event. Forecast-COT errors are large in the two Sili thick-cloud cases and oracle COT often helps GHI, so the predicted cloud trajectory is a bottleneck. But the Aug 14 Sili oracle case shows that a much more CPP-like COT trajectory can still fail to improve GHI; the GHI head's mapping, target variability, or other AGRI/solar context remains limiting.

This pilot does not isolate a causal COT-input effect: the no-COT, forecast-COT and oracle-COT heads were trained independently, even though they share the same architecture, cohort and budget. Thus each GHI delta includes differences in learned parameters as well as the changed COT source. The next discriminating check is a frozen-head counterfactual matrix: run each saved arm's head with the same image/geometry rows while swapping only zero, forecast and oracle COT tensors, then report the per-event paired GHI changes. If an individual fixed head responds favorably to oracle COT but not forecast COT, the upstream forecast is the main constraint; if it cannot exploit oracle COT, downstream fusion/optimization is the stronger constraint. This is inference-only and should precede any new training.

## Artifacts

- Figure: `results/regional_station_cot_20260927/ghi_pilot_event_audit_20260928/matched_event_cot_ghi_trajectories.png` and `.pdf`.
- Paired per-event/site/arm metrics: `results/regional_station_cot_20260927/ghi_pilot_event_audit_20260928/event_station_arm_metrics.csv` and `.json`.
- Reproducible analysis: `scripts/analyze_regional_ghi_event_cot_20260928.py`.
- Retrieved event-site COT/CPP series: `results/regional_station_cot_20260927/ghi_pilot_event_audit_20260928/cot_event_sources.json`.

The plot uses physical COT after converting stored `log1p(COT)` with `expm1`. All GHI targets are observed 15-minute interval means. The x-axis is forecast lead from the initialization time; missing values are not interpolated. See the project validation and oracle restrictions in `AGENTS.md` and `IRRADIANCE_ERROR_PREVENTION.md`.

## Analysis procedure attribution

Figure and analysis workflow informed by Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065, v2. https://doi.org/10.48550/arXiv.2609.00065.
