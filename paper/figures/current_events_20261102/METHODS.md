# Current observed-three-frame cloud-event figure

This figure uses the exact current Hubei same-time A/AC lineage from `refine-logs/hubei_same_time_geometry_20261102/`. No model was trained, no forecast-image-trained head was substituted, and no test payload was read.

## Sources

- Local saved validation predictions: `recovered/training/{Niushou,Wangzhai,Mizhuang}_seed{42,43,44}/{A,AC}_validation.npz`.
- Row keys and physical labels: `recovered/pack/rows.csv`; validation rows are selected by the saved global indices.
- Current observed frame/COT source: remote `frames/raw.npy`, `frames/cot_log1p.npy`, `frames/keys.csv` in the same experiment directory. Only the 799 distinct validation target times (20 dates) requested by the current GHI rows were transferred. `FETCH.json` records source and output checksum.
- CPP references: local `r_validation/{val_reference_cpp_cot.npy,val_valid_mask.npy,val_keys.csv}`. Masks are used as saved; missing reference values are not filled as clear sky.
- Input preparation semantics: current `prepare_frames.py` selects C01--C06/C09--C15 and corrected cosSOZ/cosRAA/day mask. `pipeline.py` applies the full-train corrected R normalization and retains physical COT as expm1 of the saved log1p field. The C01--C06 arrays are physical reflectance, including exact-HDF restored reflectance where required, rather than normalized plotting intensities.

## Case rule

1. Fix Niushou (first station in the existing study order), rather than choose the station with greatest improvement.
2. Fix 10:30--14:30 BJT on each validation date. The initial complete 17/17 requirement gave no eligible station-day anywhere because GHI scoring rows contain gaps. It was relaxed to at least 12/17 solely for availability; every gap remains visible.
3. Require retrieved station COT peak >=30 and range >=5 on scored times. These criteria choose variable high-COT processes and are not the CPP-based thick-cloud stratum.
4. Calculate the mean of per-seed GHI RMSEs for each arm; split candidates by the sign of AC minus A. Take the earliest date in each direction group, not the best/worst score. The selected dates are 2025-09-26 and 2025-09-27. The complete candidate table is retained.
5. Display fixed start/mid/end snapshots (10:30/12:30/14:30), without selecting a maximally dramatic frame. These are instantaneous final frames of the actual three-frame input, not a new single-frame model.

This is explicitly direction-stratified illustration. It is not a random case sample, not a success-frequency estimate, and not two independent weather replications. Numerical main-paper conclusions continue to use the full validation cohort.

## Plot transformations

- RGB false-colour composite: R=C03, G=C02, B=C01; `sqrt(clip(reflectance, 0, 1))` for every displayed image. No per-image contrast adaptation. Native 16x16 values use nearest-neighbour display.
- COT: `expm1(cot_log1p)`, common linear range 0--70; the maximum over displayed snapshot fields is 57.536, so no values are clipped by that scale.
- Station centre is row8/col6. The white 3x3 box encloses rows7--9 and columns5--7, exactly matching the head crop.
- GHI display curves are means over seeds42/43/44, but annotated scores are computed per seed then averaged, matching the main-paper scoring order. The curves are not reported as a separately evaluated ensemble.
- Target times label the end of each 15-minute mean GHI interval. The complete 17-target grid is reindexed; missing GHI values/predictions are NaN and break lines. COT is displayed wherever its image exists even if a GHI scoring row is absent. Sparse valid CPP points are unconnected markers.
- Main figure GHI axes are 0--150 and 0--500 W/m² for within-event readability, and this is disclosed in the caption. Both begin at zero. The additional `current_observed_events_common_ghi_axis` export sets both to 0--550. COT axes and colour scales are identical throughout.

## Reproduction

Run from the repository root with the project's NumPy/Pandas/Matplotlib Python:

1. `fetch_validation_images.py` (only if the saved validation image file is unavailable; requires existing SSH access).
2. `inspect_candidates.py`.
3. `plot_current_events.py`.

Deliverables: `current_observed_events.pdf/png`, alternative common-GHI-axis PDF/PNG, selected raw snapshot fields, all scored event trajectories, current validation prediction table, candidate table, selection/provenance JSON, and bilingual text/captions. The PNG was visually inspected at final figure proportions; text, gaps, axes and legends are readable. No synthetic figures or interpolated weather fields are used.
