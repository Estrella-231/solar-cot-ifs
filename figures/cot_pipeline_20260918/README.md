# COT input-path visualizations

These three figures visualize the implemented `AC_gate` candidate using the
source contract in `scripts/cot_gated_station_head_v3.py`. They use a fixed,
synthetic 16×16 cloud/COT field for explanation only. They do not show FY-4B,
CPP, a trained gate, or experimental performance.

- `fig_cot_pipeline_overview_20260918`: frozen inference route from AGRI
  history through SimVP and COT retrieval to the station head.
- `fig_cot_patch_alignment_20260918`: same-grid 13-channel AGRI and one-channel
  COT patch, with the verified station pixel `(8,8)` and 5×5 neighbourhood.
- `fig_cot_pixel_gate_20260918`: code-level pointwise gate and readout.

Rebuild with the three `gen_fig_cot_*.py` scripts. PDF is the manuscript
candidate; PNG is a quick preview.
