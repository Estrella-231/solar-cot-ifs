# Image-generated visual previews

These assets were produced with the built-in image-generation tool, then inspected for scientific consistency. They are visual-review assets rather than evidence of measured performance.

## Selected assets

- `architecture_final.png`: Hunan single-frame retrieval and aligned GHI mapping. Illustrative feature/image tiles; verified architectural dimensions. The overhead bypass was removed to eliminate an ambiguous arrowhead; the shared observed AGRI source is specified explicitly in text.
- `causal_workflow_final.png`: causal AGRI forecasting, frozen retrieval, four matched GHI comparisons, and the right-closed 15-minute target interval. Shared architecture is distinguished from the shared frozen historical-COT base. Schematic tiles, no measured results.
- `cot_quality_placeholder.png`: empty CPP-reference retrieval-quality panels. No scatter points or numerical scores are supplied.
- `synchronous_gain_placeholder.png`: empty five-station same-time GHI comparison and paired-effect panels. No bar heights, effect points, or uncertainty intervals are supplied.
- `strata_placeholder.png`: empty regional thickness-stratified error/effect panels. No subgroup measurements are supplied. The paper specifies the pre-existing stratum definitions.

## Reproduction and scope

Full generation prompts and targeted correction prompts are saved alongside the files. Initial/intermediate architecture and workflow files are retained for audit but are not selected for the paper. Do not treat image textures in the two diagrams as satellite observations. Genuine observational/forecast tiles and numerical figures are stored in the sibling `real_case`, `source_replacement`, and `probe` directories and are produced with data-driven plotting scripts.

The style was informed by original method and evaluation figures previously inspected in Chen et al., Applied Energy 352 (2023), 121979; Papachristopoulou et al., AMT 17 (2024), 1851–1877; and the user-provided satellite/weather-model architecture example. Original screenshots and the design analysis remain under `paper/reviews/20261030_figure_atlas` and `paper/reviews/20261030_reference_redraw`. No reference paper's measured figure is passed off as this project's result.
