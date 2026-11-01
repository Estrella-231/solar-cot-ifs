# Journal-style method illustrations

User requested the supplied pastel architecture-reference style or the historical supplementary-figure style. Two current main-text method figures are restyled using native image generation. The historical diagrams supply style only; their single-frame and residual-head structures are not reintroduced.

- White background; muted blue, lavender, cream and sage; thin dark outlines; compact serif panel headings; stacked illustrative cloud fields.
- Figure 2 retains framewise 16×16 retrieval, aligned 3×3 local fields and the current [3,17,3,3] GHI tensor; 17→32 temporal convolution, 32→32 spatial convolution and 288→64→1 readout.
- Figure 3 retains 8 observed→16 forecast imagery, frozen R and observed-image-trained frozen GHI model. The first two leads explicitly receive historical fields. From +45 minutes all three input frames are forecast.
- No experimental numbers, data figures, training models or evaluation samples are modified.
- English and Chinese consume the same two assets. Image textures remain schematic, not actual cloud-event evidence.

Prompts: ../../figures/journal_style_20261102/PROMPTS.md
Earlier generated_current_20261102 figures are preserved. This is an in-place update of main.tex/main_zh.md, not a new manuscript.

## Review and build

- Initial generation contained an incorrect internal R size and a future-R source arrow connected to history; both were corrected with native image editing before integration. Initial images remain as *_v1.png.
- Independent visual review accepted both current schematics. The U-shaped icon is intentionally not an exact layer drawing. A redundant unheaded geometry connector in the forecast schematic is cosmetic and can be simplified during final editable redraw.
- Built-in LaTeX compilation succeeded; two local pdflatex passes succeeded. Main PDF remains 27 pages; method figures are on pages 5 and 7. Rendered pages inspected, no undefined references or overfull boxes.
- Generated cloud textures are still labelled schematic. This update does not claim empirical cloud fields or new experiments.
