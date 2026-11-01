# Pastel journal-style method illustrations

Generated using the native image_gen tool. User reference supplies style only; current main.tex supplies the architecture. Earlier main illustrations and supplementary historical diagrams are retained.

## Estimation figure

Edit image 1 into a polished, restrained journal architecture diagram, taking VISUAL STYLE from image 2 (user reference) and image 3 (our earlier supplementary diagram). Image 1 is the current method; images 2 and 3 provide style ONLY: do not copy their methods, names, counts or equations.

Visual design: wide landscape about 2:1, high resolution, white background, Times New Roman-like serif typography, small bold panel headers rather than a huge title. Thin dark grey outlines and precise black right-angle arrows. Pale desaturated slate-blue, lavender, cream and sage module fills. Very subtle tensor depth only where needed, no glossy cuboids, no glowing outlines, no large sun cartoons, no banner or numbered circles. Dense but readable technical information, tidy grouping, generous clear margins. Rich schematic satellite textures only in the input/output tiles. Two vertical sections separated by a thin dashed line, in the style of the user reference.

Left 44%, heading "(a) Framewise COT retrieval":
A compact three-layer stack of illustrated AGRI cloud tiles labelled "Observed AGRI", "13 bands · 3 frames", with times "t − 30", "t − 15", "t". Each tile represents a 16 × 16 crop, not a temporal prediction.
Separate pale cream box "Solar geometry" with small text "cosSOZ · cosRAA · day mask".
Both feed a compact U-shaped network labelled "Frozen COT retriever R". Exactly TWO downsampling steps 16 × 16 -> 8 × 8 -> 4 × 4, and TWO upsampling steps back 8 × 8 -> 16 × 16, with two matching encoder-decoder skip arrows. Do not label feature widths, since regions differ.
Output stacked purple-gold cloud tiles labelled "log(1 + COT)", "16 × 16 · each frame". Mark a station-centred 3 × 3 crop with a fine dashed square and station cross. Beneath: "Station-centred 3 × 3 crop".
Represent a bypass arrow from observed AGRI and solar geometry toward the local input assembly on the right. COT also feeds that assembly. All three inputs must be visibly connected.

Right 56%, heading "(b) Three-frame irradiance estimation":
Small three-frame aligned input stack, label "[3, 17, 3, 3]" and "AGRI + geometry + standardized log-COT".
Below the stack in small legible type: "13 AGRI + 2 geometry + 1 COT + 1 zero slot".
Arrow to a compact chain inside a pale blue group: "Temporal Conv3D" / "17 → 32; kernel 3 × 3 × 3", then "Spatial Conv2D" / "32 → 32; kernel 3 × 3", then "Ordered flatten + MLP" / "288 → 64 → 1".
Can use a two-row layout within this right panel to keep labels readable, with clear arrows.
Then output box "Clear-sky index k" / "softplus".
Arrow to a multiplication circle ×, also receiving a separate small sage box "Interval-mean clear-sky GHI".
Output small sage/cream box "Station GHI" / "15-minute mean". This is a station scalar, not a regional map.
Narrow bottom note: "Matched comparison: COT slot retained / set to zero".
Tiny grey footer "Schematic cloud fields".
Keep scientific logic exact. No residual branch, no pooling, no forecast imagery in this figure, no performance numbers. Avoid excessive text wrapping; make all words and arrows crisp.

## Forecast application

Edit image 1 into a restrained, beautiful academic workflow figure matching style image 2 (user reference) and image 3 (our earlier supplementary figure). Images 2 and 3 are STYLE references ONLY; their architecture is obsolete or unrelated and must not be copied.

Visual style: wide landscape roughly 2:1, high resolution, white background, serif scientific typography similar to Times New Roman, thin dark outlines, fine black orthogonal arrows, pale desaturated blue/lavender/cream/sage boxes, restrained slightly offset stacks of cloud-map tiles. No glossy cuboids, no shadows, neon or huge headline. Use compact bold subpanel titles and align all boxes. Preserve legible scientific time annotations.

Top 62%: "(a) Forecast application".
A left-to-right causal pipeline:
"Observed AGRI" stack, "8 frames: t₀ − 105 … t₀" -> pale blue block "Image predictor" / "DyDiff / SimVP" -> blue-edged "Predicted AGRI" stack, "16 frames: +15 … +240 min".
Split future imagery into two clearly visible paths: upper direct path "AGRI"; lower path through pale lavender "Frozen retriever R" -> lavender-edged "Predicted COT" stack.
A cream "Frame-time solar geometry" box must have arrows to R AND the aligned input assembly.
The AGRI direct path and COT path join a compact "Aligned local inputs" / "3 frames · 3 × 3" assembly.
Then pale blue box "Frozen GHI model" / "Trained on observed imagery" -> sage output "Station GHI" / "15-minute mean".
The top pipeline should accommodate this chain without overlapping arrows, using two internal rows if needed.
Critically add a thin bypass from the left observed history, below the predictor, through a small box "Historical AGRI + retrieved COT", to the aligned input assembly. Label this bypass "For +15 and +30 min". This is used only for historical positions; future observed imagery must not enter.
Do not show CPP reference, future observations, residual correction, retraining or GHI observations in any input.

Bottom 38%: "(b) Three-frame alignment".
Make a neat three-row matrix of lightly stacked map cards or simple colour-bordered field tiles. Blue border denotes Observed; lavender denotes Predicted. Keep all field textures subdued, labels dark and very readable. Each row labelled forecast lead:
"+15 min": cards "t₀ − 15" [Observed], "t₀" [Observed], "t₀ + 15" [Predicted].
"+30 min": cards "t₀" [Observed], "t₀ + 15" [Predicted], "t₀ + 30" [Predicted].
"+45 min": cards "t₀ + 15" [Predicted], "t₀ + 30" [Predicted], "t₀ + 45" [Predicted].
Small legend at right "Observed history" with pale blue swatch, "Predicted fields" with pale lavender swatch.
Annotation alongside last row only: "All three frames predicted from +45 min".
Bottom short line: "AGRI, COT and geometry share the same frame times."
Tiny grey footer "Schematic cloud fields".
Preserve exact times and arrow meanings; physical output is one station scalar. Beautiful compact journal layout, not a presentation slide.

## Review corrections

Estimation v1: correct ambiguous U-Net internal dimensions to five-stage schematic and a single 16×16 input/output label; connect geometry explicitly to both R and GHI; make all COT tiles COT textures.

Forecast v1: reconnect future R input from predicted AGRI rather than observed history; give geometry its own separate connection to the GHI input assembly. Existing historical bypass and three lead rows retained.
