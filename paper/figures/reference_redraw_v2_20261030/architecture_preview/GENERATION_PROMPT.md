# Built-in image generation preview

## Refinement prompts

Refine the supplied scientific architecture preview. PRESERVE its excellent wide two-stage composition, all feature-block shapes, colors, labels, typography, dimensions and whitespace as closely as possible. Make only the following scientific corrections:
1. The "Solar geometry" tile group must NOT look like cloud observations. Replace cosSOZ and cosRAA image textures with clean smooth spatial gradients, without cloud shapes; use an entirely uniform pale/white day-mask plane with a crisp "1" in the center, representing daytime. Keep these tiles clearly geometric inputs, and keep their existing labels.
2. The long upper "same image (aligned 3 × 3 crop)" bypass must visibly originate from the FY-4B AGRI input stack. Give it a continuous clean thin connector rising from the upper-right of the AGRI stack, running above the R group to the upper AGRI+geometry path. It must not start floating in empty space. The bypass carries AGRI only; do not originate it from CPP or R output.
3. The dashed CPP-reference training-only supervision line should terminate beside the FULL 16 × 16 log(1+COT) OUTPUT of R, not the cropped 3 × 3 patch. Route the line cleanly with a short right-edge rise and point toward the lower edge of the full orange/indigo COT image; avoid crossing other graphics. Keep "masked Huber loss" and the CPP reference label.
4. In the COT preprocessing box, use exactly "Standardize" on line 1 and "log-COT" on line 2, to avoid implying a second logarithm after R already emits log(1+COT).
5. Label the second descending U-Net arrow "pool" and both upsampling arrows "up" if they can fit cleanly. All encoder/decoder numbers stay 32,64,128,64,32 and resolutions 16×16,8×8,4×4,8×8,16×16.
Keep the tiny "Architecture preview • schematic image tiles • Hunan single-frame implementation" disclosure. No new elements, charts, performance values or changes of architecture. Make a polished high-resolution preview, same wide aspect and framing.

### Final topology cleanup

Keep the supplied wide publication architecture artwork visually almost identical. Final topology cleanup, three changes ONLY:
A. On the long black "same image" bypass at the very top, REMOVE the arrowhead at the LEFT endpoint next to the satellite stack. Its left end is a plain source connection; retain ONLY the downward arrowhead at the right end into the AGRI + geometry path. Data flows left to right only.
B. DELETE the isolated extra 3×3 thumbnail immediately BELOW the large log(1+COT) output image of R, including its "3 × 3 (aligned)" caption. Keep the large 16×16 log-COT output. Add a clear thin orange directed connector from the RIGHT EDGE of that large COT output, down through the narrow interstage gap, then right into the small COT input tile at the left of the lower COT path. Label this connector simply "3 × 3 crop". This must visibly connect the R output to the GHI COT input. Do not cross the upper AGRI path.
C. DELETE the entire long dashed line labelled "masked Huber loss" along the bottom. Instead, keep the CPP reference box and add just one SHORT dashed arrow going UP from this box into the BOTTOM EDGE of the central retriever R group. Label it "training supervision". This is a training relation, not a forward input. Do not point CPP to any crop.
Preserve every other element, especially the smooth solar-geometry tiles, constant day-mask 1, U-Net feature blocks, two aligned GHI paths, all dimensions and final GHI multiplier. Clean, sharp, readable scientific preview. No new boxes or labels beyond those specified.

Date: 2026-10-30. User explicitly requested image-model previews, with appearance prioritized before later editable redrawing.

References: Baguan-solar arXiv:2603.14845v3, Figure 2 (PDF p.5); Chen et al., Applied Energy 352 (2023) 121979, Figure 4 (PDF p.6); our code-audited single-frame architecture (content reference).

The preview depicts the Hunan single-frame implementation, not the different four-arm historical-COT forecasting model. All cloud tiles are illustrative. It must not be used as a measured cloud case, retrieval map, model-performance result, or claim of forecast improvement.

## Initial prompt

Create a BEAUTIFUL, publication-style SCIENTIFIC MODEL ARCHITECTURE PREVIEW for a satellite irradiance paper. This must look like a refined academic methods figure, not a business flowchart or slide. Wide landscape, approximately 2.25:1 aspect, high resolution, white background, elegant serif scientific typography, exceptionally clear text and hierarchy.

REFERENCE ROLES:
Reference 1 (Baguan-solar): VISUAL COMPOSITION ONLY. Learn its wide left-to-right two-stage composition, visible satellite image stacks, thin module slabs, grouped submodules and deliberate branch joins. Do not copy its Swin, weather foundation model, NWP variables or scientific labels.
Reference 2 (Chen Applied Energy Fig4): VISUAL LOGIC ONLY: actual-looking input images, parallel encoding branches, clear concatenation followed by dense readout. Do not copy its architecture or labels.
Reference 3 (our plain diagram): SCIENTIFIC CONTENT AUTHORITY. Preserve the true single-frame architecture below, but completely redesign the visual layout. Do not reproduce its tall page or uniform word boxes.

DESIGN:
A sophisticated wide figure with two coordinated stages, separated by a fine vertical dashed boundary. Stage 1 occupies roughly 48%, Stage 2 52%. Large two-stage header on top: "(a) Cloud optical thickness retrieval" and "(b) Spatially aligned irradiance mapping". The eye follows a strong horizontal path. Use restrained powder blue for AGRI features, warm peach for COT features, lavender for learned transformations, sage for physical readout. White background, thin dark-gray strokes, no heavy shadows, no decorative clip-art, no large document title.
Use illustrated cloud-image tiles and translucent stacked feature planes with modest depth/perspective like the original papers. These are SCHEMATIC illustrations, NOT measured results: include a discreet footer "Architecture preview • schematic image tiles • Hunan single-frame implementation". No accuracy numbers, no empirical plots, no claims of improvement.

EXACT SCIENTIFIC CONTENT:
Stage 1, left-to-right:
1) A small stack of 13 satellite band image tiles, visibly cloud-like with a station cross, labeled "FY-4B AGRI" and "13 bands · 16 × 16".
2) Three small aligned geometry planes beneath, labeled "Solar geometry" and "cosSOZ · cosRAA · day mask". These join the AGRI input, giving 16 channels.
3) Show a compact actual U-shaped encoder-decoder IN THE CENTRAL STAGE, using 3D feature blocks of different heights: encoder 32 × 16 × 16 -> pool -> 64 × 8 × 8 -> pool -> bottleneck 128 × 4 × 4 -> transposed-conv up -> 64 × 8 × 8 -> transposed-conv up -> 32 × 16 × 16. Two clean skip concatenation arcs connect equal resolutions. Very short labels "32", "64", "128", "64", "32", with resolution labels below. Group title "COT retriever R". One small key below: "Block: [3 × 3 Conv – BN – SiLU] × 2".
4) A thin terminal slab "1 × 1 Conv + Softplus", then an orange/indigo COT image plane labeled "log(1 + COT)" and "16 × 16". The COT tile has a small outlined 3×3 selection centered on the same station. A subtle blue/peach nested-frame visual communicates 16×16 context -> aligned 3×3 crop.
5) Along the lower edge of stage1, tiny dashed training-only arrow from a small tile "CPP reference" to the retriever output with label "masked Huber loss". The R group is labeled "Frozen for GHI training". Keep supervision visually subordinate.

Stage 2:
Aligned 3×3 satellite and COT inputs are two parallel paths, with visibly matching station location. A thin long bypass from stage1 AGRI passes above R to the stage2 AGRI input, label "same image".
UPPER path: small stack "AGRI + geometry" / "15 × 3 × 3" -> two slim lavender Conv slabs labeled "1 × 1 Conv + SiLU" -> powder-blue feature stack labeled "32 × 3 × 3". Subtle channel label "15 → 32 → 32".
LOWER path: peach 3×3 COT tile -> short preprocessing slab "log-COT / standardize" -> single slim Conv slab "1 × 1 Conv + SiLU" -> peach feature stack "4 × 3 × 3", channel label "1 → 4".
These two feature stacks meet at a circled "C" (concatenation), then one COMBINED blue+peach stack labeled "36 × 3 × 3". The grid stays visibly 3×3. A short arrow labeled "Ordered flatten" leads to a dense readout block:
"324 → 64 → 32 → 1"
"MLP + SiLU"
"Softplus"
output label "k ≥ 0".
Then multiply with independently calculated interval mean clear-sky GHI at a small circled ×, arrow to a clean scalar output label "15-min station GHI".
Clear-sky input under that multiplier: "Mean clear-sky GHI". Small equation under output: "GHI = k × mean(GHIcs)".
One succinct line under the fusion paths: "All nine spatial positions retained". No pooling or GAP in the GHI head.

VISUAL QUALITY REQUIREMENTS:
Make the U-Net blocks and input/output tiles the prominent visuals, not giant text boxes. Balanced compact layout with generous but controlled spacing. Text around 24–35 pixels at a 3000px-wide scale, readable on a paper page. Every arrow must have a clear destination, no arrows through text, no unnecessary crossings. All needed labels must be spelled correctly. Large stage labels, medium module labels, small dimension labels. Use the reference images to make this feel like a top-tier research paper architecture diagram. Do not use ornate icons, flat generic cards for every node, portrait/tall layout, or giant explanatory paragraphs. This is a visual design preview, but the operator topology and dimensions above must be scientifically faithful.
