# Future-COT oracle ceiling diagnostic (2026-10-20)

## Question

Does verified future COT information improve the existing station GHI head when the future AGRI images are observed rather than extrapolated? This is a localization diagnostic for the current forecast-COT bottleneck, not a deployable forecast experiment.

## Fixed data and input contract

- Keep the five-fold date-block OOF SimVP forecast-AGRI GHI pack and its exact train/validation keys, GHI labels, AGRI, and solar geometry.
- Build an oracle copy by replacing only context COT slots whose context lead is greater than zero. Each replacement is the same-station, same-split frozen-R COT retrieved from observed AGRI at that exact context timestamp. Keep history COT unchanged.
- Use the frozen-R full train/validation frame bank, not just its compact GHI-labelled subset, because context COT frames need not themselves have a GHI label. The bank's `frame_keys.csv` supplies exact same-split frame times; the compact labelled R pack is used as an exact bank-index and centered-3×3 crop witness. Fit COT normalization on the constructed oracle training input only. Do not read test or buffer payloads; do not use CPP as model input.
- Observed future AGRI is used only to derive oracle COT. Mark all outputs `oracle_only=true`, `deployable=false`; never describe these as operational forecast gains.

## Paired training

Train A (no COT), AH (history COT only), and AC (history plus oracle future COT) with the unchanged `IntervalHead`, samples, labels, 30 epochs, batch 1792, optimizer, learning-rate schedule, and validation checkpoint rule. For each seed and station, arms share initialization and row order. Use seeds 42, 43, and 44, one isolated process per allocated A800, concurrently within one PBS job requesting three GPUs.

## Audit and decision

Require exact equality of `(station, init_time, target_time, lead, GHI row)` and truth across arms and seeds. Recompute RMSE from saved predictions, report station and lead groups including +15–60 min, and bootstrap the fixed 52 BJT target-date blocks jointly across stations and seeds. Primary diagnostic is AC−AH; also report AC−A. Negative differences favor COT.

- If oracle AC robustly improves over AH, future-COT information can help the head; prioritize improving SimVP→R COT fidelity/alignment and evaluate a deployable OOF forecast-COT method next.
- If oracle AC does not improve over AH, stop tuning COT forecast calibration on this validation set and investigate GHI-head representation, time/label alignment, and the physical meaning of the current 3×3 COT input.

The oracle result cannot pass the deployable COT gain gate and will not be added to the paper as forecast performance.

## Implementation note

The first pack attempt (`215086`) stopped in its pre-training guard after 27 seconds because the compact GHI-labelled R pack lacks some valid temporal context frames (first missing key: Sili, train, 2024-04-01 22:00 UTC). No GHI model trained. The v2 builder uses the complete 55,733-frame frozen-R bank and checks that all compact-pack bank indices and 3×3 crops match exactly before using it for context lookup. The incomplete v1 artifact is retained and excluded from scoring.
