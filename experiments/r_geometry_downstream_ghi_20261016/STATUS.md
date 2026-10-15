# Execution status — 2026-10-16

## Complete

- Produced deterministic paired real-AGRI validation visuals for old-formula
  and corrected-geometry R. Cases include clear-reference minima, thick-cloud
  maxima, and largest R disagreements at each station; shared scales and
  station pixels are shown.
- Passed preflight for the exact fixed GHI rows: 51,929 total (47,666 train,
  4,263 validation), 54,895 unique station-frame keys, and 29,639 real AGRI
  frames with sidecars. Test payloads read: 0.
- Reconstructed 64 original R train/validation inputs from AGRI plus sidecars;
  both old and corrected tensors matched within `2e-5`.
- Built the paired frozen-R COT bank. All output cells were finite; the source
  rows hash matched. Against saved R validation outputs, both arms shared 1,291
  validation rows and had maximum center-3×3 differences 0.00360 (old) and
  0.00210 (corrected) in log1p COT, below the 0.05 tolerance.
- Completed all 18 GHI fits: three arms × two stations × seeds 42/43/44. All
  checkpoint CPU-reload comparisons passed. A pre-submit review fixed the
  seed initialization so each arm pair uses its own seed's shared weights.
- Independent FP64 scoring, paired 2,000-resample BJT-date bootstrap, and a
  second row-weighted FP64 recomputation all passed. The first scoring attempt
  used an overstrict raw-decimal-vs-FP32 label comparison; the scorer was
  corrected to verify the exact FP32 labels used in training, then all scores
  recomputed successfully.
- Generated and visually inspected the three-arm GHI validation figure.
  Results and claim boundaries are in [`REPORT.md`](REPORT.md).

## PBS and provenance

- `214295.tc6000`: paired R-COT bank, completed on one node22 A800-SXM4-80GB.
- `214305.tc6000`: three-arm GHI training and scoring, completed on one node22
  A800-SXM4-80GB.
- The full COT bank and checkpoints remain on the server at
  `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/r_geometry_downstream_ghi_20261016`.
  Git retains code, compact validation predictions/scores, logs, figures, and
  manifests, not the full bank or checkpoints.
- No paper manuscript or shared tracker was changed; no test payload was read.
