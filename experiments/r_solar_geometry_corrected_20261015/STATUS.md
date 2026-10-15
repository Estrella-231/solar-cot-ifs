# Status — 2026-10-15

- State: `COMPLETE`; PBS `214255.tc6000` exited 0 after 652 seconds on node22,
  one NVIDIA A800-SXM4-80GB.
- The exact archived R train/validation pack was used (15,414 train rows and
  1,398 validation rows); no test payload was accessed.
- Old-formula and corrected-formula retrains used seed 42, the same initialization
  state SHA-256 (`b09c261243832c0644f55cd313806b205edd0413e036b9382dc88f1d5d069196`),
  row order, common pixel masks, optimizer, batch, and stopping budget. The
  validation mask contains 352,441 common pixels across 46 BJT dates per station.
- Independent FP64 scoring passed on the saved predictions. Corrected minus old
  retrain equal-station RMSE difference: `+0.08608` COT, date-cluster bootstrap
  95% CI `[-0.28234, +0.43802]`; the interval includes zero. The result does
  not establish a geometry-correction gain.
- The original R checkpoint was also evaluated on that same support. Full
  metrics, cloud strata, station effects, hashes, paired predictions, and
  interpretation limits are in `RESULT_REPORT.md` and `results/paired_v1/`.
- All local and remote scoring artifacts match by prediction SHA-256. A local
  independent rerun agrees with the server score to reporting precision (the
  only differences are floating-point rounding in the last digits).
- Best checkpoints remain on the HPC and are identified by SHA-256 in each
  arm's `artifact_hashes.json`; model weights are not committed to Git.
- Scientific boundary: validation agreement with CPP retrieval output only;
  CPP is not independent COT truth. This R comparison does not measure GHI
  forecast impact and does not authorize replacing the current R in downstream
  production.
- Test payloads read: `0`.
