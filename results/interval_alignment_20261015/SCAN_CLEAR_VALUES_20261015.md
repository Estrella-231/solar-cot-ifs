# Actual-scan clear-sky values for interval readout

Computed from the frozen 54,461-row train/validation interval manifest. For each of the three source frames, the observation time is the nominal UTC frame time plus the station-specific `obs_time_offset_s` already bound to its true HDF row. Clear-sky GHI uses the archived pvlib 0.9.1 Ineichen implementation, the same station coordinates and 0 m altitude contract as the existing label denominator.

The independent reconstruction of the existing 15-minute denominator averages Ineichen GHI at target time −12.5, −7.5 and −2.5 minutes. Its maximum absolute difference from saved `clear` is `4.999833436158951e-9 W m-2` (99th percentile `4.9522626e-9 W m-2`). This confirms the pvlib model, site coordinates, timezone conversion and interval convention used for the new per-scan `C_j` values.

| split/station | rows |
|---|---:|
| train/Sili | 23,936 |
| train/Zhujia | 26,062 |
| validation/Sili | 2,124 |
| validation/Zhujia | 2,339 |

The per-scan clear-sky range is 0–956.04 W m⁻². Zero values can occur for a frame scanned before sunrise; the sample is accepted only when its weighted sum `Σ w_j C_j` is positive. No value is filled with 1 or replaced by a cosine proxy. This is input preparation, not a GHI result. Test payloads were not read.

Artifacts: `scan_clear_values.csv`, `scan_clear_values.audit.json`, `tensor_pack_submission.json`. PBS `214221.tc6000` is building the three-frame tensors and train-only QC1 normalization; training has not started.
