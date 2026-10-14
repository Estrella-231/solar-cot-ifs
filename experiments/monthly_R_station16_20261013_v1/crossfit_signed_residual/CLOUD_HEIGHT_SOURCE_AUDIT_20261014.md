# Cloud height source audit for shadow-alignment follow-up

## Finding

The training/validation source inventory points to CPP NetCDF files that
historically contain `CTH`; the monthly GHI/COT arrays used in the residual
experiment do not contain cloud height. The current packer reads CPP `COT` and
`CLP` only. It already maps the 16x16 station crop from AGRI geolocation to
rectilinear CPP `LAT/LON`, with the accepted pack reporting maximum mapping
error 0 degrees. That mapping can be reused for CTH if the source files are
readable.

The legacy CPP source audit describes CTH as 0–17.5 km, but it found no
independent product QA field and could not associate the historical NC files
with a fully verified production run/checkpoint. It also documents possible
stale-value fill behavior in a later candidate producer. Therefore CTH is only
a candidate diagnostic field until its source file, time, coordinates, finite
mask, and value range are checked for the selected train/validation cases.

## Evidence checked

- `data/pack_trainval_v1/source_coverage.csv` contains only monthly
  train/validation time keys and CPP paths; the pack status records
  `test_payloads_read=0`, 17,600 CPP-supported unique frame times, and an
  AGRI-to-CPP grid mapping maximum error of 0 degrees.
- `scripts/pack_station16_cpp_agri.py` reads CPP `COT` and `CLP` after mapping
  each station crop through `LAT/LON`; it does not read or save `CTH`.
- `docs/CPP_QUALITY_PROVENANCE_20260905.md` documents CTH's declared range and
  the lack of independent QA/provenance closure. `docs/CPP_PRODUCER_CODE_REVIEW_20260917.md`
  records unresolved historical producer and fallback questions.
- A read-only `ssh FD-107 hostname` probe timed out in this session. No raw CPP
  NetCDF was opened, copied, or inspected during this audit.

## Gate for physical projection

Do not use predicted COT as if it supplied cloud height. Once the train/val
CPP source is reachable, extract CTH with the same coordinate lookup as COT,
check finite/range/missing rates and timestamps, then select matched
correct/failed station cases. For diagnostic-only analysis, first undo
satellite parallax using the FY-4B viewing geometry and cloud height; then
project the cloud location down-sun-shadow direction using solar azimuth and
zenith, and measure whether COT-bearing pixels overlap the station's shadow
footprint. Use the same cases and fixed spatial support for unprojected and
projected estimates. CPP-derived future CTH must remain an oracle diagnostic,
not a deployable prediction input.

Current gate: **source field exists in the documented CPP schema, but
case-level CTH extraction and physical projection are blocked on raw CPP
source access and per-case quality checks.** No projected result is claimed.
