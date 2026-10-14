# Source data and reproducibility

This audit used four validation-only FY-4B AGRI L2 CTH files, the matching Hunan monthly validation examples and the current hash-verified Hunan `grid_static.npz`. Product filenames, SHA-256 values, case-to-file mapping, split scope, and all numeric inputs are recorded in the adjacent `projection_receipt.json` and `cot_shadow_validation_case_payload.json`.

The original full-disk NetCDF CTH products are retained in the working copy under `official_cth/` but are intentionally excluded from Git to avoid publishing tens of megabytes of upstream satellite data. The exact 128×128 CTH/DQF/geolocation extracts used by the projection script and compact validation payload are included alongside the report for replay. The authoritative product format is [FY-4B AGRI L2 CTH 4 km](https://img.nsmc.org.cn/PORTAL/NSMC/DATASERVICE/DataFormat/FY4B/Data/Format/FY-4B_AGRI_L2_CTH_4000M_V1.0.1.pdf).

The validation payload contains actual C03/C02/C01 image crops, geometry sidecars and the four corresponding GHI diagnostic records. It contains no test rows. No credentials are stored here.
