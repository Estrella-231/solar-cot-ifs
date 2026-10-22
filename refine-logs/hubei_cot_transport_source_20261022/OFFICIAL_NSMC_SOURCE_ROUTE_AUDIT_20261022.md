# Official NSMC source-route feasibility audit

**Date:** 2026-10-22
**Decision:** An official alternative route is identifiable, but the CPD product is not yet field-level verified for the frozen train/validation dates. Do not start full-resolution COT training from this metadata alone.
**Scope:** Public catalogue and service-document review only. No account login, order, product download, or test/buffer access.

## Findings

1. The official FY-4B catalogue lists an AGRI L2 product named **CPD — AGRI Cloud Particle Daytime (Full Disk)**, nominal 4-km NetCDF. Its product introduction says the DCOMP retrieval produces COT and CER, followed by LWP and IWP retrievals. This is distinct in the catalogue from FY-3 MERSI/VIRR products called CPP and COT. It is a plausible daytime COT retrieval reference, but the product description does not identify the downloadable COT variable, units, scaling, fill values, QA flags, cadence details, or pixel grid.
2. The full FY-4B format-document list (38 records over four pages) was inspected, as was the portal's document search. No FY-4B CPD format guide is listed; a search for “CPD” returned zero documents. The only “云光学” search result was a FY-3F MERSI guide, which is not a CPD specification. The exact CPD file schema and version therefore remain unverified from accessible public documentation.
3. The NSMC data overview reports FY-4B full-disk AGRI L1 4-km coverage beginning 2022-06-01 and, in the indexed snapshot inspected, extending through 2026-10-14. A read-only public CPD file-list query now confirms indexed listings on two frozen Hubei **train** dates: 2025-08-04 lists 95 CPD files (about 1.52 GB), and 2026-05-10 lists 95 files (about 1.45 GB). The frozen assignment has 95 AGRI frames on each date, while its old CPP counts are 49 and 0, respectively. These two date queries show that CPD listings may fill gaps in the old CPP inventory; they do not establish complete train/validation coverage, valid retrieval pixels, or contiguous support across all required dates.
4. The official service FAQ states that satellite data are free but online orders/downloads require an audited real-name account. It documents a standard daily download allowance of 30 GB and describes different preparation times for online, nearline and offline holdings. This makes the portal a real procurement route, subject to account access and the actual CPD inventory.

## Implication for the COT forecast objective

The most efficient next path is no longer another catalogue query: obtain a **small authorized train-only CPD sample** and resolve its actual NetCDF header/format. Confirm variable names, physical units, fill values, retrieval/QC flags, nominal grid and time stamps before comparing it with our R-derived COT. If its COT field and support are suitable, it may provide a provenance-clean retrieval reference and avoid rebuilding the CPP V2 optical-depth product for this diagnostic. If not, continue the raw-input route with time-matched AGRI L1 plus the exact atmospheric/environmental inputs required by the portable producer.

After that small source contract passes, inventory all permitted train/validation dates, build the common support and station/target alignment, then run a read-only retrieval/inference and score audit. Only after data coverage and the frozen forecast gate pass should we schedule longer training. At that point, follow the workspace multicard rule: benchmark 2/4 GPUs, expand only when measured throughput improves, keep global batch and optimizer-update count unchanged, and retain enough memory headroom. This source audit alone does not satisfy the existing G1 data-coverage gate for model/GPU work.

## Public sources

- [FY-4B product catalogue](https://satellite.nsmc.org.cn/DataPortal/en/code/FY4B.html) — lists AGRI CPD and full-disk AGRI L1 products.
- [FY-4B AGRI CPD dataset detail](https://satellite.nsmc.org.cn/dataportal/cn/support/dataset.html?code=FY4B-_AGRI--_N_DISK_1330E_L2-_CPD-_MULT_NOM_YYYYMMDDhhmmss_YYYYMMDDhhmmss_4000M_V0001.NC) — product class, level, nominal projection, resolution, format and retrieval-purpose description.
- [FY-4B data overview](https://data.nsmc.org.cn/DataPortal/cn/data/dataset.html?satelliteCode=FY4B) — indexed coverage/count summary for FY-4B datasets.
- [AGRI instrument and product documentation index](https://sac347.nsmc.org.cn/nsmc/cn/instrument/AGRI.html) — instrument cadence, product portal and public format-document links.
- [NSMC data-service FAQ](https://data.nsmc.org.cn/DataPortal/cn/support/faq.html) — account, download/order and service conditions.
- [NSMC document search](https://satellite.nsmc.org.cn/DataPortal/cn/support/searchdoc.html) — searched for a CPD format manual.
- [FY-4B product introduction](https://sac347.nsmc.org.cn/nsmc/cn/theme/FY4B_product.html) — general instrument/product descriptions.

## Not established

- CPD's exact COT/COD field name, unit, retrieval range, quality-flag semantics, file format and current version. The two inspected listing records show quality-evaluation flag `0`, but its meaning is not established.
- Complete CPD inventory and valid-pixel time coverage for every frozen Hubei train/validation date; two train-day listings are not a complete coverage audit.
- CPD/AGRI/label timestamp tolerance, projection/grid transform and registration accuracy. A concrete metadata discrepancy requires resolution: the catalogue's filename template contains `1330E`, while inspected file listings use `1050E` and report center longitude 105°E. Do not infer that either string identifies the grid correctly until the file header and projection documentation are checked.
- Whether CPD can replace the historic CPP reference for already-frozen comparisons. Do not silently mix reference products or redefine existing gates.

Until these are checked from the product guide and actual train-only file headers, the existing full-resolution G1 source gate remains blocked. This is a newly identified route, not evidence that data have been acquired or that COT forecasting has improved.

## Read-only portal and documentation follow-up (2026-10-22)

- Checked the official FY-4B format-document pages 1–4 (38 entries total) and the official document-search interface. No CPD format guide was listed; query `CPD` returned zero documents. The `云光学` query returned only a FY-3F MERSI guide, which cannot define the FY-4B CPD schema.
- Queried two dates that the frozen split file explicitly assigns to `train`: 2025-08-04 and 2026-05-10. The public CPD portal listed 95 files on each date (about 1.52 GB and 1.45 GB). The frozen inventory records 95 AGRI frames on each date, with 49 and 0 old-CPP frames respectively. These are indexed file counts only; no CPD payload was opened, ordered, or downloaded.
- An inspected CPD listing entry covers 2025-08-04 15:45–15:59 UTC, reports product quality-evaluation flag `0`, generation time 16:11:20 UTC, size 229,660 bytes, and center longitude 105°E. The catalogue template instead contains `1330E`; the naming/projection relationship and quality-flag semantics remain unresolved.
- On the CPD dataset-detail page, the `格式说明` and `质量报告` entries are blank. The product itself is described as nominal projection (NOM), 4,000 m, and NetCDF; this is product-level metadata, not enough to reconstruct the geostationary projection transform or map pixel indices to the existing AGRI grid.
- Fresh read-only `qstat -u slfu` on reachable `zjnu-hpc` (`tc6000`) at 2026-10-22 08:34:40 CST showed no unfinished jobs. This is only the user's queue state; it does not establish idle GPUs. No training job was submitted because the COT/source and grid gates remain closed.
- Portal entry points: [FY-4B CPD overview](https://satellite.nsmc.org.cn/dataportal/cn/data/dataset.html?dataTypeCode=CPD&instrumentTypeCode=AGRI&satelliteCode=FY4B), [CPD data search](https://satellite.nsmc.org.cn/dataportal/cn/data/detail.html), [FY-4B format-document index](https://satellite.nsmc.org.cn/DataPortal/cn/support/document.html#format-fy4b), and [document search](https://satellite.nsmc.org.cn/DataPortal/cn/support/searchdoc.html).

## Train-only sample acquisition check (2026-10-22)

- Reopened the official CPD listing for frozen train date 2025-08-04. The row `FY4B-_AGRI--_N_DISK_1050E_L2-_CPD-_MULT_NOM_20250804154500_20250804155959_4000M_V0001.NC` is listed for 15:45:02–15:57:57 UTC, is 229,660 bytes, and is a practical single-file schema sample. Its BJT interval is 2025-08-04 23:45–23:57:57.
- The frozen train/validation AGRI manifest contains the exact corresponding train key `202508042345` (BJT), at `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/agri_trainval_staged/202508042345.npz`. This is a concrete train-only CPD/AGRI header-and-registration witness candidate; it does not establish the satellite-product pixel mapping.
- The visible portal session is logged out. The file-list modal exposes row checkboxes, metadata, and a `去购物车` action; no direct per-file download control was exposed in the accessible row controls. I did not log in, select/order files, or download a payload. The portal FAQ's account/order requirement remains the documented acquisition path.
- Fresh `qstat -fx -u slfu` on `zjnu-hpc` at 2026-10-22 08:45:17 CST returned an empty `<Data>` result. `pbsnodes -x node22/node23` lists eight GPUs per node but also scheduler jobs 213684/213751 on node22 and 213683 on node23; the queried records do not report per-GPU idle state. No GPU is known to be allocated to this task, and no availability is inferred from the empty account queue.
- Decision: source/schema gate remains blocked until an authorized route retrieves the train-only CPD sample. Once available, inspect file header, variable/unit/fill/QC metadata and projection, then join this exact AGRI key and test registration on train-only data. Do not submit regional COT/GHI training before date-coverage and QC/grid gates pass; test and buffer remain closed.
