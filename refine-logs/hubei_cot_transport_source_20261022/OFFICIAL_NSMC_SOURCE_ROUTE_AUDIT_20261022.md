# Official NSMC source-route feasibility audit

**Date:** 2026-10-22
**Decision:** An official alternative route is identifiable, but the CPD product is not yet field-level verified for the frozen train/validation dates. Do not start full-resolution COT training from this metadata alone.
**Scope:** Public catalogue and service-document review only. No account login, order, product download, or test/buffer access.

## Findings

1. The official FY-4B catalogue lists an AGRI L2 product named **CPD — AGRI Cloud Particle Daytime (Full Disk)**. This is distinct in the catalogue from FY-3 MERSI/VIRR products called CPP and COT. The name is a promising route to a daytime cloud-optical-property reference, but the catalogue entry alone does not prove which COT variable, units, QA flags, time cadence, or grid are present in downloadable files.
2. The public FY-4B AGRI instrument page links to the L1/L2 product search portal and lists format specifications for several products (including CLM, CLP, CLT, CTH, CTP and CTT), but the page's product-format table does not expose a CPD format link. The exact CPD file schema and version therefore remain unverified from the accessible public documentation.
3. The NSMC data overview reports FY-4B full-disk AGRI L1 4-km coverage beginning 2022-06-01 and, in the latest indexed snapshot inspected, extending through 2026-10-14. This overlaps the frozen Hubei train/validation period (2025-08-01 to 2026-05-27) for AGRI L1; it does not establish CPD coverage over those dates or enough contiguous valid CPD scenes.
4. The official service FAQ states that satellite data are free but online orders/downloads require an audited real-name account. It documents a standard daily download allowance of 30 GB and describes different preparation times for online, nearline and offline holdings. This makes the portal a real procurement route, subject to account access and the actual CPD inventory.

## Implication for the COT forecast objective

The most efficient next path is to first query the official portal for a **small train-only date block** of FY-4B AGRI CPD and retrieve its format guide/sample metadata. Confirm variable names, physical units, fill values, retrieval/QC flags, nominal grid and time stamps before comparing it with our R-derived COT. If the CPD COT field and support are suitable, it may provide a provenance-clean reference and avoid rebuilding the CPP V2 optical-depth product for this diagnostic. If not, continue the raw-input route with time-matched AGRI L1 plus the exact atmospheric/environmental inputs required by the portable producer.

After that small source contract passes, inventory all permitted train/validation dates, build the common support and station/target alignment, then run a read-only retrieval/inference and score audit. Only after data coverage and the frozen forecast gate pass should we schedule longer training. At that point, follow the workspace multicard rule: benchmark 2/4 GPUs, expand only when measured throughput improves, keep global batch and optimizer-update count unchanged, and retain enough memory headroom. This source audit alone does not satisfy the existing G1 data-coverage gate for model/GPU work.

## Public sources

- [FY-4B product catalogue](https://satellite.nsmc.org.cn/DataPortal/en/code/FY4B.html) — lists AGRI CPD and full-disk AGRI L1 products.
- [FY-4B data overview](https://data.nsmc.org.cn/DataPortal/cn/data/dataset.html?satelliteCode=FY4B) — indexed coverage/count summary for FY-4B datasets.
- [AGRI instrument and product documentation index](https://sac347.nsmc.org.cn/nsmc/cn/instrument/AGRI.html) — instrument cadence, product portal and public format-document links.
- [NSMC data-service FAQ](https://data.nsmc.org.cn/DataPortal/cn/support/faq.html) — account, download/order and service conditions.
- [FY-4B product introduction](https://sac347.nsmc.org.cn/nsmc/cn/theme/FY4B_product.html) — general instrument/product descriptions.

## Not established

- CPD's exact COT/COD field name, unit, retrieval range, quality semantics, file format and current version.
- CPD inventory/count and valid time coverage for the frozen Hubei train/validation dates.
- CPD/AGRI/label timestamp tolerance, projection/grid transform and registration accuracy.
- Whether CPD can replace the historic CPP reference for already-frozen comparisons. Do not silently mix reference products or redefine existing gates.

Until these are checked from the product guide and actual train-only file headers, the existing full-resolution G1 source gate remains blocked. This is a newly identified route, not evidence that data have been acquired or that COT forecasting has improved.
