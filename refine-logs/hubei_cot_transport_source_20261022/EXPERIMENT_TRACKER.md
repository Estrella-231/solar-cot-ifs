# Hubei regional COT transport/evolution tracker

Last updated: 2026-10-22

| Stage | Status | Evidence / next action |
|---|---|---|
| G0 attempt13 gate | COMPLETE, FAILED SKILL GATE | See ../hubei_solar_geometry_repair_20261021/ATTEMPT13_COT_AND_AGRI_GUARD_AUDIT_20261022.md. Do not tune on the same validation cohort. |
| G1 full-region Hubei CPP source | BLOCKED ON FD-201 ROUTE | Historical FD-201 read/extraction on 2026-10-20 proves the raw CPP source was readable for 3,479 corrected-UTC train/val keys, but only a 16x16 station-local sidecar was exported. Current reachable ZJNU `tc6000` does not mount the raw root; the current desktop SSH config cannot resolve `fd201`. Re-establish the prior read-only route and inventory the wider train/val AGRI index. |
| G1P station-patch sequence support | COMPLETE; NOT ENOUGH FOR TRAINING | Current full-support sidecar gives 764 overlapping train and 59 validation 8→16 windows; validation spans only 3 initialization dates (15 hourly-stride / 4 non-overlap windows). Do not train/select a 4h model on this sparse validation support. The 22,061-row train/val AGRI index has 9,673/1,022 potential windows up to the last known CPP time, but raw CPP file/QC/grid coverage for those keys remains unverified. See `CPP_LOCAL_SEQUENCE_SUPPORT_AUDIT.md`. |
| G2 immutable region cache | NOT STARTED | Requires G1 PASS. |
| G3 Hubei regional R | NOT STARTED | Requires G1/G2 PASS; random init and Hubei-only sources. |
| G4 transport/source COT forecast | NOT STARTED | Requires regional R and source coverage; compare persistence, advection, forecast-AGRI-to-R, and transport-plus-residual. |
| G5 paired GHI test | NOT STARTED | Requires G4 COT skill gate. |

## Live resource snapshot

- Current live `qstat -u slfu` query at 2026-10-22 01:44:08+08 returned no listed jobs. No new PBS task was submitted in this update.
- The train/val AGRI key index and sequence-support audits are CPU-only. Four A800s on node22 remain the preferred candidate for a later long training only after raw CPP support and validation gates pass; profile worker counts and verify allocation before use.
- Test and buffer payloads remain closed.
