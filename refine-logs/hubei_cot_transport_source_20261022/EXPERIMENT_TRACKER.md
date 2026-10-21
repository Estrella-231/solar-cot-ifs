# Hubei regional COT transport/evolution tracker

Last updated: 2026-10-22

| Stage | Status | Evidence / next action |
|---|---|---|
| G0 attempt13 gate | COMPLETE, FAILED SKILL GATE | See ../hubei_solar_geometry_repair_20261021/ATTEMPT13_COT_AND_AGRI_GUARD_AUDIT_20261022.md. Do not tune on the same validation cohort. |
| G1 full-region Hubei CPP source | BLOCKED ON ACCESS | Existing corrected CPP bank is station-local 16x16. Original source path in extraction metadata is not visible on zjnu-hpc; configured source endpoints do not authenticate/connect. Restore read-only source access, then inventory train/val only. |
| G2 immutable region cache | NOT STARTED | Requires G1 PASS. |
| G3 Hubei regional R | NOT STARTED | Requires G1/G2 PASS; random init and Hubei-only sources. |
| G4 transport/source COT forecast | NOT STARTED | Requires regional R and source coverage; compare persistence, advection, forecast-AGRI-to-R, and transport-plus-residual. |
| G5 paired GHI test | NOT STARTED | Requires G4 COT skill gate. |

## Live resource snapshot

- Current live qstat query for slfu returned no listed jobs. No new PBS task was submitted in this update.
- Four A800s on node22 are the preferred candidate for a later long training, conditional on G1/G2 and live allocation availability. Source inventory itself is CPU/I/O only.
- Test and buffer payloads remain closed.
