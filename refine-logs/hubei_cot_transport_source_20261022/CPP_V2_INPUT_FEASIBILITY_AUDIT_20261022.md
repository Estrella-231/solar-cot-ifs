# CPP V2 train/validation input feasibility audit

**Date:** 2026-10-22  
**Decision:** No full-region CPP production or model training is feasible from the currently reachable ZJNU files. This is a storage-coverage finding, not evidence that the upstream CPP product does not exist.  
**Scope:** Read-only path, file-size, and array-header inspection restricted to the existing Hubei train/validation key index. No test/buffer payloads or test keys were read.

## Frozen candidate index

`agri_paths_trainval_staged.csv` contains 22,061 train/validation rows: 20,206 train and 1,855 validation, spanning BJT `2025-08-01 00:00` through `2026-05-27 07:45`. These are the current candidate keys, not evidence of full-resolution input or CPP coverage.

The header-only inspection of the first train NPZ (`202508010000.npz`) found `patch.npy` with shape `(20, 16, 16)`, plus station geometry, station observations, masks, and metadata. It is a station-patch training sample; it is not a 4051×4051 AGRI L1 HDF and cannot be passed to the portable full-disk CPP V2 pipeline or expanded into regional targets.

## Reachable CPP V2 smoke directory

Under `runs/cpp_v2_20260917/input`, the file inventory contains exactly:

| Input | Available files | Coverage implication |
|---|---:|---|
| Raw FY4B AGRI L1 HDF | 1 (2025-08-04 06:45 UTC scene) | One scene only; not a contiguous train/validation history or full target inventory. |
| ERA5 temperature profile | 1 (2025-08-04) | Does not provide the profile/environment inputs for all candidate keys. |
| CPP V2 output | 0 files | No generated CPP product to reuse. |
| Run logs | only `pip.pid` and `pip_install.log` | No completed inference receipt. |

The input folder does not contain the other time-matched CPP V2 environment inputs. The existing static code review identifies the producer's required channels as AGRI C11–C15 and SAZ, ERA5 temperature and relative-humidity profiles at 300/500/850/1000 hPa, SKT, and IGBP LCCS; it also identifies missing per-output provenance and QA limitations. See `docs/CPP_PRODUCER_CODE_REVIEW_20260917.md`.

## Other reachable shared-copy entries

- `/public/home/slfu/ttzhou/swc/Data_Pool/data/FY/FY4B/AGRI/4KM` contains 7 HDF files, all on 2023-01-23. Their timestamps do not cover the frozen 2025–2026 candidate period.
- `/public/home/slfu/ttzhou/swc/Data_Pool/data/ERA5/profile/T` and `/R` each contain one file under 2023, not the candidate period.
- `/public/home/slfu/ttzhou/swc/Data_Pool_3/chenyi/data/FY/FY4B/AGRI/L2/CLP` exposes one 2023-01-23 CLP NC. It is not the train/validation CPP/COT inventory.
- A static 2022 global MCD12 IGBP LCCS file is present under `Data_Pool_3/yangzx/Dataset/LCCS`; this single static layer does not resolve the missing date-matched AGRI and ERA5 inputs.
- The expected `/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP` root remains absent on `zjnu-hpc`.

## Consequence and next action

The reachable files cannot produce a complete, date-matched regional CPP COT bank for the frozen train/validation keys. No GPU job was submitted. At the live check on 2026-10-22 06:16 server time (timezone not recorded), `qstat -u slfu` showed no unfinished jobs; this is not evidence that shared GPUs were idle or reserved.

Resume production only after obtaining either (a) the original train/validation CPP files plus their metadata, or (b) full-resolution time-matched AGRI L1 HDF and all required environment inputs for the candidate period. Then run a bounded train-only producer smoke, verify timestamp, QC, grid transform, QA, and provenance, and inventory validation only after the producer contract passes. For any ensuing multi-hour regional training, profile 1/2/4 GPUs early (expand to 8 only if available and faster), keep global batch and update/loss accounting fixed, and stage inputs locally once before training. Do not use this report to relax the G1 source gate or open test/buffer data.
