# Hubei CPP source interface audit

**State:** local interface audit complete; full-region raw-source inventory remains blocked before authentication. No CPP payload, test payload, or buffer payload was opened in this audit.

## What the current station-patch pipeline actually carries

The G1S cache builder, `station_patch_pilot_v1/build_station_patch_cache.py`, reads an existing station-local CPP sidecar. Its required arrays are `time_bjt`, `split`, `cot`, `cot_valid_mask`, `station_local_centers`, and `station_names`; the COT arrays must be shaped `(n_frames, 16, 16)`. The companion CSV supplies the expected UTC basename/path and valid-COT/valid-CLP counts. The builder verifies BJT = UTC + 8 h, filename timestamp, split, station order, center pixels, and the 16×16 shape.

It then writes `cpp_cot.npy` and `cpp_valid.npy` alongside historical R-COT and deterministic geometry arrays. The training cache does **not** contain the raw CPP CLP field, cloud-top height/pressure, full-resolution geolocation, or raw CPP quality variables. A valid-CLP pixel count in the CSV is only a count; it is not a CLP field. Therefore this sidecar supports the bounded station-neighborhood pilot but cannot be expanded into a 256×256 regional product or a cloud-height/parallax analysis.

## Live source-route check

The desktop SSH configuration has an entry named `FD-201` pointing to `10.158.205.201` with user `zjnu` and the configured `id_ed25519_fd201` identity. The lowercase `fd201` name used in the earlier probe is not the configured alias. Retrying with the exact alias resolved and opened TCP port 22, but the remote closed the connection during SSH key exchange (`kex_exchange_identification: Connection closed by remote host`) before authentication or any remote command. Thus the source was not reached, and this result says nothing about whether CPP files exist or are readable on the server.

The configured `FD-107` alias was also tested as a read-only fallback; SSH reported `Connection closed by 10.158.205.107 port 22` before the command ran. This provides no evidence about its mount contents either.

On the follow-up route check, the additional configured aliases `FD-172` (`10.158.205.122`) and `FD-248` (`10.158.205.248`) were also probed in batch mode. Both connections closed before `hostname`, identity, or filesystem commands ran. These two routes therefore provide no evidence about CPP availability or their mounts; they are recorded as currently unreachable from this workstation.

A read-only TCP reachability probe from `tc6000` to port 22 on all four configured `10.158.205.*` endpoints timed out. No downstream SSH authentication or filesystem command was attempted. This narrows the current blocker to network reachability from the reachable compute host as well; it still does not imply that CPP is absent upstream.

The reachable ZJNU login remains a separate machine and does not mount the original `/home/Data_Pool_3` source root. The last successful historical extraction from FD-201 proves that source was readable in that earlier session for 3,479 train/validation timestamps, but those exported data are only station-local 16×16 crops.

## Minimum raw-source inventory needed to unlock G1

When the FD-201 SSH service permits a session, perform a read-only inventory for the frozen train/validation candidate manifest only:

1. For each candidate key, derive the CPP UTC timestamp from BJT − 8 h, check the exact source path, file size, and checksum, and record missing files without substituting another time.
2. Read NetCDF headers and relevant variables only; record product/build version, variable names, dimensions, units, fill values, coordinate metadata, and time attributes. Identify COT, CLP, native quality flags, and any cloud-top property separately.
3. For valid CPP fields, count finite/in-range pixels and QC pass fractions over the complete Hubei 256×256 crop and at all three station neighborhoods. Keep per-frame and per-date support counts; do not fill missing retrievals as clear sky.
4. Match CPP geolocation to the Hubei `grid_static` coordinates, validate orientation, offsets, resolution, and known station pixels with a handful of raw-pixel witnesses, then verify the full inventory. Preserve source-to-crop index transforms and hashes.
5. Verify contiguous 8-history + 16-future sequences independently within train and validation, and publish immutable split-separated caches plus missing-key, quality-support, mapping, and hash manifests. Never access test or buffer payloads.

Cloud-height projection is a separate later calculation. It requires an identified and quality-screened height/pressure field, satellite viewing geometry, solar geometry, and explicit coordinate transforms. The existing COT sidecar contains none of those height inputs; no parallax or cloud-shadow correction can be inferred from it.

## Decision

Keep full-region G1 blocked until a read-only source session succeeds and the inventory above passes. Do not train a regional R/transport model or start GHI-head experiments from the 16×16 sidecar. The station-patch contextual same-R diagnostic is complete separately; it does not release this source gate.

## Follow-up read-only inventory on ZJNU (2026-10-22)

- Fresh `FD-201` and `FD-107` probes again closed before a remote command could run. On `zjnu-hpc`, `qstat -u slfu` returned no jobs; this is queue evidence only and does not prove any GPU is free.
- The reachable pool `/public/home/slfu/ttzhou/swc/Data_Pool_3` exists. Within its FY-4B tree, the only visible branch is `AGRI`; under `AGRI/L2`, the only product branch is `CLP/2023/202301`. It contains one filename-visible 4-km CLP file for 2023-01-23 06:30 UTC. A recursive filename search found zero files with `CPP` in their names.
- The inventory used directory entries and filenames/sizes only; no NetCDF variables or test/buffer payloads were opened. This one old CLP file is neither a CPP COT reference field nor date-complete for the frozen 2025–2026 Hubei train/validation manifest. It cannot satisfy G1 or replace the CPP sidecar.
- **Decision unchanged:** full-region source gate remains blocked on the original CPP route. No training job was submitted and no GPU was used. Continue source-route recovery; do not infer upstream CPP absence from this ZJNU mount.

### Route recheck after CPD sample identification (2026-10-22 08:50 CST)

- Workstation SSH to the configured `FD-201` alias again closed at `10.158.205.201:22` before any remote command executed.
- From `zjnu-hpc`, TCP port 22 on `10.158.205.201` again timed out. This confirms the route is still unavailable from both current paths; it does not establish whether the source archive exists.
- No CPP/CPD payload was read, no authentication attempt or order was made, and no PBS/GPU task was submitted. The official train-only CPD sample identified in `OFFICIAL_NSMC_SOURCE_ROUTE_AUDIT_20261022.md` remains the next source artifact needed.
