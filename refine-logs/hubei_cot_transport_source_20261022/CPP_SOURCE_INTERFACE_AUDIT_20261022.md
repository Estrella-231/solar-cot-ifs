# Hubei CPP source interface audit

**State:** local interface audit complete; full-region raw-source inventory remains blocked before authentication. No CPP payload, test payload, or buffer payload was opened in this audit.

## What the current station-patch pipeline actually carries

The G1S cache builder, `station_patch_pilot_v1/build_station_patch_cache.py`, reads an existing station-local CPP sidecar. Its required arrays are `time_bjt`, `split`, `cot`, `cot_valid_mask`, `station_local_centers`, and `station_names`; the COT arrays must be shaped `(n_frames, 16, 16)`. The companion CSV supplies the expected UTC basename/path and valid-COT/valid-CLP counts. The builder verifies BJT = UTC + 8 h, filename timestamp, split, station order, center pixels, and the 16×16 shape.

It then writes `cpp_cot.npy` and `cpp_valid.npy` alongside historical R-COT and deterministic geometry arrays. The training cache does **not** contain the raw CPP CLP field, cloud-top height/pressure, full-resolution geolocation, or raw CPP quality variables. A valid-CLP pixel count in the CSV is only a count; it is not a CLP field. Therefore this sidecar supports the bounded station-neighborhood pilot but cannot be expanded into a 256×256 regional product or a cloud-height/parallax analysis.

## Live source-route check

The desktop SSH configuration has an entry named `FD-201` pointing to `10.158.205.201` with user `zjnu` and the configured `id_ed25519_fd201` identity. The lowercase `fd201` name used in the earlier probe is not the configured alias. Retrying with the exact alias resolved and opened TCP port 22, but the remote closed the connection during SSH key exchange (`kex_exchange_identification: Connection closed by remote host`) before authentication or any remote command. Thus the source was not reached, and this result says nothing about whether CPP files exist or are readable on the server.

The configured `FD-107` alias was also tested as a read-only fallback; SSH reported `Connection closed by 10.158.205.107 port 22` before the command ran. This provides no evidence about its mount contents either.

On the follow-up route check, the additional configured aliases `FD-172` (`10.158.205.122`) and `FD-248` (`10.158.205.248`) were also probed in batch mode. Both connections closed before `hostname`, identity, or filesystem commands ran. These two routes therefore provide no evidence about CPP availability or their mounts; they are recorded as currently unreachable from this workstation.

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
