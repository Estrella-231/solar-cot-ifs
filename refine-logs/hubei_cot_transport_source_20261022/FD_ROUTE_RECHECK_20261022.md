# Read-only CPP source route recheck

Date: 2026-10-22 (local workstation)

## Scope

Recheck whether the historical Fudan source route can currently run a harmless remote command, and whether the reachable ZJNU login exposes the original CPP roots. No product payload, train/validation array, test payload, or buffer payload was opened. No PBS job was submitted.

## Observations

| Route | Result |
|---|---|
| `FD-201` | SSH connected to `10.158.205.201:22`, then the remote closed the connection before `hostname` or any filesystem command ran. |
| `FD-107` | SSH connected to `10.158.205.107:22`, then the remote closed the connection before `hostname` or any filesystem command ran. |
| `zjnu-hpc` | Read-only command succeeded; remote hostname is `tc6000`. Neither `/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP` nor `/public/home/slfu/ttzhou/swc/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP` exists there. |

The ZJNU path checks establish only that those two candidate roots are absent on `tc6000`; they do not establish that CPP is absent on the Fudan storage. The Fudan results establish only that no remote command could be executed from this workstation at this time.

## Decision

Keep regional-source gate G1 blocked. Do not substitute the 16x16 station sidecar for full-resolution CPP, do not fabricate missing CPP, and do not launch regional COT training until the read-only source route or a verified train/validation transfer is available. The next useful change is restored Fudan SSH or a documented source export to the reachable workspace. Test and buffer remain closed.
