#!/usr/bin/env python3
"""Independent metadata-only audit of the PBS path-remapped AGRI manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--remapped", type=Path, required=True)
    p.add_argument("--days", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-split-sha256", required=True)
    p.add_argument("--expected-rows", type=int, default=24705)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.days) != a.expected_split_sha256:
        raise ValueError("Frozen day assignment checksum mismatch")
    source, mapped, days = read_rows(a.source), read_rows(a.remapped), read_rows(a.days)
    if len(source) != a.expected_rows or len(mapped) != a.expected_rows:
        raise ValueError(f"Unexpected source/remapped lengths: {len(source)} / {len(mapped)}")
    day_split = {r["date_BJT"].replace("-", ""): r["split"] for r in days}
    if len(day_split) != len(days):
        raise ValueError("Duplicate date in frozen day assignment")
    split_counts: Counter[str] = Counter()
    files_seen: set[str] = set()
    for i, (s, m) in enumerate(zip(source, mapped)):
        if s["time_bjt"] != m["time_bjt"] or s["split"] != m["split"]:
            raise ValueError(f"Key/split changed at manifest row {i}")
        if s["split"] not in ("train", "val") or day_split.get(s["time_bjt"][:8]) != s["split"]:
            raise ValueError(f"Manifest split disagrees with frozen date assignment: {s}")
        path = Path(m["path_agri"])
        try:
            rel = path.relative_to(a.stage_root)
        except ValueError as exc:
            raise ValueError(f"Remapped path outside staged root: {path}") from exc
        if len(rel.parts) != 2 or rel.parts[0] != s["time_bjt"][:6]:
            raise ValueError(f"Staged path month disagrees with time key: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path) in files_seen:
            raise ValueError(f"Duplicate staged path: {path}")
        files_seen.add(str(path))
        split_counts[m["split"]] += 1
    result = {
        "state": "HUBEI_TRAINVAL_PBS_REMAP_INDEPENDENT_METADATA_AUDIT_PASS_TEST_CLOSED",
        "source_manifest_sha256": sha(a.source),
        "remapped_manifest_sha256": sha(a.remapped),
        "day_assignment_sha256": sha(a.days),
        "stage_root": str(a.stage_root),
        "rows": len(mapped), "unique_time_keys": len({r["time_bjt"] for r in mapped}),
        "unique_staged_paths": len(files_seen), "paths_exist": len(files_seen),
        "split_counts": dict(split_counts), "key_split_values_preserved": True,
        "date_split_assignment_matches": True, "test_or_buffer_rows": 0,
        "payload_values_read": False,
    }
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
