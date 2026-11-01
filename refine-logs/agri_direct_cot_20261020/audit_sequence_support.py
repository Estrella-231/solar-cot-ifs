#!/usr/bin/env python3
"""Freeze metadata-only 24-frame U/H/V support lists for the direct-COT screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_selected(path: Path, splits: set[str]):
    result = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits:
                continue
            paths = row["data_relpaths"].split("|")
            if len(paths) != 24 or len(paths) != len(set(paths)):
                raise ValueError(f"bad 24-frame path support: {row.get('seq_id')}")
            result.append((row, paths))
    if not result:
        raise ValueError(f"no selected rows in {path}")
    return result


def read_lines(path: Path) -> set[str]:
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-manifest", type=Path, required=True)
    p.add_argument("--u-frames", type=Path, required=True)
    p.add_argument("--h-frames", type=Path, required=True)
    p.add_argument("--outer-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    urows = read_selected(a.fold_manifest, {"train"})
    hrows = read_selected(a.fold_manifest, {"val"})
    vrows = read_selected(a.outer_manifest, {"val"})
    u_frames, h_frames = read_lines(a.u_frames), read_lines(a.h_frames)
    if u_frames & h_frames:
        raise ValueError("fold1 U/H official frame lists overlap")

    for label, rows, allowed in (("U", urows, u_frames), ("H", hrows, h_frames)):
        bad = [(r["seq_id"], sorted(set(paths) - allowed)[:3]) for r, paths in rows if not set(paths) <= allowed]
        if bad:
            raise ValueError(f"{label} 24-frame window escapes its official frame list: {bad[:3]}")

    def support(rows):
        all_frames = set()
        targets = set()
        for _, paths in rows:
            all_frames.update(paths)
            targets.update(paths[8:])
        return all_frames, targets

    us, ut = support(urows)
    hs, ht = support(hrows)
    vs, vt = support(vrows)
    intersections = {
        "u_h_support": len(us & hs), "u_v_support": len(us & vs), "h_v_support": len(hs & vs),
        "u_v_targets": len(ut & vt), "h_v_targets": len(ht & vt),
        "u_h_targets": len(ut & ht),
    }
    if any(intersections.values()):
        raise ValueError(f"24-frame/target support overlap found: {intersections}")

    a.output.mkdir(parents=True, exist_ok=False)
    selected = {"U": urows, "H": hrows, "V": vrows}
    summaries = {}
    for label, rows in selected.items():
        out = a.output / f"{label.lower()}_sequences.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=("seq_id", "BJT_start", "split", "data_relpaths"))
            w.writeheader()
            for row, _ in rows:
                w.writerow({k: row[k] for k in w.fieldnames})
        sf, st = support(rows)
        summaries[label] = {
            "sequence_rows": len(rows), "unique_support_frames": len(sf), "unique_future_target_frames": len(st),
            "sequence_manifest_sha256": sha(out),
        }
    report = {
        "state": "PASS_24_FRAME_SUPPORT_DISJOINT_U_H_V",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1_outer_with_fold1_inner_date_group_screen",
        "fold_manifest_sha256": sha(a.fold_manifest), "outer_manifest_sha256": sha(a.outer_manifest),
        "u_frame_list_sha256": sha(a.u_frames), "h_frame_list_sha256": sha(a.h_frames),
        "sets": summaries, "intersections": intersections,
        "support_policy": "metadata-only; each selected sequence has 8 history + 16 future frame paths; U/H support checked against official fold frame lists; V consists only of outer validation rows",
        "test_or_buffer_numeric_payloads_opened": 0,
        "test_rows_selected": 0,
        "performance_scoring": False,
    }
    (a.output / "support_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
