#!/usr/bin/env python3
"""Create one immutable OpenSTL config for a frozen Hubei OOF S fold.

The fold manifests and normalization files are prepared on the server. This
helper deliberately changes only the index path and stage root in the already
audited strict-S config; all model/training hyperparameters remain identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--fold-index", type=Path, required=True)
    p.add_argument("--stage-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    a = p.parse_args()

    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    if not a.base_config.is_file() or not a.fold_index.is_file():
        raise FileNotFoundError("Base config or fold index is missing")

    with a.fold_index.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["split", "data_paths", "aux_paths"]:
            raise ValueError(f"Unexpected index columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows or any(r["split"] not in {"train", "val"} for r in rows):
        raise ValueError("Fold S index must contain train/val rows only")
    if sum(r["split"] == "val" for r in rows) != 407:
        raise ValueError("Fold S must retain the fixed 407-row validation set")
    for row in rows:
        data_paths, aux_paths = row["data_paths"].split("|"), row["aux_paths"].split("|")
        if len(data_paths) != 24 or len(aux_paths) != 24:
            raise ValueError("Every indexed sample must have 24 data and aux frames")
        if any(not Path(x).is_absolute() for x in data_paths + aux_paths):
            raise ValueError("All indexed payload paths must be absolute")

    text = a.base_config.read_text(encoding="utf-8")
    old_root = 'data_root = "/tmp/zjnu_hubei_simvp_stage_strict24_same_day_20261016"'
    old_index = 'index_csv = data_root + "/index_trainval.csv"'
    if text.count(old_root) != 1 or text.count(old_index) != 1:
        raise ValueError("Base config no longer matches the audited strict-S template")
    text = text.replace(old_root, f'data_root = {a.stage_root!r}')
    text = text.replace(old_index, f'index_csv = {str(a.fold_index)!r}')
    text += (
        f"\n# OOF fold {a.fold}; generated from the frozen strict-S template.\n"
        f"# fold_index_sha256={sha256(a.fold_index)}\n"
        f"# base_config_sha256={sha256(a.base_config)}\n"
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(text, encoding="utf-8")
    print(
        f"fold={a.fold} rows={len(rows)} train={sum(r['split']=='train' for r in rows)} "
        f"val=407 index_sha256={sha256(a.fold_index)} "
        f"base_config_sha256={sha256(a.base_config)} output={a.output} "
        f"output_sha256={sha256(a.output)}"
    )


if __name__ == "__main__":
    main()
