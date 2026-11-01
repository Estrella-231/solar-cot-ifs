"""Pure-Python exact validation sequence alignment for the DropPath-0 bank."""
from __future__ import annotations


def select_validation_rows(manifest_rows: list[dict[str, str]],
                           base_index_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str], list[int]]:
    val_rows = [r for r in manifest_rows if r.get("split") == "val"]
    by_id = {r["seq_id"]: (i, r) for i, r in enumerate(val_rows)}
    if len(by_id) != len(val_rows):
        raise ValueError("monthly validation manifest contains duplicate sequence ids")
    val_ids = [r["seq_id"] for r in base_index_rows if r.get("split") == "val"]
    if len(val_ids) != len(set(val_ids)) or not val_ids:
        raise ValueError("baseline bank validation ids are empty or duplicated")
    missing = [sid for sid in val_ids if sid not in by_id]
    if missing:
        raise ValueError(f"candidate manifest lacks {len(missing)} baseline validation sequence ids")
    selected = [by_id[sid][1] for sid in val_ids]
    manifest_indices = [by_id[sid][0] for sid in val_ids]
    if any(len(r["data_relpaths"].split("|")) != 24 for r in selected):
        raise ValueError("validation sequence must list exactly 24 source frame paths")
    return selected, val_ids, manifest_indices
