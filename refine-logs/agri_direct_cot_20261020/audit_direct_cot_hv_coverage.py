#!/usr/bin/env python3
"""Metadata-only coverage audit for direct-COT H/V forecast GHI rows.

Reads only frozen sequence manifests, forecast-bank indices, and label-key CSVs.
It never opens numeric arrays or test/buffer paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_map(path: Path) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {"train": {}, "val": {}}
    for r in read_csv(path):
        split = r["split"]
        if split not in out:
            continue
        i = int(r["sequence_index_within_split"])
        if i in out[split]:
            raise ValueError(f"duplicate {split} sequence index {i}")
        out[split][i] = r["seq_id"]
    return out


def audit(support: Path, sequence_index: Path, labels: Path,
          old_fold1_index: Path, old_fold1_manifest: Path,
          exclusions_path: Path) -> dict:
    h_rows = read_csv(support / "h_sequences.csv")
    v_rows = read_csv(support / "v_sequences.csv")
    h_ids = {r["seq_id"] for r in h_rows}
    v_ids = {r["seq_id"] for r in v_rows}
    if len(h_ids) != len(h_rows) or len(v_ids) != len(v_rows) or h_ids & v_ids:
        raise ValueError("H/V sequence IDs are duplicated or overlap")

    idx = index_map(sequence_index)
    old_rows = read_csv(old_fold1_index)
    old_ids = {r["seq_id"] for r in old_rows if r["split"] == "train"}
    if len(old_ids) != len([r for r in old_rows if r["split"] == "train"]):
        raise ValueError("duplicate sequence IDs in old fold-1 prediction index")
    old_manifest = json.loads(old_fold1_manifest.read_text(encoding="utf-8"))
    if old_manifest.get("state") != "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS":
        raise ValueError("old fold-1 forecast bank is not complete")
    if old_manifest.get("future_observed_agri_read") is not False or old_manifest.get("cpp_read") is not False:
        raise ValueError("old fold-1 forecast bank violates causal input boundary")

    summaries = {}
    represented_ids = {}
    key_sha = {}
    exclusion_obj = json.loads(exclusions_path.read_text(encoding="utf-8"))
    exclusions = {(r["station"].lower(), r["bjt_date"]): r["reason"]
                  for r in exclusion_obj["exclusions"]}
    if exclusion_obj.get("date_timezone") != "Asia/Shanghai":
        raise ValueError("station-day exclusions must use BJT/Asia-Shanghai dates")
    for split, wanted, label_split in (("H", h_ids, "train"), ("V", v_ids, "val")):
        seqsplit = "train" if split == "H" else "val"
        by_idx = idx[seqsplit]
        keys_path = labels / label_split / "keys.csv"
        key_rows = read_csv(keys_path)
        counts: Counter[tuple[str, str]] = Counter()
        keys_seen = set()
        scoreable_keys_seen = set()
        excluded_counts: Counter[str] = Counter()
        bad_seq_index = 0
        for r in key_rows:
            j = int(r["sequence_index_within_split"])
            sid = by_idx.get(j)
            if sid is None:
                bad_seq_index += 1
                continue
            if sid not in wanted:
                continue
            k = (r["station"].lower(), r["init_time_utc"], r["target_time_utc"], int(r["lead_minutes"]))
            if k in keys_seen:
                raise ValueError(f"duplicate label key in {split}: {k}")
            keys_seen.add(k)
            counts[(sid, r["station"].lower())] += 1
            target_bjt_date = (datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S")
                               + timedelta(hours=8)).date().isoformat()
            exclusion_key = (r["station"].lower(), target_bjt_date)
            if exclusion_key in exclusions:
                excluded_counts[f"{exclusion_key[0]}:{exclusion_key[1]}"] += 1
            else:
                scoreable_keys_seen.add(k)
        seqs_with_any = {sid for sid, _ in counts}
        seq_station_pairs = len(counts)
        full_pairs = sum(n == 16 for n in counts.values())
        summaries[split] = {
            "manifest_sequences": len(wanted),
            "sequences_with_any_labeled_GHI": len(seqs_with_any),
            "sequence_station_pairs_with_any_label": seq_station_pairs,
            "sequence_station_pairs_with_all_16_leads": full_pairs,
            "common_labeled_rows": len(keys_seen),
            "rows_by_station": {s: sum(1 for k in keys_seen if k[0] == s) for s in ("sili", "zhujia")},
            "quality_excluded_rows": dict(excluded_counts),
            "common_scoreable_rows_after_station_day_exclusions": len(scoreable_keys_seen),
            "scoreable_rows_by_station": {s: sum(1 for k in scoreable_keys_seen if k[0] == s)
                                          for s in ("sili", "zhujia")},
            "unmapped_label_key_indices": bad_seq_index,
            "label_keys_sha256": sha(keys_path),
        }
        represented_ids[split] = seqs_with_any
        key_sha[split] = sha(keys_path)

    if not represented_ids["H"].issubset(old_ids):
        raise ValueError("some labeled H sequences are absent from the old fold-1 bank")
    # The old bank may contain no-label rows, but they cannot enter paired GHI scoring.
    summaries["old_fold1_bank"] = {
        "sequences": len(old_ids),
        "H_sequences": len(old_ids & h_ids),
        "not_H_sequences": len(old_ids - h_ids),
        "labeled_H_sequences_missing_from_old_bank": len(represented_ids["H"] - old_ids),
        "H_manifest_sequences_without_any_GHI_label": len(h_ids - represented_ids["H"]),
        "V_manifest_sequences_with_any_GHI_label": len(represented_ids["V"]),
    }
    return {
        "state": "PASS_METADATA_ONLY_HV_GHI_COVERAGE_AUDIT",
        "sequence_index_sha256": sha(sequence_index),
        "h_sequences_sha256": sha(support / "h_sequences.csv"),
        "v_sequences_sha256": sha(support / "v_sequences.csv"),
        "old_fold1_index_sha256": sha(old_fold1_index),
        "old_fold1_manifest_sha256": sha(old_fold1_manifest),
        "station_day_exclusions_sha256": sha(exclusions_path),
        "splits": summaries,
        "payloads_opened": False,
        "test_or_buffer_paths_read": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    for n in ("support", "sequence-index", "labels", "old-fold1-index", "old-fold1-manifest", "exclusions", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.support, a.sequence_index, a.labels,
                   a.old_fold1_index, a.old_fold1_manifest, a.exclusions)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
