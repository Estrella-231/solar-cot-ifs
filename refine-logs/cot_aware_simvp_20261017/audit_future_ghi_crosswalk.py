"""Metadata-only mapping of forecast init/lead rows to monthly-split GHI labels.

Reads only train/validation sequence metadata and the already approved
train/validation 15-minute GHI row pack. It never opens satellite/CPP arrays
or test/buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np


SPLIT = "hunan_monthly_block_2d_buffer1d_seed42_v1"
STATIONS = ("sili", "zhujia")
LEADS = tuple(range(15, 241, 15))
UTC_OFFSET = timedelta(hours=8)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--sequence-audit", type=Path, required=True)
    p.add_argument("--ghi-rows", type=Path, required=True)
    p.add_argument("--ghi-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seqaudit = json.loads(a.sequence_audit.read_text(encoding="utf-8"))
    ghiman = json.loads((a.ghi_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if seqaudit.get("split_id") != SPLIT or ghiman.get("split_id") != SPLIT:
        raise ValueError("monthly split id mismatch")
    if ghiman.get("test_payload_opened") is not False:
        raise ValueError("GHI pack does not certify test payload exclusion")
    if ghiman.get("sample_key") != "(station,target_time_utc)":
        raise ValueError("unexpected GHI row key")

    # Parse only train/validation sequence keys. Test and buffer paths are not
    # used for target generation or opened as payloads.
    seqs = []
    split_index = Counter()
    for r in rows(a.sequences):
        if r["split"] in ("train", "val"):
            r["split_index"] = split_index[r["split"]]
            split_index[r["split"]] += 1
            seqs.append(r)
    ghi_rows = [(i, r) for i, r in enumerate(rows(a.ghi_rows)) if r["split"] in ("train", "val")]
    labels = {}
    for row_index, r in ghi_rows:
        key = (r["station"].lower(), r["target_time_utc"])
        if key in labels:
            raise ValueError(f"duplicate GHI key: {key}")
        labels[key] = (r["split"], row_index)

    totals = Counter()
    per_lead = {str(k): Counter() for k in LEADS}
    target_multiplicity = Counter()
    key_split, key_seq, key_station, key_lead, key_ghi = [], [], [], [], []
    for seq in seqs:
        split = seq["split"]
        if split not in ("train", "val"):
            raise ValueError("non-train/val sequence entered audit")
        start_bjt = datetime.strptime(seq["BJT_start"], "%Y-%m-%d %H:%M:%S")
        init_bjt = start_bjt + timedelta(minutes=105)
        for lead in LEADS:
            target_utc = (init_bjt + timedelta(minutes=lead) - UTC_OFFSET).strftime("%Y-%m-%d %H:%M:%S")
            for station in STATIONS:
                totals["candidate"] += 1
                per_lead[str(lead)]["candidate"] += 1
                label = labels.get((station, target_utc))
                if label is None:
                    totals["missing_label"] += 1
                    per_lead[str(lead)]["missing_label"] += 1
                    continue
                label_split, ghi_index = label
                if label_split != split:
                    totals["split_mismatch"] += 1
                    per_lead[str(lead)]["split_mismatch"] += 1
                    continue
                totals["matched"] += 1
                totals[f"{split}_matched"] += 1
                totals[f"{station}_{split}_matched"] += 1
                per_lead[str(lead)]["matched"] += 1
                per_lead[str(lead)][f"{station}_matched"] += 1
                target_multiplicity[(station, target_utc)] += 1
                key_split.append(0 if split == "train" else 1)
                key_seq.append(seq["split_index"])
                key_station.append(0 if station == "sili" else 1)
                key_lead.append(LEADS.index(lead))
                key_ghi.append(ghi_index)

    key_path = a.output.with_name(a.output.stem + "_keys.npz")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_arrays = {
        "split_code": np.asarray(key_split, dtype=np.uint8),
        "sequence_index_within_split": np.asarray(key_seq, dtype=np.int32),
        "station_index": np.asarray(key_station, dtype=np.uint8),
        "lead_index": np.asarray(key_lead, dtype=np.uint8),
        "ghi_row_index": np.asarray(key_ghi, dtype=np.int32),
    }
    if key_arrays["sequence_index_within_split"].size != totals["matched"]:
        raise AssertionError("key array length mismatch")
    keytmp = key_path.with_suffix(key_path.suffix + ".tmp")
    with keytmp.open("wb") as f:
        np.savez_compressed(f, **key_arrays)
    keytmp.replace(key_path)

    out = {
        "status": "PASS_METADATA_ONLY_FUTURE_GHI_CROSSWALK",
        "split_id": SPLIT,
        "time_contract": "init=BJT_start+105min; target=init+lead; UTC target=BJT target-8h",
        "label_contract": "15-minute interval-ending target; leads +15..+240 min",
        "stations": list(STATIONS),
        "leads_minutes": list(LEADS),
        "sequence_counts_train_val": {s: sum(r["split"] == s for r in seqs) for s in ("train", "val")},
        "ghi_label_counts_train_val": dict(Counter(r["split"] for _, r in ghi_rows)),
        "candidate_rows_station_init_lead": totals["candidate"],
        "matched_rows_station_init_lead": totals["matched"],
        "matched_by_split": {s: totals[f"{s}_matched"] for s in ("train", "val")},
        "matched_by_station_split": {f"{st}_{sp}": totals[f"{st}_{sp}_matched"] for st in STATIONS for sp in ("train", "val")},
        "missing_label_rows": totals["missing_label"],
        "split_mismatch_rows": totals["split_mismatch"],
        "per_lead": {k: dict(v) for k, v in per_lead.items()},
        "unique_station_target_labels_used": len(target_multiplicity),
        "target_label_multiplicity_min_max": ([min(target_multiplicity.values()), max(target_multiplicity.values())] if target_multiplicity else None),
        "source_sha256": {"sequences": sha(a.sequences), "sequence_audit": sha(a.sequence_audit), "ghi_rows": sha(a.ghi_rows), "ghi_pack_manifest": sha(a.ghi_pack / "pack_manifest.json")},
        "paired_key_array": {"path": key_path.name, "rows": totals["matched"], "sha256": sha(key_path), "fields": list(key_arrays)},
        "payload_reads": {"agri": 0, "cpp": 0, "ghi_arrays": 0, "test": 0, "buffer": 0},
    }
    if totals["split_mismatch"]:
        out["warning"] = "Do not train/score mismatched keys; investigate split construction before building forecast inputs."
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    tmp.replace(a.output)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
