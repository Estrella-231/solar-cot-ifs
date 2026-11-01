"""Independent semantic check for the metadata-only future-GHI crosswalk."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

LEADS = tuple(range(15, 241, 15))
STATIONS = ("sili", "zhujia")
UTC_OFFSET = timedelta(hours=8)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--ghi-rows", type=Path, required=True)
    p.add_argument("--crosswalk", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    seqs = {s: [] for s in ("train", "val")}
    for row in csv_rows(a.sequences):
        if row["split"] in seqs:
            seqs[row["split"]].append(row)
    labels = {}
    with a.ghi_rows.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row["split"] not in ("train", "val"):
                continue
            key = (row["station"].lower(), row["target_time_utc"])
            if key in labels:
                raise AssertionError(f"duplicate source label key {key}")
            labels[key] = (row["split"], i)

    report = json.loads(a.crosswalk.read_text(encoding="utf-8"))
    with np.load(a.keys, allow_pickle=False) as z:
        split_code = z["split_code"]
        seq_index = z["sequence_index_within_split"]
        station_index = z["station_index"]
        lead_index = z["lead_index"]
        ghi_index = z["ghi_row_index"]
    arrays = (split_code, seq_index, station_index, lead_index, ghi_index)
    if len({len(x) for x in arrays}) != 1:
        raise AssertionError("crosswalk key arrays have different lengths")
    if len(split_code) != report["matched_rows_station_init_lead"]:
        raise AssertionError("key count differs from report")

    counts = Counter()
    seen_forecast_keys = set()
    for i in range(len(split_code)):
        split = "train" if int(split_code[i]) == 0 else "val" if int(split_code[i]) == 1 else None
        if split is None:
            raise AssertionError("invalid split code")
        si, li, gi = int(seq_index[i]), int(lead_index[i]), int(ghi_index[i])
        st = int(station_index[i])
        if st not in (0, 1) or li not in range(16) or not (0 <= si < len(seqs[split])):
            raise AssertionError(f"out-of-range key at row {i}")
        station = STATIONS[st]
        seq = seqs[split][si]
        start = datetime.strptime(seq["BJT_start"], "%Y-%m-%d %H:%M:%S")
        init = start + timedelta(minutes=105)
        target = (init + timedelta(minutes=LEADS[li]) - UTC_OFFSET).strftime("%Y-%m-%d %H:%M:%S")
        source = labels.get((station, target))
        if source != (split, gi):
            raise AssertionError(f"key mismatch at row {i}: {(split, station, target, gi)} != {source}")
        key = (split, si, st, li)
        if key in seen_forecast_keys:
            raise AssertionError(f"duplicate forecast key {key}")
        seen_forecast_keys.add(key)
        counts[(split, station)] += 1

    expected = report["matched_by_station_split"]
    for station in STATIONS:
        for split in ("train", "val"):
            if counts[(split, station)] != expected[f"{station}_{split}"]:
                raise AssertionError(f"count mismatch {station}/{split}")
    result = {
        "state": "PASS_INDEPENDENT_FUTURE_GHI_CROSSWALK_KEY_AUDIT",
        "rows": len(split_code),
        "matched_by_station_split": {f"{station}_{split}": counts[(split, station)]
                                     for station in STATIONS for split in ("train", "val")},
        "unique_forecast_keys": len(seen_forecast_keys),
        "source_sha256": {"sequences": sha(a.sequences), "ghi_rows": sha(a.ghi_rows),
                          "crosswalk": sha(a.crosswalk), "keys": sha(a.keys)},
        "payload_reads": {"agri": 0, "cpp": 0, "ghi_arrays": 0, "test": 0, "buffer": 0},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
