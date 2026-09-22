"""Pack four fixed forecast leads paired with train/validation CPP COT labels.

Future observed AGRI never enters the forecast input. The only observed-future
fields read from the R pack are supervised CPP target/mask and deterministic
geometry for an exact alignment check.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


LEADS = (15, 60, 120, 240)
STATIONS = ("sili", "zhujia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.utcoffset() != timedelta(0):
        raise RuntimeError("non-UTC pair key")
    return dt


def pack_split(root, output, split, references, raw_geometry):
    cache = root / "data/frozen_forecast_trainval_20260907_v1" / split
    with (cache / "index.csv").open(newline="") as f:
        sequences = list(csv.DictReader(f))
    if any(int(r["index"]) != i or r["split"] != split for i, r in enumerate(sequences)):
        raise RuntimeError("forecast cache index ordering/split mismatch")
    matched = []
    sequence_matches = defaultdict(list)
    for row in sequences:
        sequence = int(row["index"])
        init = stamp(row["init_time_utc"])
        for lead in LEADS:
            target = init + timedelta(minutes=lead)
            for station, station_index in zip(STATIONS, (0, 1)):
                ref = references.get((station, target))
                if ref is None:
                    continue
                record = (sequence, station_index, lead // 15 - 1, ref)
                sequence_matches[sequence].append((len(matched), record))
                matched.append(record)
    if not matched:
        raise RuntimeError("zero matched forecast-domain COT pairs")
    target_dir = output / split
    target_dir.mkdir()
    array = np.lib.format.open_memmap(target_dir / "forecast_x.npy", mode="w+",
                                      dtype=np.float32, shape=(len(matched), 16, 16, 16))
    written = np.zeros(len(matched), dtype=bool)
    complete = json.loads((cache / "complete.json").read_text())
    if complete["state"] != "COMPLETE" or complete["test_used"] is not False:
        raise RuntimeError("forecast cache incomplete or test-tainted")
    for shard_number, entry in enumerate(complete["shards"], 1):
        shard_path = cache / entry["name"]
        if sha(shard_path) != entry["sha256"]:
            raise RuntimeError("forecast shard hash mismatch: " + entry["name"])
        with np.load(shard_path, allow_pickle=False) as shard:
            indices = shard["indices"]
            agri = shard["predicted_agri_physical"]
            geometry = shard["geometry"]
        if agri.shape != (len(indices), 2, 16, 13, 16, 16) or geometry.shape != (len(indices), 2, 16, 3, 16, 16):
            raise RuntimeError("forecast shard shape mismatch")
        for local, sequence in enumerate(indices):
            for position, (sequence_key, station, lead_index, ref) in sequence_matches.get(int(sequence), ()):
                if int(sequence) != sequence_key or written[position]:
                    raise RuntimeError("forecast pair duplicate/mismatch")
                geom = geometry[local, station, lead_index]
                if not np.array_equal(geom, raw_geometry[ref, 13:]):
                    raise RuntimeError("forecast and CPP target geometry differ")
                joined = np.concatenate((agri[local, station, lead_index], geom), axis=0)
                if joined.shape != (16, 16, 16) or not np.isfinite(joined).all():
                    raise RuntimeError("invalid forecast R input")
                array[position] = joined
                written[position] = True
        if shard_number % 100 == 0:
            array.flush()
            print("PACK", split, shard_number, len(complete["shards"]), int(written.sum()), flush=True)
    if not written.all():
        raise RuntimeError("missing matched forecast R inputs")
    array.flush()
    records = np.asarray(matched, dtype=np.int64)
    np.save(target_dir / "keys.npy", records)
    return {"pairs": len(matched), "unique_cpp_targets": len(set(records[:, 3])),
            "forecast_sequences": len(sequences), "shards_verified": len(complete["shards"]),
            "lead_counts": {str(lead): int((records[:, 2] == lead // 15 - 1).sum()) for lead in LEADS},
            "forecast_x_sha256": sha(target_dir / "forecast_x.npy"),
            "keys_sha256": sha(target_dir / "keys.npy"),
            "source_index_sha256": sha(cache / "index.csv"),
            "source_complete_sha256": sha(cache / "complete.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite forecast-domain COT pilot pack")
    source = root / "data/cot_repaired_pack_20260907_v1"
    status = json.loads((source / "pack_status.json").read_text())
    if status["state"] != "COMPLETE" or status["test_payloads_read"] != 0:
        raise RuntimeError("CPP reference pack incomplete")
    with (source / "rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18514:
        raise RuntimeError("CPP reference row count drift")
    references = {split: {} for split in ("train", "val")}
    for index, row in enumerate(rows):
        split = "val" if row["split"] == "validation" else row["split"]
        if split not in references:
            raise RuntimeError("test CPP reference unexpectedly in train/val pack")
        key = (row["station_id"].lower(), stamp(row["timestamp_utc"]))
        if key in references[split]:
            raise RuntimeError("duplicate CPP target key")
        references[split][key] = index
    raw_geometry = np.load(source / "x_raw.npy", mmap_mode="r")
    output.mkdir(parents=True)
    result = {"state": "PACKING", "test_used": False,
              "forecast_leads_minutes": LEADS,
              "input": "13 SimVP forecast physical AGRI + 3 deterministic geometry",
              "label": "matched train/validation CPP reference COT from original R pack",
              "R_pack_rows_sha256": sha(source / "rows.csv"),
              "source_code_sha256": sha(__file__), "splits": {}}
    for split in ("train", "val"):
        result["splits"][split] = pack_split(root, output, split, references[split], raw_geometry)
        (output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
    result["state"] = "COMPLETE_FORECAST_DOMAIN_COT_PILOT_PACK"
    (output / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
