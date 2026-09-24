"""Read-only Hunan geometry, station-row and history-COT contract audit.

This script does not load future AGRI or test images.  It checks every train/val
manifest and GHI row, then reads solar auxiliary maps for a small fixed witness.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


STEP = timedelta(minutes=15)
BJT_OFFSET = timedelta(hours=8)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc_path(path):
    return datetime.strptime(Path(path).stem, "%Y%m%d%H%M")


def audit(project, data_root, source_root):
    config = json.loads((project / "configs/s_frozen_hunan_seed42.json").read_text())
    manifest = Path(config["manifest"])
    norm = Path(config["normalization"])
    for path, expected in ((manifest, config["manifest_sha256"]),
                           (norm, config["normalization_sha256"]),
                           (data_root / "grid_static.npz", config["grid_sha256"])):
        if sha256(path) != expected:
            raise RuntimeError(f"asset hash mismatch: {path}")
    if list(config["lead_minutes"]) != list(range(15, 241, 15)):
        raise RuntimeError("canonical lead list mismatch")
    if list(config["center_pixel"]) != [8, 8]:
        raise RuntimeError("station center mismatch")
    for station, box in config["station_patches"].items():
        r0, r1, c0, c1 = box
        if (r1 - r0, c1 - c0) != (16, 16) or not (0 <= r0 < r1 <= 256 and 0 <= c0 < c1 <= 256):
            raise RuntimeError(f"invalid station patch: {station}")

    rows_by_split = {"train": [], "val": []}
    seq_index = {"train": {}, "val": {}}
    with manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            split = "val" if row["split"] == "validation" else row["split"]
            if split not in rows_by_split:
                continue  # test is intentionally unopened
            paths = row["data_relpaths"].split("|")
            times = [parse_utc_path(p) for p in paths]
            if len(times) != 24 or any(b - a != STEP for a, b in zip(times, times[1:])):
                raise RuntimeError(f"noncanonical 24-frame sequence {row['seq_id']}")
            if times[0] + BJT_OFFSET != datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S"):
                raise RuntimeError(f"BJT/UTC mismatch {row['seq_id']}")
            if row["seq_id"] in seq_index[split]:
                raise RuntimeError(f"duplicate sequence {row['seq_id']}")
            seq_index[split][row["seq_id"]] = len(rows_by_split[split])
            rows_by_split[split].append((row, times))

    head_pack = project / "data/head_pack_trainval_20260908_v1"
    counts = {"train": 0, "val": 0}
    with (head_pack / "rows.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            split = "val" if row["split"] == "validation" else row["split"]
            if split not in counts:
                raise RuntimeError("test row entered train/val head pack")
            index = int(row["cache_index"])
            seq_id = row["simvp_seq_id"]
            if seq_index[split].get(seq_id) != index:
                raise RuntimeError(f"head sequence index mismatch: {split} {seq_id}")
            _, times = rows_by_split[split][index]
            lead_index = int(row["lead_index"])
            lead = int(row["lead_minutes"])
            if lead_index not in range(16) or lead != (lead_index + 1) * 15:
                raise RuntimeError("lead index mismatch")
            init = datetime.fromisoformat(row["init_time_bjt"]).replace(tzinfo=None)
            target = datetime.fromisoformat(row["target_time_bjt"]).replace(tzinfo=None)
            if init != times[7] + BJT_OFFSET or target != times[8 + lead_index] + BJT_OFFSET:
                raise RuntimeError("init/target mismatch")
            label_times = [datetime.fromisoformat(x).replace(tzinfo=None)
                           for x in row["label_record_times_bjt"].split("|")]
            if label_times != [target - timedelta(minutes=10), target - timedelta(minutes=5), target]:
                raise RuntimeError("15-minute GHI interval mismatch")
            station = row["station"].lower()
            if station not in config["station_patches"] or int(row["station_index"]) != list(config["station_patches"]).index(station):
                raise RuntimeError("station mapping mismatch")
            counts[split] += 1

    import sys
    sys.path.insert(0, str(source_root))
    from hunan_data import resolve_aux
    witness = []
    for split in ("train", "val"):
        split_rows = rows_by_split[split]
        # Fixed by index or target clock time, never by a model result.
        witness_indices = {0, len(split_rows) // 2, len(split_rows) - 1}
        for clock_hour in (6, 12, 18):
            match = next((i for i, (_, frame_times) in enumerate(split_rows)
                          if (frame_times[8] + BJT_OFFSET).hour == clock_hour and
                          (frame_times[8] + BJT_OFFSET).minute == 0), None)
            if match is None:
                raise RuntimeError(f"missing {clock_hour}:00 BJT geometry witness in {split}")
            witness_indices.add(match)
        for index in sorted(witness_indices):
            row, times = split_rows[index]
            for frame_index in (0, 7, 8, 15, 23):
                target_bjt = times[frame_index] + BJT_OFFSET
                aux_path = resolve_aux(data_root, target_bjt)
                with np.load(aux_path, allow_pickle=False) as aux:
                    maps = [np.asarray(aux[k]) for k in ("cosSOZ", "cosRAA", "day_mask")]
                if any(m.shape != (256, 256) or not np.isfinite(m).all() for m in maps):
                    raise RuntimeError(f"invalid geometry {aux_path}")
                r0, _, c0, _ = config["station_patches"]["sili"]
                witness.append({"split": split, "sequence_index": index, "seq_id": row["seq_id"],
                                "frame_index": frame_index, "role": "history" if frame_index < 8 else "target",
                                "lead_minutes": None if frame_index < 8 else 15 * (frame_index - 7),
                                "utc": times[frame_index].isoformat(), "bjt": target_bjt.isoformat(),
                                "aux_file": str(aux_path),
                                "sili_center_geometry": [float(m[r0 + 8, c0 + 8]) for m in maps]})
    history = project / "data/history_cot_trajectory_real_agri_rpair_20260923/original_R"
    audit_path = history / "audit.json"
    if not audit_path.exists():
        raise RuntimeError("missing original-R historical COT audit")
    history_audit = json.loads(audit_path.read_text())
    if history_audit.get("state") != "COMPLETE_HISTORY_COT_TRAJECTORY_SIDECAR":
        raise RuntimeError("historical COT bank not complete")
    for split in ("train", "val"):
        array = np.load(history / f"{split}_history_cot_log1p.npy", mmap_mode="r")
        if array.shape != (len(rows_by_split[split]), 2, 8, 1, 16, 16):
            raise RuntimeError("historical COT shape mismatch")
        for index in (0, len(array) // 2, len(array) - 1):
            if not np.isfinite(array[index]).all() or (array[index] < 0).any():
                raise RuntimeError("historical COT witness not finite/nonnegative")
    return {"state": "PASS_B0_CONTRACT_AUDIT", "test_used": False,
            "manifest_sha256": config["manifest_sha256"], "normalization_sha256": config["normalization_sha256"],
            "grid_sha256": config["grid_sha256"], "head_rows": counts,
            "sequences": {k: len(v) for k, v in rows_by_split.items()},
            "historical_cot_audit_sha256": sha256(audit_path),
            "geometry_contract": "24 chronological maps: history indices 0..7 and target indices 8..23",
            "label_contract": "target interval ends at target; raw 5-minute records target-10,-5,0",
            "station_patches": config["station_patches"], "station_center": config["center_pixel"],
            "witness": witness, "script_sha256": sha256(__file__)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite B0 audit")
    report = audit(args.project, args.data_root, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("state", "head_rows", "sequences", "test_used")}), flush=True)


if __name__ == "__main__":
    main()
