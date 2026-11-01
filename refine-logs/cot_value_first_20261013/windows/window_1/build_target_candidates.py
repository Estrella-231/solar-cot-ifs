"""Freeze metadata-only train/validation target candidates for execution window 1.

This does not open AGRI, GHI, CPP, or test payloads. It expands the already
audited 8-to-16 manifest's frame paths, deduplicates by valid time, and emits
one station/target candidate key. Workbook label completeness is checked later
on the data host and is deliberately not inferred here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLAN_DIR = ROOT / "refine-logs/cot_value_first_20261013"
MANIFEST = ROOT / "experiments/monthly_block_20261002/manifests_v1/hunan_8to16.csv"
DAY_ASSIGNMENT = ROOT / "experiments/monthly_block_20261002/manifests_v1/day_assignment.csv"
GRID_AUDIT = ROOT / "audits/station_grid_contract_20260908_v1.json"
OUT = PLAN_DIR / "windows/window_1"
FRAME_RE = re.compile(r"/(\d{12})\.npy$")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    day_split = {}
    with DAY_ASSIGNMENT.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            day_split[row["date_BJT"]] = row["split"]

    frames: dict[str, dict[str, str]] = defaultdict(dict)
    split_for_frame = {}
    n_sequences = defaultdict(int)
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            n_sequences[split] += 1
            paths = row["data_relpaths"].split("|")
            if len(paths) != 24:
                raise ValueError(f"expected 24 paths, got {len(paths)} in {row['seq_id']}")
            for rel in paths:
                match = FRAME_RE.search("/" + rel.replace("\\", "/"))
                if not match:
                    raise ValueError(f"unrecognized frame path: {rel}")
                stamp = datetime.strptime(match.group(1), "%Y%m%d%H%M")
                target_bjt = stamp + timedelta(hours=8)
                date_bjt = target_bjt.strftime("%Y-%m-%d")
                assigned = day_split.get(date_bjt)
                if assigned != split:
                    raise ValueError(f"split/date mismatch: {split} {date_bjt} -> {assigned}")
                prior = split_for_frame.setdefault(stamp.isoformat(), split)
                if prior != split:
                    raise ValueError(f"frame crosses splits: {stamp} {prior}/{split}")
                old = frames[split].setdefault(stamp.isoformat(), rel)
                if old != rel:
                    raise ValueError(f"same timestamp maps to multiple paths: {stamp}")

    expected = {"train": 51236, "val": 4497}
    for split, n in expected.items():
        if len(frames[split]) != n:
            raise ValueError(f"unique {split} frames {len(frames[split])} != frozen {n}")

    grid = json.loads(GRID_AUDIT.read_text(encoding="utf-8"))
    if grid["state"] != "PASS_STATION_GRID_CONTRACT_ONLY" or grid["test_used"]:
        raise ValueError("station grid audit is not the expected metadata-only receipt")
    stations = {x["station"]: x for x in grid["stations"]}
    station_meta = {
        "sili": {"name_zh": "四里", "lat": 25.9478611111, "lon": 112.347916667},
        "zhujia": {"name_zh": "竺家", "lat": 29.566666667, "lon": 112.45},
    }

    outpath = OUT / "target_candidates_train_val.csv"
    columns = ["split", "station", "target_time_utc", "target_time_bjt",
               "label_record_times_bjt", "agri_frame_relpath", "station_grid_rc",
               "station_patch_rc", "head_3x3_patch_slice_rc_exclusive",
               "true_hdf_rc", "ghi_label_state"]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for split in ("train", "val"):
            for stamp_text, rel in sorted(frames[split].items()):
                utc = datetime.fromisoformat(stamp_text)
                bjt = utc + timedelta(hours=8)
                records = "|".join((bjt-timedelta(minutes=m)).isoformat(sep=" ")
                                    for m in (10, 5, 0))
                for station in ("sili", "zhujia"):
                    s = stations[station]
                    meta = station_meta[station]
                    if [round(s["station_latlon"][0], 9), round(s["station_latlon"][1], 9)] != [
                        round(meta["lat"], 9), round(meta["lon"], 9)
                    ]:
                        raise ValueError(f"station metadata mismatch: {station}")
                    writer.writerow({
                        "split": split,
                        "station": station,
                        "target_time_utc": utc.isoformat(sep=" "),
                        "target_time_bjt": bjt.isoformat(sep=" "),
                        "label_record_times_bjt": records,
                        "agri_frame_relpath": rel,
                        "station_grid_rc": ",".join(map(str, s["nearest_grid_row_col"])),
                        "station_patch_rc": ",".join(map(str, s["patch_nearest_row_col"])),
                        "head_3x3_patch_slice_rc_exclusive": "7:10,7:10",
                        "true_hdf_rc": ",".join(map(str, s["true_HDF_row_col"])),
                        "ghi_label_state": "pending_source_host_read",
                    })

    contract = {
        "schema": "cot_value_first_window1_data_contract_v1",
        "state": "PARTIAL_METADATA_CANDIDATES_READY_NEEDS_GHI_AND_GEOMETRY_AUDIT",
        "created_at_bjt": "2026-10-13",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "split_manifest_sha256": sha(MANIFEST),
        "day_assignment_sha256": sha(DAY_ASSIGNMENT),
        "target_unit": "one unique observed AGRI nominal target timestamp per station; no overlapping init windows multiply samples",
        "target_time": "AGRI nominal UTC filename; BJT=UTC+08:00; contemporaneous target with init_time=null; no test payload or targets emitted",
        "stations": {
            name: {**station_meta[name],
                   "grid_rc": stations[name]["nearest_grid_row_col"],
                   "patch_rc": stations[name]["patch_nearest_row_col"],
                   "true_hdf_rc": stations[name]["true_HDF_row_col"],
                   "patch_bounds_rc_exclusive": stations[name]["bounds"]}
            for name in ("sili", "zhujia")
        },
        "model_geometry_inputs": {
            "planes": ["cosSOZ", "cosRAA"],
            "spatial_alignment": "per-pixel target-time geometry, cropped on the same 3x3 grid as AGRI/COT",
            "day_mask": "eligibility only; not an input plane; station target is eligible when solar zenith <90deg",
            "scaling": "retain bounded cosine values; do not fit per-window or per-image normalization",
            "status": "candidate aligned with window2 prototype; archived validation formula/crop witness passes, but current monthly-row geometry alignment remains pending",
        },
        "ghi_label": {
            "window": "(target_time-15min,target_time]",
            "source_records_bjt": ["target-10min", "target-5min", "target"],
            "observed_ghi": "arithmetic mean of the three interval-ending 5-minute measurements; require all three present, finite, nonnegative",
            "clear_sky": "Ineichen evaluated at the three 5-minute interval midpoints target-12.5/-7.5/-2.5min; use their mean",
            "kt": "mean(GHI)/mean(clear_sky_GHI), unclipped; GHI is label/evaluation only",
            "main_mask": "complete valid GHI and clear-sky interval labels AND deterministic target station day flag (SZA<90deg) AND interval mean clear-sky GHI>0",
            "sensitivity_mask": "same records, day flag, and interval mean clear-sky GHI>=100 W m-2",
        },
        "cot": {
            "source": "frozen R applied to the complete native AGRI input view at the same target time, then crop station-centered 3x3; no R input-view change",
            "transform": "log1p once; one normalization fit on train-only values; no per-image centering",
            "cpp_reference_mask": "finite COT in [0,100] AND CLP in {0,1,2} AND all 13 AGRI channels valid AND day_mask>0.5 AND cosSOZ>cos(80deg); zero COT retained if reference-valid",
            "cpp_reference_does_not_filter_GHI_cohort": True,
        },
        "candidate_counts_before_station_GHI_validity": {
            "train_unique_agri_times": len(frames["train"]),
            "val_unique_agri_times": len(frames["val"]),
            "train_station_target_keys": 2*len(frames["train"]),
            "val_station_target_keys": 2*len(frames["val"]),
            "sequences_by_split": dict(n_sequences),
        },
        "target_manifest": outpath.name,
        "target_manifest_sha256": sha(outpath),
        "station_grid_receipt_sha256": sha(GRID_AUDIT),
        "test_used": False,
        "limitations": [
            "Candidate metadata keys are not yet the final GHI cohort; original station workbook availability/validity and interval labels must be reread on the data host.",
            "The selected clear-sky midpoint values correspond to the centers of the canonical target-10/-5/0 five-minute measurement intervals; W0 static code audit confirms this time alignment. No endpoint-sampling comparison is required to freeze the contract.",
            "The old saved GHI cohort has not yet been keyed against this new candidate list; do not treat old-cohort labels as monthly-contract labels.",
            "Station coordinates are the authoritative project schema; the grid receipt does not independently survey station geolocation.",
        ],
    }
    cpath = OUT / "data_contract.json"
    cpath.write_text(json.dumps(contract, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    audit = {
        "state": contract["state"], "manifest_sha256": sha(MANIFEST),
        "day_assignment_sha256": sha(DAY_ASSIGNMENT),
        "unique_train_frames": len(frames["train"]), "unique_val_frames": len(frames["val"]),
        "target_candidate_rows": 2*(len(frames["train"])+len(frames["val"])),
        "split_leakage": False, "date_assignment_matches": True,
        "test_payload_opened": False, "target_manifest_sha256": sha(outpath),
        "data_contract_sha256": sha(cpath),
    }
    (OUT / "target_candidate_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
