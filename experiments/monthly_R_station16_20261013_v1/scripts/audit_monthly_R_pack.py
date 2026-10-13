#!/usr/bin/env python3
"""Audit the completed train/validation-only monthly R data package."""
import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np

EXPECTED_CANDIDATE_SHA = "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807"
EXPECTED_CPP_SHA = "0a303ec6965be360689f3891e96b5e4085ad3e6550b2db6504a8b2ac35565dd2"
EXPECTED_SPLIT_COUNTS = {"train": 51236, "val": 4497}
STATIONS = ("sili", "zhujia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    status = json.loads((a.pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or status.get("test_payloads_read") != 0:
        raise RuntimeError("pack completion/test gate failed")
    if status.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA or status.get("cpp_inventory_sha256") != EXPECTED_CPP_SHA:
        raise RuntimeError("frozen train/val manifest binding failed")
    if status.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise RuntimeError("unexpected date split")
    if status.get("cpp_grid_mapping_max_error_deg", float("inf")) > 1e-4:
        raise RuntimeError("CPP-to-AGRI coordinate mapping exceeds tolerance")
    with (a.pack / "source_coverage.csv").open(newline="", encoding="utf-8") as f:
        coverage = list(csv.DictReader(f))
    if len(coverage) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise RuntimeError(f"candidate frame coverage mismatch: {len(coverage)}")
    if any(r["split"] not in EXPECTED_SPLIT_COUNTS for r in coverage):
        raise RuntimeError("test or unknown split in coverage")
    candidate_counts = {s: sum(r["split"] == s for r in coverage) for s in EXPECTED_SPLIT_COUNTS}
    if candidate_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"candidate split counts differ: {candidate_counts}")
    with (a.pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected_rows = 2 * len(coverage)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected two station rows per candidate frame, got {len(rows)}/{expected_rows}")
    keyed = {(r["target_time_utc"], r["station"]): r for r in rows}
    if len(keyed) != len(rows) or any(st not in STATIONS for _, st in keyed):
        raise RuntimeError("duplicate station/time row or unknown station")
    for c in coverage:
        for st in STATIONS:
            r = keyed.get((c["target_time_utc"], st))
            if r is None or r["split"] != c["split"]:
                raise RuntimeError("coverage-to-station row join failed")
    supervised = [r for r in rows if r["state"] == "SUPERVISED"]
    idx = np.asarray([int(r["array_index"]) for r in supervised], np.int64)
    if len(np.unique(idx)) != len(idx) or not np.array_equal(np.sort(idx), np.arange(len(idx))):
        raise RuntimeError("supervised array_index values are not a unique contiguous range")
    if len(supervised) != int(status["packed_examples"]):
        raise RuntimeError("row/array supervised count mismatch")
    arrays = {n: np.load(a.pack / f"{n}.npy", mmap_mode="r", allow_pickle=False) for n in ("x_raw", "cot", "mask")}
    if arrays["x_raw"].shape != (len(supervised), 16, 16, 16) or arrays["cot"].shape != (len(supervised), 1, 16, 16) or arrays["mask"].shape != arrays["cot"].shape:
        raise RuntimeError("packed tensor shape mismatch")
    by_split = {}
    for split in ("train", "val"):
        recs = [r for r in supervised if r["split"] == split]
        ids = np.asarray([int(r["array_index"]) for r in recs], np.int64)
        ms = np.asarray(arrays["mask"][ids], bool)
        ys = np.asarray(arrays["cot"][ids], np.float32)
        xs = np.asarray(arrays["x_raw"][ids], np.float32)
        if not ms.any() or not np.isfinite(ys[ms]).all() or np.any((ys[ms] < 0) | (ys[ms] > 100)):
            raise RuntimeError(f"invalid supervised labels for {split}")
        if not np.isfinite(xs).all(axis=1)[ms[:, 0]].all():
            raise RuntimeError(f"nonfinite input under mask for {split}")
        by_split[split] = dict(supervised_examples=len(recs), valid_pixels=int(ms.sum()),
                               exact_zero_pixels=int(((ys == 0) & ms).sum()),
                               thick_ge30_pixels=int(((ys >= 30) & ms).sum()))
    state_counts = {}
    for split in ("train", "val"):
        state_counts[split] = {}
        for r in rows:
            if r["split"] == split:
                state_counts[split][r["state"]] = state_counts[split].get(r["state"], 0) + 1
        if sum(state_counts[split].values()) != 2 * EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError("per-split state totals do not match station-frame candidates")
    error_path = a.pack / "source_errors.csv"
    with error_path.open(newline="", encoding="utf-8-sig") as f:
        unresolved_errors = list(csv.DictReader(f))
    unresolved_frames = {r["target_time_utc"] for r in unresolved_errors}
    source_error_rows = [r for r in rows if r["state"] == "SOURCE_READ_ERROR"]
    if len(unresolved_frames) != int(status.get("source_read_error_frames", -1)):
        raise RuntimeError("unresolved source-error frame count differs from pack status")
    if len(source_error_rows) != 2 * len(unresolved_frames):
        raise RuntimeError("source-error rows do not account for both stations per frame")
    if any(r["target_time_utc"] not in unresolved_frames for r in source_error_rows):
        raise RuntimeError("unresolved error rows and source_errors.csv keys differ")
    if status.get("source_retry_recovered_frames", 0):
        recovery_path = a.pack / "source_error_recheck.json"
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        if sha(recovery_path) != status.get("source_error_recheck_sha256"):
            raise RuntimeError("source-error recovery report hash mismatch")
        if recovery.get("test_payloads_read") != 0:
            raise RuntimeError("recovery report violates the test-closed contract")
        if int(status["source_retry_recovered_frames"]) != int(status.get("first_pass_source_read_error_frames", -1)):
            raise RuntimeError("recovered frame count does not match first-pass errors")
    report = dict(state="PASS_MONTHLY_R_STATION16_PACK_AUDIT", split_id=status["split_id"],
                  test_payloads_read=0, candidate_sha256=status["candidate_sha256"],
                  cpp_inventory_sha256=status["cpp_inventory_sha256"], pack_status_sha256=sha(a.pack / "pack_status.json"),
                  arrays={n: dict(shape=list(arrays[n].shape), sha256=sha(a.pack / f"{n}.npy")) for n in arrays},
                  candidate_frames_by_split=candidate_counts, station_row_count=len(rows),
                  source_read_error_frames=status.get("source_read_error_frames"),
                  sample_states_by_split=state_counts, supervised_by_split=by_split,
                  unresolved_source_error_frames=len(unresolved_frames),
                  first_pass_source_error_frames=status.get("first_pass_source_read_error_frames", len(unresolved_frames)),
                  retry_recovered_source_error_frames=status.get("source_retry_recovered_frames", 0),
                  cpp_grid_mapping_max_error_deg=status["cpp_grid_mapping_max_error_deg"],
                  valid_zero_cot_retained=status.get("valid_zero_cot_retained"),
                  missing_cpp_is_not_clear_sky=status.get("missing_cpp_is_not_clear_sky"),
                  claim_scope="CPP-supported daytime retrieval-reference pixels; not independent truth")
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
