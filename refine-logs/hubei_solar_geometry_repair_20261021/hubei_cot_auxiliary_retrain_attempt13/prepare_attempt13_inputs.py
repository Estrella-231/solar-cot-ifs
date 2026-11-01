#!/usr/bin/env python3
"""Validate/recover node-local corrected data and build a physical-COT sidecar."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np


BASE_DEFAULT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021")
RPACK_DEFAULT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2")
CPP_SIDECAR_DEFAULT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/cpp_attempt2/cpp_supervision_trainval.npz")
OLD_STAGE_DEFAULT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017")
CANONICAL_DEFAULT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017/data")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def validate_data(prepared: Path, rpack: Path, *, check_payload_files: bool) -> dict[str, object]:
    ready_path = prepared / "DATA_READY.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("state") != "CORRECTED_SOLAR_TRAIN_VAL_DATA_READY":
        raise ValueError(f"Unexpected DATA_READY state: {ready.get('state')}")
    if ready.get("S_sequences") != {"train": 4086, "val": 407}:
        raise ValueError(f"Wrong corrected S train/val counts: {ready.get('S_sequences')}")
    if ready.get("S_frames") != 10473 or ready.get("test_payloads_read") != 0 or ready.get("buffer_payloads_read") != 0:
        raise ValueError("Corrected S data receipt has wrong frame/test/buffer counts")
    audit_path = prepared / ("DATA_AUDIT_PASS_attempt13.json" if prepared.name == "prepared_attempt13" else "DATA_AUDIT_PASS_attempt12.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not str(audit.get("state", "")).startswith("PASS_PREPARED_ATTEMPT12_TRAIN_VAL_KEYS_LABELS_AGRI_GEOMETRY"):
        raise ValueError(f"Prepared-data audit did not pass: {audit.get('state')}")
    if audit.get("test_payloads_read") != 0 or audit.get("buffer_payloads_read") != 0:
        raise ValueError("Prepared-data audit reports test/buffer reads")

    index = prepared / "indexes" / "full.csv"
    rows = read_rows(index)
    counts = {k: sum(r["split"] == k for r in rows) for k in ("train", "val")}
    if counts != {"train": 4086, "val": 407} or len(rows) != 4493:
        raise ValueError(f"Wrong full SimVP index split counts: {counts}")
    if any(r["split"] not in {"train", "val"} for r in rows):
        raise ValueError("Non-train/validation sequence in SimVP index")
    used: set[str] = set()
    for row in rows:
        data_paths = row["data_paths"].split("|")
        aux_paths = row["aux_paths"].split("|")
        if len(data_paths) != 24 or len(aux_paths) != 24:
            raise ValueError("A SimVP sequence does not contain exactly 24 aligned frames")
        used.update(data_paths)
        used.update(aux_paths)
    if check_payload_files:
        missing = [p for p in used if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"Node-local corrected cache misses {len(missing)} train/val paths; first={missing[0]}")
    norm = prepared / "norms" / "full.npz"
    if sha(norm) != ready["S_norms"]["full"]:
        raise ValueError("Frozen train-only SimVP normalization does not match DATA_READY")

    pack_status = json.loads((rpack / "pack_status.json").read_text(encoding="utf-8"))
    if pack_status.get("state") != "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED":
        raise ValueError("R CPP pack is not the corrected train/val-only pack")
    if pack_status.get("test_payloads_read") != 0 or pack_status.get("buffer_payloads_read") != 0:
        raise ValueError("R CPP pack reports test/buffer payload reads")
    hashes = json.loads((rpack / "SHA256.json").read_text(encoding="utf-8"))
    if {n: sha(rpack / n) for n in hashes} != hashes:
        raise ValueError("Corrected R pack SHA mismatch")
    rrows = read_rows(rpack / "rows.csv")
    if len(rrows) != 3479 or sum(r["split"] == "train" for r in rrows) != 3178 or sum(r["split"] == "val" for r in rrows) != 301:
        raise ValueError("Corrected R pack has wrong train/val row counts")
    if any(r["split"] not in {"train", "val"} for r in rrows):
        raise ValueError("R pack contains a non-train/validation row")
    return {"prepared_dir": str(prepared), "S_cache": ready["S_cache"],
            "index": str(index), "index_sha256": sha(index), "S_norm": str(norm),
            "S_norm_sha256": sha(norm), "sequence_counts": counts,
            "unique_train_val_frame_paths": len(used), "node_cache_paths_checked": bool(check_payload_files),
            "R_pack": str(rpack), "R_rows": len(rrows), "R_pack_sha256": hashes,
            "DATA_READY_sha256": sha(ready_path), "DATA_AUDIT_sha256": sha(audit_path),
            "test_payloads_read": 0, "buffer_payloads_read": 0}


def build_if_cache_missing(base: Path, attempt: Path, py: str, rpack: Path, cpp_sidecar: Path) -> Path:
    p12 = base / "prepared_attempt12"
    ready = json.loads((p12 / "DATA_READY.json").read_text(encoding="utf-8"))
    cache = Path(ready["S_cache"])
    try:
        validate_data(p12, rpack, check_payload_files=True)
        print(f"REUSING_NODE22_VERIFIED_ATTEMPT12_CACHE {cache}", flush=True)
        return p12
    except (FileNotFoundError, OSError) as exc:
        print(f"ATTEMPT12_NODE_CACHE_UNAVAILABLE; rebuilding isolated attempt13 data: {exc}", flush=True)

    p13 = base / "prepared_attempt13"
    if p13.exists():
        raise FileExistsError(f"Refusing to overwrite {p13}; use a fresh attempt directory")
    prep = Path("/dev/shm/hubei_cot_aux_preparation_attempt13")
    cache13 = Path("/dev/shm/hubei_solar_corrected_20261021_attempt13")
    if prep.exists() or cache13.exists():
        raise FileExistsError("Attempt13 recovery cache already exists; preserve it and inspect before retry")
    archive = base / "preparation_attempt6.tar"
    restore_archive = base / "restoration_requests_attempt2.tar.gz"
    if sha(archive) != "7169be51bff5a20d2cfa53543528cc2914ffa6e2d931833c6b64b5c8742d38e4":
        raise ValueError("Preparation source archive SHA mismatch")
    if sha(restore_archive) != "59fd880be1b5f1e56a2964f0733ad068115ebda19977ef1686e82f60a6043f8f":
        raise ValueError("Radiance restoration request archive SHA mismatch")
    prep.mkdir(parents=True)
    with tarfile.open(archive, "r:") as tf:
        tf.extractall(prep)
    with tarfile.open(restore_archive, "r:gz") as tf:
        tf.extractall(prep)
    code = base / "code_pipeline_attempt13"
    builder = code / "build_corrected_data_attempt13.py"
    command = [py, str(builder), "--base", str(base), "--preparation", str(prep),
               "--cache", str(cache13), "--old-stage", str(OLD_STAGE_DEFAULT),
               "--rpack", str(rpack), "--cpp-sidecar", str(cpp_sidecar),
               "--restored", str(base / "restored_attempt3"), "--grid", str(base / "grid_static256.npz"),
               "--canonical-data-root", str(CANONICAL_DEFAULT), "--force-rebuild"]
    subprocess.run(command, check=True)
    subprocess.run([py, str(code / "audit_prepared_attempt13.py"), "--base", str(base),
                    "--preparation", str(prep), "--rpack", str(rpack),
                    "--cpp-sidecar", str(cpp_sidecar), "--grid", str(base / "grid_static256.npz")], check=True)
    validate_data(p13, rpack, check_payload_files=True)
    return p13


def verify_sidecar(sidecar: Path, prepared: Path, rpack: Path) -> dict[str, object]:
    manifest = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "PASS_TRAIN_VAL_CPP_TARGET_TIME_ALIGNMENT":
        raise ValueError(f"CPP target sidecar failed: {manifest.get('state')}")
    if manifest.get("test_payloads_read") != 0 or manifest.get("buffer_payloads_read") != 0 or manifest.get("source_cot_scale") != 100.0:
        raise ValueError("CPP sidecar has wrong scale or test/buffer provenance")
    rrows = read_rows(rpack / "rows.csv")
    r_by_time = {r["time_bjt"]: i for i, r in enumerate(rrows)}
    r_target = np.load(rpack / "target.npy", mmap_mode="r", allow_pickle=False)
    r_mask = np.load(rpack / "mask.npy", mmap_mode="r", allow_pickle=False)
    all_compared = 0
    for split in ("train", "val"):
        keys = read_rows(sidecar / f"{split}_keys.csv")
        target = np.load(sidecar / f"{split}_target_physical.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(sidecar / f"{split}_mask.npy", mmap_mode="r", allow_pickle=False)
        n_expected = 4086 if split == "train" else 407
        if target.shape != (n_expected, 16, 1, 16, 16) or mask.shape != target.shape:
            raise ValueError(f"Unexpected {split} sidecar shape: {target.shape}/{mask.shape}")
        for i, row in enumerate(keys):
            times = row["target_times_bjt"].split("|")
            if len(times) != 16:
                raise ValueError("CPP target sidecar lost a future lead")
            for lead, timestamp in enumerate(times):
                ri = r_by_time.get(timestamp)
                if ri is None or rrows[ri]["split"] != split:
                    if mask[i, lead].any() or np.any(target[i, lead] != 0):
                        raise ValueError(f"Unexpected CPP target leaked across split or unmatched key {timestamp}")
                    continue
                expected_mask = np.asarray(r_mask[ri], dtype=bool)
                expected_physical = np.asarray(r_target[ri], dtype=np.float32) * np.float32(100.0)
                if not np.array_equal(mask[i, lead], expected_mask):
                    raise ValueError(f"CPP mask mismatch at {split}/{timestamp}")
                if not np.array_equal(target[i, lead], expected_physical):
                    raise ValueError(f"CPP physical COT scale/alignment mismatch at {split}/{timestamp}")
                all_compared += 1
    return {"state": "PASS_ATTEMPT13_CPP_PHYSICAL_SCALE_AND_EXACT_TIME_JOIN",
            "sidecar": str(sidecar), "sidecar_manifest_sha256": sha(sidecar / "manifest.json"),
            "physical_target_arrays_verified_against_Rpack": all_compared,
            "conversion": "R_pack target (physical COT / 100) * 100 = physical COT [0,100]",
            "test_payloads_read": 0, "buffer_payloads_read": 0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=BASE_DEFAULT)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--python", default="/public/home/slfu/miniconda3/envs/swc/bin/python")
    p.add_argument("--r-pack", type=Path, default=RPACK_DEFAULT)
    p.add_argument("--cpp-sidecar", type=Path, default=CPP_SIDECAR_DEFAULT)
    a = p.parse_args()
    if (a.attempt / "prepared_inputs.json").exists():
        out = a.attempt / "prepared_inputs.json"
        existing = json.loads(out.read_text(encoding="utf-8"))
        if (existing.get("state") != "PASS_ATTEMPT13_TRAIN_VAL_INPUTS_AND_CPP_PHYSICAL_TARGETS"
                or existing.get("test_payloads_read") != 0
                or existing.get("buffer_payloads_read") != 0):
            raise ValueError("Existing attempt13 preflight receipt is not a valid train/validation-only PASS")
        prepared = Path(existing["prepared_dir"])
        data = validate_data(prepared, a.r_pack, check_payload_files=True)
        sidecar = Path(existing["sidecar"])
        side = verify_sidecar(sidecar, prepared, a.r_pack)
        if (data["DATA_READY_sha256"] != existing["prepared_data"]["DATA_READY_sha256"]
                or data["index_sha256"] != existing["prepared_data"]["index_sha256"]
                or side["sidecar_manifest_sha256"] != existing["sidecar_audit"]["sidecar_manifest_sha256"]):
            raise ValueError("Existing attempt13 prepared inputs changed since the previous PASS")
        print(json.dumps({"state": "REUSED_VERIFIED_ATTEMPT13_PRECHECK",
                          "prepared_inputs_sha256": sha(out), "data": data, "sidecar": side,
                          "test_payloads_read": 0, "buffer_payloads_read": 0}, indent=2), flush=True)
        return
    a.attempt.mkdir(parents=True, exist_ok=True)
    prepared = build_if_cache_missing(a.base, a.attempt, a.python, a.r_pack, a.cpp_sidecar)
    data = validate_data(prepared, a.r_pack, check_payload_files=True)
    sidecar = a.attempt / "cpp_target_sidecar"
    builder = a.attempt / "build_cpp_target_sidecar_attempt13.py"
    subprocess.run([a.python, str(builder), "--prepared", str(prepared),
                    "--r-pack", str(a.r_pack), "--output", str(sidecar)], check=True)
    side = verify_sidecar(sidecar, prepared, a.r_pack)
    result = {"state": "PASS_ATTEMPT13_TRAIN_VAL_INPUTS_AND_CPP_PHYSICAL_TARGETS",
              "prepared_data": data, "sidecar_audit": side,
              "prepared_dir": str(prepared), "S_cache": data["S_cache"],
              "index": data["index"], "S_norm": data["S_norm"],
              "r_pack": str(a.r_pack), "r_checkpoint": str(a.base / "R_runs_attempt12" / "full" / "best.pt"),
              "r_norm": str(prepared / "R_packs" / "full" / "norm.json"),
              "sidecar": str(sidecar), "test_payloads_read": 0, "buffer_payloads_read": 0}
    out = a.attempt / "prepared_inputs.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
