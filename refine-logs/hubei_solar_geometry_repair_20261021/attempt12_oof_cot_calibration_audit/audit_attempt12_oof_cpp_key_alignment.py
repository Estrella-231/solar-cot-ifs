#!/usr/bin/env python3
"""Read-only remote audit of attempt12 OOF forecast-COT/CPP key alignment.

Run from Windows with:
  Get-Content -Raw this_file | ssh zjnu-hpc 'python3 -'

Only CSV/JSON metadata and NPY headers are read. No NPY payload is loaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021")
ROLES = [f"fold_{i}" for i in range(5)] + ["full"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def npy_header(path: Path):
    # Read only the NPY header; deliberately do not deserialize array values.
    import numpy as np

    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    return {"shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran), "file_bytes": path.stat().st_size}


def main():
    date_folds = {}
    for r in rows(BASE / "preparation_attempt1/metadata/train_date_folds.csv"):
        date_folds[r["date_BJT"].replace("-", "")] = int(r["heldout_fold"])
    manifest_path = BASE / "COT_banks_attempt12/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_roles = {x["role"]: x for x in manifest["source_roles"]}
    r_pack_rows_path = BASE / "prepared_attempt12/R_packs/full/rows.csv"
    r_pack_rows = rows(r_pack_rows_path)
    r_pack_target = BASE / "prepared_attempt12/R_packs/full/target.npy"
    r_pack_mask = BASE / "prepared_attempt12/R_packs/full/mask.npy"
    output = {
        "audit_state": "READ_ONLY_METADATA_AND_HEADER_ALIGNMENT_AUDIT",
        "remote_base": str(BASE),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "array_payloads_loaded": False,
        "date_fold_assignment": {
            "path": str(BASE / "preparation_attempt1/metadata/train_date_folds.csv"),
            "sha256": sha(BASE / "preparation_attempt1/metadata/train_date_folds.csv"),
            "rows": len(date_folds),
        },
        "cot_bank_manifest": {
            "path": str(manifest_path),
            "sha256": sha(manifest_path),
            "state": manifest.get("state"),
            "test_payloads_read": manifest.get("test_payloads_read"),
            "buffer_payloads_read": manifest.get("buffer_payloads_read"),
        },
        "full_r_pack_reference": {
            "rows_path": str(r_pack_rows_path),
            "rows_sha256": sha(r_pack_rows_path),
            "total_rows": len(r_pack_rows),
            "train_rows": sum(x.get("split") == "train" for x in r_pack_rows),
            "validation_rows": sum(x.get("split") == "val" for x in r_pack_rows),
            "target_path": str(r_pack_target),
            "target_header": npy_header(r_pack_target),
            "mask_path": str(r_pack_mask),
            "mask_header": npy_header(r_pack_mask),
            "pack_sha_manifest_path": str(BASE / "prepared_attempt12/R_packs/full/SHA256.json"),
            "pack_sha_manifest_sha256": sha(BASE / "prepared_attempt12/R_packs/full/SHA256.json"),
            "pack_sha_manifest": json.loads((BASE / "prepared_attempt12/R_packs/full/SHA256.json").read_text(encoding="utf-8")),
            "alignment_contract": "build_corrected_data.py appends each rows.csv record in the same order as target[ii] and mask[ii,None]; rows are keyed by time_bjt and index is the role-pack row index",
        },
        "roles": {},
    }

    r_pack_val_supervised = [x for x in r_pack_rows if x.get("split") == "val" and int(x.get("valid_cot_pixels", 0)) > 0]
    r_run_val_key_path = BASE / "R_runs_attempt12/full/val_keys.csv"
    r_run_val_keys = rows(r_run_val_key_path)
    r_run_val_times = [x["time_bjt"] for x in r_run_val_keys]
    r_pack_val_times = [x["time_bjt"] for x in r_pack_val_supervised]
    output["full_r_pack_reference"]["r_run_full_val_key_crosscheck"] = {
        "R_run_val_key_path": str(r_run_val_key_path),
        "R_run_val_key_sha256": sha(r_run_val_key_path),
        "R_run_val_key_rows": len(r_run_val_times),
        "R_pack_val_supervised_rows": len(r_pack_val_times),
        "same_time_set": set(r_run_val_times) == set(r_pack_val_times),
        "same_order": r_run_val_times == r_pack_val_times,
        "duplicate_R_run_val_time_keys": len(r_run_val_times) - len(set(r_run_val_times)),
    }

    for role in ROLES:
        bank_role = role if role != "full" else "validation"
        s_key = BASE / "S_banks_attempt12" / role / "val/keys.csv"
        s_index = BASE / "prepared_attempt12/indexes" / ("full.csv" if role == "full" else f"{role}_oof.csv")
        rdir = BASE / "R_runs_attempt12" / role
        if role == "full":
            r_key = rdir / "val_keys.csv"
            rrows = rows(r_key)
            mask_path = rdir / "val_valid_mask.npy"
            cpp_path = rdir / "val_reference_cpp_cot.npy"
            r_key_source = "R_runs_attempt12/full validation keys/reference/mask"
        else:
            # R_runs/fold_i/val_* are the shared external validation set.
            # OOF training target references live in full R pack train rows
            # belonging to this held-out train-date fold.
            r_key = r_pack_rows_path
            rrows = [x for x in r_pack_rows if x.get("split") == "train" and date_folds.get(x["time_bjt"][:8]) == int(role[-1])]
            mask_path = r_pack_mask
            cpp_path = r_pack_target
            r_key_source = "prepared_attempt12/R_packs/full train rows filtered to held-out date fold; target.npy and mask.npy use rows.csv index order"
        cot_path = BASE / "COT_banks_attempt12" / f"{bank_role}.npy"
        srows = rows(s_key)
        index_rows = rows(s_index)
        if role != "full" and any(x.get("split") != "val" for x in index_rows):
            raise ValueError(f"{role}: non-val index rows found")
        if role == "full":
            index_rows = [x for x in index_rows if x.get("split") == "val"]
        r_times = [x["time_bjt"] for x in rrows]
        r_counts = Counter(r_times)
        r_unique = set(r_times)
        r_ordered = r_times == sorted(r_times)
        s_inits = [x["init_time_bjt"] for x in srows]
        s_init_counts = Counter(s_inits)
        frames = []
        bad_lead = []
        frame_dates = set()
        init_fold_mismatch = 0
        target_fold_mismatch = 0
        unknown_dates = 0
        for k in srows:
            init = k["init_time_bjt"]
            targets = k["target_times_bjt"].split("|")
            if len(targets) != 16:
                raise ValueError(f"{role} {init}: expected 16 targets, got {len(targets)}")
            if role != "full":
                if date_folds.get(init[:8]) != int(role[-1]):
                    init_fold_mismatch += 1
            ti = datetime.strptime(init, "%Y%m%d%H%M")
            for j, target in enumerate(targets):
                lead = (j + 1) * 15
                if datetime.strptime(target, "%Y%m%d%H%M") != ti + timedelta(minutes=lead):
                    bad_lead.append({"init": init, "target": target, "lead_minutes": lead})
                frame_dates.add(target[:8])
                if role != "full":
                    fold = date_folds.get(target[:8])
                    if fold is None:
                        unknown_dates += 1
                    elif fold != int(role[-1]):
                        target_fold_mismatch += 1
                frames.append((init, target, lead))
        matched = [x for x in frames if x[1] in r_unique]
        unmatched = [x for x in frames if x[1] not in r_unique]
        matched_targets = {x[1] for x in matched}
        extra_r_times = sorted(r_unique - {x[1] for x in frames})
        duplicate_frame_tuples = len(frames) - len(set(frames))
        duplicate_s_sequences = len(s_inits) - len(set(s_inits))
        bank_header = npy_header(cot_path)
        cpp_header = npy_header(cpp_path)
        mask_header = npy_header(mask_path)
        run_meta_path = rdir / "run_metadata.json"
        r_status_path = rdir / "status.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        r_status = json.loads(r_status_path.read_text(encoding="utf-8"))
        entry = manifest_roles[bank_role]
        r_fold_val_key_path = rdir / "val_keys.csv"
        r_fold_val_keys = rows(r_fold_val_key_path)
        r_fold_val_times = {x["time_bjt"] for x in r_fold_val_keys}
        if role == "full":
            cpp_valid_times = {x["time_bjt"] for x in r_pack_val_supervised}
            cpp_supervision_row_source = "R_packs/full rows.csv split=val with valid_cot_pixels>0, cross-checked to R_runs/full/val_keys.csv"
        else:
            cpp_valid_times = {x["time_bjt"] for x in rrows if int(x.get("valid_cot_pixels", 0)) > 0}
            cpp_supervision_row_source = "R_packs/full rows.csv train split and same held-out date fold; valid_cot_pixels from audited CPP/SZA80 mask"
        matched_valid = [x for x in matched if x[1] in cpp_valid_times]
        matched_no_valid = [x for x in matched if x[1] not in cpp_valid_times]
        role_result = {
            "role": role,
            "bank_role_manifest_name": bank_role,
            "s_index": {"path": str(s_index), "rows": len(index_rows), "sha256": sha(s_index)},
            "s_key_csv": {"path": str(s_key), "rows": len(srows), "sha256": sha(s_key), "duplicate_init_keys": duplicate_s_sequences,
                          "init_sorted": s_inits == sorted(s_inits)},
            "forecast_frame_keys": {"rows": len(frames), "expected_rows": len(srows) * 16,
                                    "duplicate_init_target_lead_tuples": duplicate_frame_tuples,
                                    "bad_canonical_lead_rows": len(bad_lead),
                                    "unique_target_times": len({x[1] for x in frames}),
                                    "date_fold_init_mismatches": init_fold_mismatch,
                                    "date_fold_target_mismatches": target_fold_mismatch,
                                    "target_dates_missing_from_fold_assignment": unknown_dates},
            "r_cpp_key_csv": {"path": str(r_key), "rows": len(rrows), "sha256": sha(r_key),
                              "key_source": r_key_source,
                              "duplicate_time_keys": sum(n - 1 for n in r_counts.values() if n > 1),
                              "unique_time_keys": len(r_unique), "time_sorted": r_ordered,
                              "split_values": sorted({x.get("split", "") for x in rrows})},
            "r_run_fold_validation_key_separation": {
                "path": str(r_fold_val_key_path),
                "sha256": sha(r_fold_val_key_path),
                "rows": len(r_fold_val_keys),
                "distinct_target_times": len(r_fold_val_times),
                "OOF_forecast_frames_matching_this_validation_keyset": sum(x[1] in r_fold_val_times for x in frames),
                "interpretation": "R_runs/fold_i/val_keys are shared external validation keys, not the held-out training-fold CPP labels; use R_packs/full train rows filtered to the held-out fold for OOF calibration targets",
            },
            "keyed_alignment": {"matched_forecast_frames": len(matched), "unmatched_forecast_frames": len(unmatched),
                                "coverage_percent": round(100 * len(matched) / len(frames), 6) if frames else 0,
                                "matched_with_nonempty_cpp_mask": len(matched_valid),
                                "matched_but_zero_valid_cpp_pixels": len(matched_no_valid),
                                "unique_matched_cpp_times": len(matched_targets),
                                "cpp_times_not_used_by_any_forecast_frame": len(extra_r_times),
                                "positional_order_equal": False,
                                "join_rule": "forecast COT frame (init_time_bjt,target_times_bjt[j],lead=15*(j+1)) joins by target_time_bjt to the same held-out-fold R CPP key/mask row; use target-date folds, never row-position pairing",
                                "cpp_supervision_row_source": cpp_supervision_row_source},
            "cpp_files_metadata_only": {"reference_path": str(cpp_path), "reference_header": cpp_header,
                                         "mask_path": str(mask_path), "mask_header": mask_header,
                                         "cpp_rows_match_key_rows": (cpp_header["shape"][0] == len(r_pack_rows) if role != "full" else cpp_header["shape"][0] == len(rrows)),
                                         "mask_rows_match_key_rows": (mask_header["shape"][0] == len(r_pack_rows) if role != "full" else mask_header["shape"][0] == len(rrows))},
            "cot_bank": {"path": str(cot_path), "header_only": bank_header,
                         "manifest_sha256": entry["COT_sha256"],
                         "manifest_pred_agri_sha256": entry["forecast_pred_sha256"],
                         "manifest_future_geometry_sha256": entry["forecast_geometry_sha256"],
                         "manifest_source_roles_sha256": entry["COT_sha256"],
                         "S_checkpoint_sha256": entry["corrected_S_checkpoint_sha256"],
                         "R_checkpoint_sha256": entry["R_checkpoint_sha256"],
                         "R_normalization_sha256": entry["R_normalization_sha256"],
                         "R_inference_code_sha256": entry["R_inference_code_sha256"]},
            "R_run": {"status_path": str(r_status_path), "status": r_status.get("state"),
                      "best_epoch": r_status.get("best_epoch"), "run_metadata_path": str(run_meta_path),
                      "run_metadata_sha256": sha(run_meta_path),
                      "status_sha256": sha(r_status_path),
                      "pack_status_sha256": run_meta.get("pack_status_sha256"),
                      "val_reference_cpp_sha256_metadata": run_meta.get("val_reference_cpp_sha256"),
                      "val_valid_mask_sha256_metadata": run_meta.get("val_valid_mask_sha256")},
        }
        output["roles"][role] = role_result
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
