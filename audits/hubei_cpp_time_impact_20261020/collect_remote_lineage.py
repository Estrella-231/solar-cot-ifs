"""Read-only collection of named Hubei train/validation lineage receipts.

Run on zjnu-hpc with --host zjnu or FD-201 with --host fd201.
No training, test/buffer arrays, or raw satellite/CPP payloads are opened.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "sha256": sha(path),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "json": json.loads(path.read_text(encoding="utf-8-sig"))}


def time_audit(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    differences = Counter()
    splits = Counter()
    samples = []
    unparsed = []
    for row in rows:
        assert row.get("split") in ("train", "val"), "Refuse non-train/val ledger"
        splits[row["split"]] += 1
        stamp = row.get("time_bjt", row.get("timestamp_bjt", ""))
        digits = re.sub(r"\D", "", stamp)[:12]
        cpp = row.get("path_cpp_correct_utc", row.get("cpp_path", row.get("path_cpp", row.get("path_cpp_fixed_v2", ""))))
        match = re.search(r"FY4B_AGRI_(\d{14})", cpp)
        if not match or len(digits) != 12:
            unparsed.append({"keys": list(row), "time": stamp, "cpp": cpp})
            continue
        bjt = datetime.strptime(digits, "%Y%m%d%H%M")
        source_utc = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        hours = (source_utc - (bjt - timedelta(hours=8))).total_seconds() / 3600
        differences[str(hours)] += 1
        if len(samples) < 3:
            samples.append({"time_bjt": stamp, "cpp_path": cpp,
                            "expected_cpp_utc": (bjt-timedelta(hours=8)).isoformat(),
                            "source_offset_hours": hours})
    return {"path": str(path), "exists": True, "sha256": sha(path), "rows": len(rows),
            "splits": dict(splits), "source_offset_from_correct_utc_hours": dict(differences),
            "unparsed_rows": len(unparsed), "unparsed_examples": unparsed[:1], "samples": samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", choices=("zjnu", "fd201"), required=True)
    args = p.parse_args()
    out = {"host": args.host, "collected_utc": datetime.now(timezone.utc).isoformat(),
           "scope": "named train/validation receipts and existing file hashes only",
           "test_buffer_payloads_opened": 0, "receipts": {}, "file_hashes": {}, "time_ledgers": {}}
    if args.host == "zjnu":
        root = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907")
        old = root / "experiments/hubei_r_runtime_gate_20261015_v1"
        new = root / "experiments/hubei_cloud_probability_20261020"
        rebuilt = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_al_same_time_20261017/source_rebuild_20261017_v2")
        initial_pack = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_al_same_time_20261017/pack")
        files = {
            "old_r_metadata": old / "runs/hubei_R_station16_fixedv2_v1_seed42/run_metadata.json",
            "old_r_status": old / "runs/hubei_R_station16_fixedv2_v1_seed42/status.json",
            "old_bank_22061": old / "cot_bank_full_trainval_v1/BANK_REPORT.json",
            "old_rebuilt_bank_24705": rebuilt / "r_bank/BANK_REPORT.json",
            "old_initial_ghi_pack": initial_pack / "pack_report.json",
            "new_r_pack": new / "R_pack_correctutc_attempt2/pack_status.json",
            "new_r_pack_report": new / "R_pack_correctutc_attempt2/pack_report.json",
            "new_r_pack_audit": new / "R_pack_correctutc_attempt2/independent_audit.json",
            "new_cpp_extraction": new / "cpp_attempt2/extraction_report.json",
            "new_r_metadata": new / "R_correctutc_attempt3/run_metadata.json",
            "new_r_status": new / "R_correctutc_attempt3/status.json",
            "new_r_validation": new / "R_correctutc_attempt3/validation_fp64.json",
            "new_cot_bank": new / "cot_bank_correctutc_attempt3/BANK_REPORT.json",
            "new_pcloud_bank": new / "pcloud_bank_correctutc_attempt3/BANK_REPORT.json",
            "new_ghi_pack": new / "ghi_fourarm_pack_attempt5/pack_report.json",
            "new_ghi_pack_audit": new / "audits/fourarm_pack_audit_attempt5.json",
        }
        for seed in (42, 43, 44):
            files[f"new_ghi_seed{seed}"] = new / f"ghi_fourarm_seed{seed}_attempt2/run_metadata.json"
        for name, path in files.items():
            out["receipts"][name] = receipt(path)
        hash_files = {
            "old_r_checkpoint": old / "runs/hubei_R_station16_fixedv2_v1_seed42/best.pt",
            "new_r_checkpoint": new / "R_correctutc_attempt3/best.pt",
            "new_r_target": new / "R_pack_correctutc_attempt2/target.npy",
            "new_r_mask": new / "R_pack_correctutc_attempt2/mask.npy",
            "new_r_x_norm": new / "R_pack_correctutc_attempt2/x_norm.npy",
            "new_cot_bank_array": new / "cot_bank_correctutc_attempt3/cot_pred_trainval_full.npy",
            "new_cot_bank_rows": new / "cot_bank_correctutc_attempt3/rows.csv",
        }
        for filename in ("agri_raw.npy", "geometry.npy", "ghi_wm2.npy", "clear_sky_wm2.npy", "rows.csv", "cot_pred_3x3.npy", "p_cloud_3x3.npy"):
            hash_files[f"new_ghi_pack/{filename}"] = new / "ghi_fourarm_pack_attempt5" / filename
        for name, path in hash_files.items():
            out["file_hashes"][name] = {"path": str(path), "exists": path.is_file(),
                                        "sha256": sha(path) if path.is_file() else None}
        out["time_ledgers"]["corrected_cpp"] = time_audit(new / "cpp_attempt2/rows_correct_utc.csv")
    else:
        root = Path("/home/Data_Pool/zjnu/hubei_cot_independent_20261016")
        for fold in range(5):
            pack = root / f"manifests/oof_fold_manifests_20261016_v5/folds/fold_{fold}"
            run = root / f"oof_training/R_fold{fold}_seed42_20261016"
            out["time_ledgers"][f"fold{fold}"] = time_audit(pack / "r_trainval_manifest.csv")
            for name, path in {"pack_status": pack / "r_pack/pack_status.json",
                               "run_metadata": run / "run_metadata.json",
                               "status": run / "status.json"}.items():
                out["receipts"][f"fold{fold}/{name}"] = receipt(path)
            ckpt = run / "best.pt"
            out["file_hashes"][f"fold{fold}/checkpoint"] = {
                "path": str(ckpt), "exists": ckpt.is_file(), "sha256": sha(ckpt) if ckpt.is_file() else None}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
