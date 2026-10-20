#!/usr/bin/env python3
"""Read-only integrity audit for corrected Hubei SimVP validation forecast banks."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

BASE = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021")
EXPECTED = {"full": 407, "fold_0": 823, "fold_1": 797, "fold_2": 850, "fold_3": 809, "fold_4": 807}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def audit(base: Path, require_all: bool) -> dict:
    sys.path.insert(0, str(base / "code_pipeline_attempt12"))
    import audit_oof_bank_lineage as keys_audit
    checked = {}
    for role, expected_n in EXPECTED.items():
        ready_path = base / "S_runs_attempt12" / role / "S_BANK_READY.json"
        if not ready_path.is_file():
            continue
        ready = json.loads(ready_path.read_text())
        assert ready["state"] == "CORRECTED_SOLAR_S_BANK_COMPLETE"
        assert ready["solar_geometry_corrected"] is True
        assert ready["test_payloads_read"] == 0
        if role == "full":
            index_path = base / "prepared_attempt12/indexes/full.csv"
            index_rows = [r for r in keys_audit.rows(index_path) if r["split"] == "val"]
        else:
            index_path = base / "prepared_attempt12/indexes" / f"{role}_oof.csv"
            index_rows = keys_audit.rows(index_path)
        bank_dir = base / "S_banks_attempt12" / role / "val"
        key_path = bank_dir / "keys.csv"
        bank_rows = keys_audit.rows(key_path)
        assert len(index_rows) == len(bank_rows) == expected_n, (role, len(index_rows), len(bank_rows), expected_n)
        for i, (src, bank) in enumerate(zip(index_rows, bank_rows)):
            init, targets = keys_audit.bjt_sequence(src)
            assert bank["split"] == "val" and bank["init_time_bjt"] == init
            assert bank["target_times_bjt"].split("|") == targets, (role, i)
        report = json.loads((base / "S_banks_attempt12" / role / "BANK_REPORT.json").read_text())
        assert report["test_payloads_read"] == report["buffer_payloads_read"] == 0
        assert report["sequences_written_by_split"] == {"train": 0, "val": expected_n}
        arrays = {}
        for name, shape in (("pred_agri_patch.npy", (expected_n, 16, 13, 16, 16)),
                            ("true_agri_patch.npy", (expected_n, 16, 13, 16, 16)),
                            ("future_geometry_patch.npy", (expected_n, 16, 3, 16, 16))):
            path = bank_dir / name
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            assert arr.shape == shape and np.isfinite(arr).all(), (role, name, arr.shape)
            arrays[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": True, "sha256": sha(path)}
        checked[role] = {"n_sequences": expected_n, "index_sha256": sha(index_path),
                         "keys_sha256": sha(key_path), "checkpoint_sha256": ready["checkpoint_sha256"],
                         "normalization_sha256": ready["normalization_sha256"],
                         "utc_plus8_time_keys_match": True, "arrays": arrays}
    if require_all:
        assert set(checked) == set(EXPECTED), (set(checked), set(EXPECTED))
    return {"state": "PASS_ALL_CORRECTED_SOLAR_BANKS" if len(checked) == len(EXPECTED) else "PASS_PARTIAL_CORRECTED_SOLAR_BANKS",
            "checked_roles": checked, "ready_roles": sorted(checked), "expected_roles": list(EXPECTED),
            "all_ready": len(checked) == len(EXPECTED), "test_payloads_read": 0, "buffer_payloads_read": 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.base, args.require_all)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
