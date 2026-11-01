#!/usr/bin/env python3
"""Strictly audit the train/validation station-patch forecast cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def audit(cache: Path, source_rows: Path, out: Path) -> dict:
    report_path = cache / "cache_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_HUBEI_STATION_PATCH_COT_GEOMETRY_CACHE_TRAINVAL_ONLY":
        raise ValueError(f"unexpected cache state: {report.get('state')}")
    if report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError("cache report does not certify closed test/buffer")
    source_rows_hash = sha256(source_rows)
    if report["input_sources"]["r_rows_sha256"] != source_rows_hash:
        raise ValueError("cache R rows hash differs from supplied source rows")
    rows = read_csv(source_rows)
    bank_rows = read_csv(cache / "bank_rows.csv")
    if len(rows) != 22061 or len(bank_rows) != len(rows):
        raise ValueError("unexpected bank row count")
    for i, (src, got) in enumerate(zip(rows, bank_rows)):
        expected = {"index": str(i), "time_bjt": src["time_bjt"], "split": src["split"].lower()}
        if got != expected:
            raise ValueError(f"bank row order/key/split mismatch at {i}: {got} != {expected}")
    n = len(rows)
    arrays = {
        "geometry": np.load(cache / "geometry.npy", mmap_mode="r", allow_pickle=False),
        "history_cot_log": np.load(cache / "history_cot_log.npy", mmap_mode="r", allow_pickle=False),
        "history_valid": np.load(cache / "history_valid.npy", mmap_mode="r", allow_pickle=False),
        "cpp_cot": np.load(cache / "cpp_cot.npy", mmap_mode="r", allow_pickle=False),
        "cpp_valid": np.load(cache / "cpp_valid.npy", mmap_mode="r", allow_pickle=False),
    }
    expected_shapes = {
        "geometry": (n, 3, 16, 16), "history_cot_log": (n, 1, 16, 16),
        "history_valid": (n, 1, 16, 16), "cpp_cot": (n, 1, 16, 16),
        "cpp_valid": (n, 1, 16, 16),
    }
    for name, arr in arrays.items():
        if arr.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape mismatch: {arr.shape}")
        expected_sha = report["files"][f"{name}.npy"]
        if sha256(cache / f"{name}.npy") != expected_sha:
            raise ValueError(f"{name} checksum mismatch")

    # Full finite/range checks in bounded chunks.
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        g = np.asarray(arrays["geometry"][start:stop], dtype=np.float32)
        h = np.asarray(arrays["history_cot_log"][start:stop], dtype=np.float32)
        hv = np.asarray(arrays["history_valid"][start:stop], dtype=bool)
        y = np.asarray(arrays["cpp_cot"][start:stop], dtype=np.float32)
        ym = np.asarray(arrays["cpp_valid"][start:stop], dtype=bool)
        if not all(np.isfinite(x).all() for x in (g, h, y)):
            raise ValueError(f"nonfinite values in rows {start}:{stop}")
        if np.any((g[:, 0] < -1.001) | (g[:, 0] > 1.001)) or np.any((g[:, 1] < -1.001) | (g[:, 1] > 1.001)):
            raise ValueError(f"geometry cosine range error in rows {start}:{stop}")
        if np.any((g[:, 2] < 0) | (g[:, 2] > 1)):
            raise ValueError(f"day-mask range error in rows {start}:{stop}")
        if np.any((h < 0) | (h > 1.00001)) or np.any(h[~hv] != 0):
            raise ValueError(f"history COT normalization/invalid fill error in rows {start}:{stop}")
        if np.any((y < 0) | (y > 100)) or np.any(y[~ym] != 0):
            raise ValueError(f"CPP target range/invalid fill error in rows {start}:{stop}")

    pair_rows = read_csv(cache / "forecast_pairs.csv")
    by_time = {dt(r["time_bjt"]): i for i, r in enumerate(rows)}
    split_by_time = {dt(r["time_bjt"]): r["split"].lower() for r in rows}
    pair_counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    pair_seen: set[tuple[str, str, int]] = set()
    for line, r in enumerate(pair_rows, 2):
        split = r["split"].lower()
        init_t, target_t = dt(r["init_time_bjt"]), dt(r["target_time_bjt"])
        lead = int(r["lead_minutes"])
        init_i, target_i = int(r["init_index"]), int(r["target_index"])
        if split not in pair_counts or (target_t - init_t) != timedelta(minutes=lead):
            raise ValueError(f"invalid pair split/lead at line {line}")
        if by_time.get(init_t) != init_i or by_time.get(target_t) != target_i:
            raise ValueError(f"pair index/time mismatch at line {line}")
        if split_by_time[init_t] != split or split_by_time[target_t] != split:
            raise ValueError(f"pair crosses split at line {line}")
        hist = [init_t - timedelta(minutes=15 * k) for k in range(7, -1, -1)]
        if any(t not in by_time or split_by_time.get(t) != split for t in hist):
            raise ValueError(f"pair has incomplete/cross-split history at line {line}")
        nvalid = int(np.asarray(arrays["cpp_valid"][target_i, 0], dtype=bool).sum())
        if nvalid < 1 or nvalid != int(r["target_valid_pixels"]):
            raise ValueError(f"target mask/pair count mismatch at line {line}")
        key = (split, init_t.strftime("%Y%m%d%H%M"), lead)
        if key in pair_seen:
            raise ValueError(f"duplicate init/lead pair at line {line}")
        pair_seen.add(key)
        pair_counts[split]["pairs"] += 1
        pair_counts[split][f"lead_{lead}"] += 1
        pair_counts[split][f"target_date_{target_t:%Y%m%d}"] += 1
    summary = {}
    for split in ("train", "val"):
        c = pair_counts[split]
        summary[split] = {
            "pairs": c["pairs"],
            "unique_target_dates": len([k for k in c if k.startswith("target_date_")]),
            "pairs_by_lead_minutes": {str(l): c[f"lead_{l}"] for l in range(15, 241, 15)},
            "station_center_valid_pixels": {
                name: int(arrays["cpp_valid"][:, 0, row, col].sum())
                for name, (row, col) in zip(["Niushou", "Wangzhai", "Mizhuang"], [[8, 6], [9, 8], [7, 10]])
            },
        }
    val = summary["val"]
    if val["unique_target_dates"] < 12 or min(val["pairs_by_lead_minutes"].values()) < 200:
        raise ValueError(f"validation support collapsed after physical masks: {val}")
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "state": "PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED",
        "cache_report_sha256": sha256(report_path),
        "source_bank_rows_sha256": source_rows_hash,
        "trainval_rows": n,
        "forecast_pairs": summary,
        "R_train_predictions_are_oof": False,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "scope": "16x16 station-neighborhood pilot only; CPP is a retrieval reference, not independent truth.",
    }
    (out / "cache_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "CACHE_AUDIT.md").write_text(
        "# Station-patch COT forecast cache audit\n\n"
        f"State: `{result['state']}`\n\n"
        f"- Train pairs: {summary['train']['pairs']}\n"
        f"- Validation pairs: {val['pairs']} across {val['unique_target_dates']} target dates\n"
        f"- Minimum validation pairs per lead: {min(val['pairs_by_lead_minutes'].values())}\n"
        "- Train R-COT inputs are not OOF; this remains an exploratory validation pilot.\n"
        "- CPP is a retrieval reference; scope is the shared 16x16 station patch, not the full region.\n"
        "- Test/buffer payloads read: 0/0.\n", encoding="utf-8"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--source-bank-rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.cache, a.source_bank_rows, a.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
