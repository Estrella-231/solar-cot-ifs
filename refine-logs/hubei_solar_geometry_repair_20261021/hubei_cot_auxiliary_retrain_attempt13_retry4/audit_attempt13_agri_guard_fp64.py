#!/usr/bin/env python3
"""Independent, read-only AGRI safeguard audit for attempt13 validation banks.

The normal attempt13 scorer evaluates COT only. This audit checks whether the
auxiliary-loss arm damages its AGRI forecast on exactly the same validation
sequences. Stored 16x16 physical AGRI patches support a local physical-unit
comparison; BANK_REPORT also supplies a full-field normalized-space metric.
It deliberately does not mislabel that normalized metric as physical RMSE.

No training, checkpoint selection, CPP access, or test/buffer access occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_N = 407
EXPECTED_PRED_SHAPE = (EXPECTED_N, 16, 13, 16, 16)
STATION_RC = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LEADS_MIN = tuple(range(15, 241, 15))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_keys(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} validation keys in {path}, found {len(rows)}")
    if any(row.get("split") != "val" for row in rows):
        raise ValueError(f"Non-validation row found in {path}")
    ids = [row.get("sample_index", "") for row in rows]
    if len(set(ids)) != EXPECTED_N:
        raise ValueError(f"Duplicate/empty sample_index in {path}")
    return rows


def report_full_field(report: dict[str, Any], path: Path) -> dict[str, Any]:
    if report.get("state") != "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK":
        raise ValueError(f"Unexpected bank state in {path}")
    if report.get("inferred_splits") != ["val"]:
        raise ValueError(f"Expected val-only inference in {path}")
    if report.get("sequences_written_by_split", {}).get("val") != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} validation sequences in {path}")
    if report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError(f"Bank report indicates test/buffer reads: {path}")
    counts = report.get("split_counts", {})
    if counts.get("test") != 0 or counts.get("buffer") != 0:
        raise ValueError(f"Index contains test/buffer rows: {path}")
    metric = report.get("full_field_validation_metrics", {}).get("val")
    if not isinstance(metric, dict) or metric.get("element_count", 0) <= 0:
        raise ValueError(f"Missing full-field validation metric in {path}")
    if metric.get("sequence_count") != EXPECTED_N:
        raise ValueError(f"Full-field metric has unexpected sequence count in {path}")
    return metric


def rmse_rows(pred: np.ndarray, truth: np.ndarray, scope: str) -> list[dict[str, Any]]:
    """Return float64 physical AGRI RMSE by forecast lead for a selected crop."""
    rows: list[dict[str, Any]] = []
    for li, lead in enumerate(LEADS_MIN):
        p = np.asarray(pred[:, li], dtype=np.float64)
        y = np.asarray(truth[:, li], dtype=np.float64)
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite physical AGRI values in {scope}, lead={lead}")
        err = p - y
        rows.append({
            "scope": scope,
            "lead_minutes": lead,
            "n_values": int(err.size),
            "rmse_wm2": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
            "mae_wm2": float(np.mean(np.abs(err), dtype=np.float64)),
            "bias_wm2": float(np.mean(err, dtype=np.float64)),
        })
    return rows


def load_arm(root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], dict[str, Any]]:
    val = root / "val"
    report_path = root / "BANK_REPORT.json"
    pred_path = val / "pred_agri_patch.npy"
    truth_path = val / "true_agri_patch.npy"
    geom_path = val / "future_geometry_patch.npy"
    keys_path = val / "keys.csv"
    for path in (report_path, pred_path, truth_path, geom_path, keys_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = read_json(report_path)
    full_metric = report_full_field(report, report_path)
    pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    geom = np.load(geom_path, mmap_mode="r", allow_pickle=False)
    if pred.shape != EXPECTED_PRED_SHAPE or truth.shape != EXPECTED_PRED_SHAPE:
        raise ValueError(f"Unexpected AGRI patch shapes in {root}: {pred.shape}, {truth.shape}")
    if geom.shape != (EXPECTED_N, 16, 3, 16, 16):
        raise ValueError(f"Unexpected future geometry shape in {root}: {geom.shape}")
    return report, pred, truth, geom, read_keys(keys_path), {
        "bank_report": sha256(report_path), "pred_agri_patch": sha256(pred_path),
        "true_agri_patch": sha256(truth_path), "future_geometry_patch": sha256(geom_path),
        "keys": sha256(keys_path), "full_field_metric": full_metric,
    }


def audit(mse_root: Path, aux_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mse = load_arm(mse_root)
    aux = load_arm(aux_root)
    rm, pm, ym, gm, km, hm = mse
    ra, pa, ya, ga, ka, ha = aux

    # Both arms must use the exact frozen evaluation contract and truth support.
    for field in ("normalization_sha256", "index_sha256", "patch_origin_crop_256", "patch_size", "lead_minutes"):
        if rm.get(field) != ra.get(field):
            raise RuntimeError(f"MSE/COT-aux bank contract differs in {field}")
    if km != ka:
        raise RuntimeError("MSE and COT-aux validation keys or row order differ")
    if not np.array_equal(ym, ya):
        raise RuntimeError("Validation AGRI targets differ between arms")
    if not np.array_equal(gm, ga):
        raise RuntimeError("Future solar-geometry patches differ between arms")
    if not all(np.isfinite(np.asarray(x)).all() for x in (pm, pa, ym, gm)):
        raise ValueError("Non-finite data found in one of the common validation banks")

    # Build named per-lead rows directly, retaining physical AGRI units.
    paired: list[dict[str, Any]] = []
    scopes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        "all_16x16_patch_13_channels": (pm, ym, pa, ya),
    }
    for name, (row, col) in STATION_RC.items():
        sl = (slice(None), slice(None), slice(None), slice(row - 1, row + 2), slice(col - 1, col + 2))
        scopes[f"{name}_3x3"] = (pm[sl], ym[sl], pa[sl], ya[sl])
    records = []
    for scope, (mp, my, ap, ay) in scopes.items():
        mrows = rmse_rows(mp, my, scope)
        arows = rmse_rows(ap, ay, scope)
        for mr, ar in zip(mrows, arows):
            records.extend(({"arm": "mse_control", **mr}, {"arm": "cot_aux", **ar}))
            delta = ar["rmse_wm2"] - mr["rmse_wm2"]
            paired.append({
                "scope": scope,
                "lead_minutes": mr["lead_minutes"],
                "mse_rmse_wm2": mr["rmse_wm2"],
                "cot_aux_rmse_wm2": ar["rmse_wm2"],
                "delta_rmse_wm2_aux_minus_mse": delta,
                "relative_change_pct_positive_is_worse": 100.0 * delta / mr["rmse_wm2"] if mr["rmse_wm2"] else None,
            })

    mm = hm["full_field_metric"]
    am = ha["full_field_metric"]
    if mm.get("element_count") != am.get("element_count"):
        raise RuntimeError("Full-field normalized metric element counts differ")
    m_norm_rmse = float(np.sqrt(mm["normalized_mse"]))
    a_norm_rmse = float(np.sqrt(am["normalized_mse"]))
    result = {
        "state": "ATTEMPT13_VALIDATION_AGRI_PATCH_AUDIT_COMPLETE_FULL_FIELD_PHYSICAL_GATE_UNRESOLVED",
        "scope": "407 validation windows; physical AGRI in saved 16x16 patches; same val keys, targets, geometry, normalization and index",
        "interpretation": "AGRI physical RMSE is local-patch only. The available full-field metric is normalized-space RMSE and is reported separately; it is not physical-unit full-field RMSE and cannot satisfy the preregistered full-field physical AGRI gate.",
        "full_field_normalized_space": {
            "mse_control_rmse": m_norm_rmse,
            "cot_aux_rmse": a_norm_rmse,
            "relative_change_pct_positive_is_worse": 100.0 * (a_norm_rmse - m_norm_rmse) / m_norm_rmse if m_norm_rmse else None,
            "element_count": int(mm["element_count"]),
            "physical_full_field_guard": "NOT_EVALUATED: source inference bank stores only 16x16 physical patches and a pooled normalized full-field summary",
        },
        "common_validation_sequences": EXPECTED_N,
        "same_targets_exact": True,
        "same_future_geometry_exact": True,
        "same_keys_and_order_exact": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "input_hashes": {"mse_control": hm, "cot_aux": ha},
        "paired_rmse_rows": paired,
    }
    return result, records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mse-bank", type=Path, required=True, help="MSE-control SimVP bank root")
    p.add_argument("--cot-aux-bank", type=Path, required=True, help="COT-aux SimVP bank root")
    p.add_argument("--output", type=Path, required=True, help="New output directory; refuses overwrite")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    result, rows = audit(a.mse_bank, a.cot_aux_bank)
    a.output.mkdir(parents=True)
    (a.output / "agri_guard_fp64.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (a.output / "agri_rmse_by_arm_lead.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scope", "arm", "lead_minutes", "n_values", "rmse_wm2", "mae_wm2", "bias_wm2"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (a.output / "agri_paired_delta_by_lead.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scope", "lead_minutes", "mse_rmse_wm2", "cot_aux_rmse_wm2", "delta_rmse_wm2_aux_minus_mse", "relative_change_pct_positive_is_worse"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(result["paired_rmse_rows"])
    print(json.dumps({k: result[k] for k in ("state", "common_validation_sequences", "full_field_normalized_space", "test_payloads_read", "buffer_payloads_read")}, indent=2))


if __name__ == "__main__":
    main()
