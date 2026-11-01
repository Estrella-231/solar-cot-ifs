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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_N = 407
EXPECTED_PRED_SHAPE = (EXPECTED_N, 16, 13, 16, 16)
STATION_RC = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LEADS_MIN = tuple(range(15, 241, 15))
FULL_FIELD_PIXELS = 256 * 256
BOOTSTRAP_REPLICATES = 2000
PREDICTED_AGRI_INDICES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]  # C01-C06, C09-C15


def bjt_date_key(value: Any) -> str:
    """Return the BJT calendar date for compact or ISO-like timestamp keys."""
    text = str(value).strip()
    if len(text) >= 8 and text[:8].isdigit():
        ymd = text[:8]
    elif len(text) >= 10 and text[4] == "-" and text[7] == "-":
        ymd = text[:10].replace("-", "")
    else:
        raise ValueError(f"Unsupported init_time_bjt date format: {text!r}")
    try:
        parsed = datetime.strptime(ymd, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid init_time_bjt calendar date: {text!r}") from exc
    return parsed.strftime("%Y-%m-%d")


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


def date_block_bootstrap_rmse_delta(
    mse_sse: np.ndarray,
    aux_sse: np.ndarray,
    dates: list[str],
    elements_per_sequence: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Paired date-block bootstrap for aux-minus-control pooled RMSE."""
    m = np.asarray(mse_sse, dtype=np.float64)
    a = np.asarray(aux_sse, dtype=np.float64)
    if m.shape != a.shape or m.shape[0] != len(dates) or m.shape[0] == 0:
        raise ValueError(f"Invalid paired SSE shapes/date count: {m.shape}, {a.shape}, {len(dates)}")
    if not np.isfinite(m).all() or not np.isfinite(a).all():
        raise ValueError("SSE arrays must be finite and align one-to-one with validation dates")
    unique_dates = sorted(set(dates))
    date_to_index = {day: i for i, day in enumerate(unique_dates)}
    row_day = np.asarray([date_to_index[d] for d in dates], dtype=np.int64)
    day_n = np.bincount(row_day, minlength=len(unique_dates)).astype(np.float64)
    tail_shape = m.shape[1:]
    m_day = np.zeros((len(unique_dates),) + tail_shape, dtype=np.float64)
    a_day = np.zeros_like(m_day)
    np.add.at(m_day, row_day, m)
    np.add.at(a_day, row_day, a)

    # Draw whole initialization dates with replacement and reuse exactly the
    # same date draws for both arms. Compact date weights keep memory bounded.
    d = len(unique_dates)
    draws = rng.integers(0, d, size=(BOOTSTRAP_REPLICATES, d))
    weights = np.zeros((BOOTSTRAP_REPLICATES, d), dtype=np.float64)
    boot_rows = np.repeat(np.arange(BOOTSTRAP_REPLICATES), d)
    np.add.at(weights, (boot_rows, draws.reshape(-1)), 1.0)
    boot_n = weights @ day_n
    m_boot = (weights @ m_day.reshape(d, -1)).reshape((BOOTSTRAP_REPLICATES,) + tail_shape)
    a_boot = (weights @ a_day.reshape(d, -1)).reshape((BOOTSTRAP_REPLICATES,) + tail_shape)
    denom = boot_n.reshape((BOOTSTRAP_REPLICATES,) + (1,) * len(tail_shape)) * float(elements_per_sequence)
    m_rmse_boot = np.sqrt(m_boot / denom)
    a_rmse_boot = np.sqrt(a_boot / denom)
    delta_boot = a_rmse_boot - m_rmse_boot
    relative_boot = np.divide(100.0 * delta_boot, m_rmse_boot,
                              out=np.full_like(delta_boot, np.nan), where=m_rmse_boot > 0)

    total_n = len(dates) * float(elements_per_sequence)
    m_rmse = np.sqrt(np.sum(m, axis=0, dtype=np.float64) / total_n)
    a_rmse = np.sqrt(np.sum(a, axis=0, dtype=np.float64) / total_n)
    delta = a_rmse - m_rmse
    relative = np.divide(100.0 * delta, m_rmse,
                         out=np.full_like(delta, np.nan), where=m_rmse > 0)
    return {
        "mse_rmse": m_rmse,
        "cot_aux_rmse": a_rmse,
        "delta_rmse_aux_minus_mse": delta,
        "relative_change_pct_positive_is_worse": relative,
        "delta_ci95_low": np.quantile(delta_boot, 0.025, axis=0),
        "delta_ci95_high": np.quantile(delta_boot, 0.975, axis=0),
        "relative_ci95_low": np.nanquantile(relative_boot, 0.025, axis=0),
        "relative_ci95_high": np.nanquantile(relative_boot, 0.975, axis=0),
        "n_date_blocks": len(unique_dates),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "BJT initialization date; paired dates sampled identically for both arms",
    }


def load_arm(root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], dict[str, Any], dict[str, np.ndarray]]:
    val = root / "val"
    report_path = root / "BANK_REPORT.json"
    pred_path = val / "pred_agri_patch.npy"
    truth_path = val / "true_agri_patch.npy"
    geom_path = val / "future_geometry_patch.npy"
    keys_path = val / "keys.csv"
    error_sums_path = val / "full_field_error_sums.npz"
    for path in (report_path, pred_path, truth_path, geom_path, keys_path, error_sums_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = read_json(report_path)
    full_metric = report_full_field(report, report_path)
    pred = np.load(pred_path, allow_pickle=False)
    truth = np.load(truth_path, allow_pickle=False)
    geom = np.load(geom_path, allow_pickle=False)
    with np.load(error_sums_path, allow_pickle=False) as z:
        error_sums = {k: np.asarray(z[k]).copy() for k in ("normalized_sse", "physical_sse", "physical_sae", "physical_bias")}
        pixels_per_sequence = int(np.asarray(z["pixels_per_sequence"]).item())
    if pred.shape != EXPECTED_PRED_SHAPE or truth.shape != EXPECTED_PRED_SHAPE:
        raise ValueError(f"Unexpected AGRI patch shapes in {root}: {pred.shape}, {truth.shape}")
    if geom.shape != (EXPECTED_N, 16, 3, 16, 16):
        raise ValueError(f"Unexpected future geometry shape in {root}: {geom.shape}")
    for key, values in error_sums.items():
        if values.shape != (EXPECTED_N, 16, 13) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key} sufficient statistics in {error_sums_path}: {values.shape}/{values.dtype}")
    if pixels_per_sequence != FULL_FIELD_PIXELS:
        raise ValueError(f"Unexpected full-field pixel count in {error_sums_path}: {pixels_per_sequence}")
    expected_hash = report.get("full_field_validation_metrics", {}).get("val", {}).get("per_sample_error_sums_sha256")
    if expected_hash != sha256(error_sums_path):
        raise ValueError(f"Per-sample full-field error sums hash does not match BANK_REPORT: {error_sums_path}")
    return report, pred, truth, geom, read_keys(keys_path), {
        "bank_report": sha256(report_path), "pred_agri_patch": sha256(pred_path),
        "true_agri_patch": sha256(truth_path), "future_geometry_patch": sha256(geom_path),
        "keys": sha256(keys_path), "full_field_error_sums": sha256(error_sums_path),
        "full_field_metric": full_metric,
    }, error_sums


def audit(mse_root: Path, aux_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mse = load_arm(mse_root)
    aux = load_arm(aux_root)
    rm, pm, ym, gm, km, hm, em = mse
    ra, pa, ya, ga, ka, ha, ea = aux

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

    date_keys = [bjt_date_key(row["init_time_bjt"]) for row in km]
    if em["physical_sse"].shape != ea["physical_sse"].shape:
        raise RuntimeError("MSE/COT-aux full-field sufficient-statistic shapes differ")

    normalized = date_block_bootstrap_rmse_delta(
        em["normalized_sse"].sum(axis=(1, 2), dtype=np.float64),
        ea["normalized_sse"].sum(axis=(1, 2), dtype=np.float64),
        date_keys, 16 * 13 * FULL_FIELD_PIXELS, np.random.default_rng(13042),
    )
    full_field = date_block_bootstrap_rmse_delta(
        em["physical_sse"], ea["physical_sse"], date_keys,
        FULL_FIELD_PIXELS, np.random.default_rng(13043),
    )
    full_field_rows = []
    for li, lead in enumerate(LEADS_MIN):
        for ci, input_channel_index in enumerate(PREDICTED_AGRI_INDICES):
            full_field_rows.append({
                "channel": f"C{input_channel_index + 1:02d}",
                "channel_index_in_13ch_output": ci,
                "lead_minutes": lead,
                "mse_rmse_original_agri_unit": float(full_field["mse_rmse"][li, ci]),
                "cot_aux_rmse_original_agri_unit": float(full_field["cot_aux_rmse"][li, ci]),
                "delta_rmse_aux_minus_mse_original_agri_unit": float(full_field["delta_rmse_aux_minus_mse"][li, ci]),
                "relative_change_pct_positive_is_worse": float(full_field["relative_change_pct_positive_is_worse"][li, ci]),
                "delta_ci95_low": float(full_field["delta_ci95_low"][li, ci]),
                "delta_ci95_high": float(full_field["delta_ci95_high"][li, ci]),
                "relative_ci95_low_pct": float(full_field["relative_ci95_low"][li, ci]),
                "relative_ci95_high_pct": float(full_field["relative_ci95_high"][li, ci]),
                "n_date_blocks": full_field["n_date_blocks"],
            })

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
    m_norm_rmse = float(normalized["mse_rmse"])
    a_norm_rmse = float(normalized["cot_aux_rmse"])
    result = {
        "state": "ATTEMPT13_VALIDATION_AGRI_PAIRED_GUARD_COMPLETE",
        "scope": "407 validation windows; full-field per-sequence error sums plus physical AGRI in saved 16x16 patches; same val keys, targets, geometry, normalization and index",
        "interpretation": "Normalized full-field RMSE is a pooled scale-free aggregate. Physical AGRI is reported per channel and lead because channel units/scales are not pooled. Date-block intervals describe validation uncertainty, not independent test evidence.",
        "full_field_normalized_space_paired": {
            "mse_control_rmse": m_norm_rmse,
            "cot_aux_rmse": a_norm_rmse,
            "delta_rmse_aux_minus_mse": float(normalized["delta_rmse_aux_minus_mse"]),
            "relative_change_pct_positive_is_worse": float(normalized["relative_change_pct_positive_is_worse"]),
            "delta_ci95_low": float(normalized["delta_ci95_low"]),
            "delta_ci95_high": float(normalized["delta_ci95_high"]),
            "relative_ci95_low_pct": float(normalized["relative_ci95_low"]),
            "relative_ci95_high_pct": float(normalized["relative_ci95_high"]),
            "element_count": int(mm["element_count"]),
            "n_date_blocks": normalized["n_date_blocks"],
            "bootstrap_replicates": normalized["bootstrap_replicates"],
            "bootstrap_unit": normalized["bootstrap_unit"],
        },
        "full_field_physical_per_channel_lead": {
            "metric_units": "original AGRI units; one channel and one lead at a time",
            "pixels_per_sequence": FULL_FIELD_PIXELS,
            "n_date_blocks": full_field["n_date_blocks"],
            "bootstrap_replicates": full_field["bootstrap_replicates"],
            "bootstrap_unit": full_field["bootstrap_unit"],
            "rows": full_field_rows,
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
    with (a.output / "full_field_agri_paired_delta_by_channel_lead.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(result["full_field_physical_per_channel_lead"]["rows"][0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(result["full_field_physical_per_channel_lead"]["rows"])
    print(json.dumps({k: result[k] for k in ("state", "common_validation_sequences", "full_field_normalized_space_paired", "test_payloads_read", "buffer_payloads_read")}, indent=2))


if __name__ == "__main__":
    main()
