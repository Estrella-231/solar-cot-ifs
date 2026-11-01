"""Metric primitives for the Hubei forecast evaluation pipeline.

This module is intentionally data-source agnostic. It accepts arrays and keys
already supplied by a caller; it does not open or unlock any project payload.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import re
from typing import Any

import numpy as np


def _float_vector(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _metrics_core(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    err = p - y
    rmse = float(np.sqrt(np.mean(np.square(err), dtype=np.float64)))
    mae = float(np.mean(np.abs(err), dtype=np.float64))
    bias = float(np.mean(err, dtype=np.float64))
    centered = y - float(np.mean(y, dtype=np.float64))
    sst = float(np.sum(np.square(centered), dtype=np.float64))
    r2 = None if sst == 0.0 else float(1.0 - np.sum(np.square(err), dtype=np.float64) / sst)
    mean_y = float(np.mean(y, dtype=np.float64))
    return {
        "n": int(y.size),
        "rmse_wm2": rmse,
        "mae_wm2": mae,
        "bias_wm2": bias,
        "r2": r2,
        # The project protocol defines normalized errors against mean observed GHI.
        "nrmse_mean_observed_pct": None if mean_y == 0.0 else float(100.0 * rmse / mean_y),
        "nmae_mean_observed_pct": None if mean_y == 0.0 else float(100.0 * mae / mean_y),
    }


def forecast_metrics(
    truth_wm2: Any,
    prediction_wm2: Any,
    *,
    clear_sky_wm2: Any | None = None,
    clear_sky_threshold_wm2: float | None = None,
) -> dict[str, Any]:
    """Compute the primary and supporting metrics on one exact support set.

    nRMSE/nMAE use mean observed GHI as denominator. If that denominator is
    zero, the normalized metrics are unavailable rather than silently set to 0.
    A clear-sky sensitivity is emitted only when the caller explicitly supplies
    both clear-sky values and its predeclared threshold.
    """
    y = _float_vector("truth_wm2", truth_wm2)
    p = _float_vector("prediction_wm2", prediction_wm2)
    if y.shape != p.shape:
        raise ValueError("truth and prediction must have identical shapes")
    if (y < 0).any() or (p < 0).any():
        raise ValueError("GHI truth and prediction must be nonnegative")

    result: dict[str, Any] = _metrics_core(y, p)
    if clear_sky_wm2 is None:
        if clear_sky_threshold_wm2 is not None:
            raise ValueError("clear-sky threshold requires clear_sky_wm2")
        return result
    cs = _float_vector("clear_sky_wm2", clear_sky_wm2)
    if cs.shape != y.shape:
        raise ValueError("clear-sky values must match truth and prediction shapes")
    if clear_sky_threshold_wm2 is None or not np.isfinite(clear_sky_threshold_wm2):
        raise ValueError("clear-sky threshold must be finite")
    keep = cs >= float(clear_sky_threshold_wm2)
    if keep.any():
        sensitivity = _metrics_core(y[keep], p[keep])
    else:
        sensitivity = {"n": 0, "rmse_wm2": None, "mae_wm2": None,
                       "bias_wm2": None, "r2": None,
                       "nrmse_mean_observed_pct": None,
                       "nmae_mean_observed_pct": None}
    result["clear_sky_sensitivity"] = {
        "threshold_wm2": float(clear_sky_threshold_wm2),
        **sensitivity,
    }
    return result


def metrics_by_lead(
    truth_wm2: Any,
    prediction_wm2: Any,
    lead_minutes: Any,
    *,
    clear_sky_wm2: Any | None = None,
    clear_sky_threshold_wm2: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Return metrics for every observed lead, without changing its support."""
    y = _float_vector("truth_wm2", truth_wm2)
    p = _float_vector("prediction_wm2", prediction_wm2)
    lead_raw = np.asarray(lead_minutes)
    if lead_raw.ndim != 1 or lead_raw.shape != y.shape or p.shape != y.shape:
        raise ValueError("truth, prediction, and lead vectors must have equal 1-D shapes")
    if not np.issubdtype(lead_raw.dtype, np.number) or not np.isfinite(lead_raw).all():
        raise ValueError("lead_minutes must be finite numeric values")
    if not np.equal(lead_raw, np.floor(lead_raw)).all():
        raise ValueError("lead_minutes must be integer-valued")
    leads = lead_raw.astype(np.int64)
    cs: np.ndarray | None = None
    if clear_sky_wm2 is not None:
        cs = _float_vector("clear_sky_wm2", clear_sky_wm2)
        if cs.shape != y.shape:
            raise ValueError("clear-sky values must match the other vectors")
    elif clear_sky_threshold_wm2 is not None:
        raise ValueError("clear-sky threshold requires clear_sky_wm2")

    out: dict[int, dict[str, Any]] = {}
    for lead in sorted(set(leads.tolist())):
        mask = leads == lead
        out[int(lead)] = forecast_metrics(
            y[mask], p[mask],
            clear_sky_wm2=None if cs is None else cs[mask],
            clear_sky_threshold_wm2=clear_sky_threshold_wm2 if cs is not None else None,
        )
    return out


def validate_common_key_support(groups: Mapping[str, Sequence[tuple[Any, ...]]]) -> tuple[tuple[Any, ...], ...]:
    """Require unique, identical key sets for every scored arm/seed group."""
    if len(groups) < 2:
        raise ValueError("at least two prediction groups are required")
    canonical: dict[str, set[tuple[Any, ...]]] = {}
    for name, keys in groups.items():
        seq = [tuple(key) for key in keys]
        if not seq:
            raise ValueError(f"empty key support in {name}")
        if len(seq) != len(set(seq)):
            raise ValueError(f"duplicate key in {name}")
        canonical[name] = set(seq)
    first_name = next(iter(canonical))
    ref = canonical[first_name]
    mismatches = {name: (len(keys - ref), len(ref - keys))
                  for name, keys in canonical.items() if keys != ref}
    if mismatches:
        raise ValueError(f"prediction groups do not share exact key support: {mismatches}")
    return tuple(sorted(ref))


def _weighted_rmse(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights, dtype=np.float64))
    if denom <= 0.0:
        raise ValueError("bootstrap draw has zero support for a station")
    err = p - y
    value = float(np.sqrt(np.sum(weights * np.square(err), dtype=np.float64) / denom))
    if not np.isfinite(value):
        raise ValueError("bootstrap produced a non-finite RMSE")
    return value


def _target_date_from_key(key: tuple[Any, ...]) -> str:
    if len(key) < 2:
        raise ValueError(f"prediction key must include target_time: {key}")
    value = str(key[1])
    formats = (
        (r"\d{12}", "%Y%m%d%H%M"),
        (r"\d{14}", "%Y%m%d%H%M%S"),
        (r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", "%Y-%m-%d %H:%M"),
        (r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
    )
    for pattern, fmt in formats:
        if re.fullmatch(pattern, value):
            normalized = value.replace("T", " ")
            try:
                return datetime.strptime(normalized, fmt).strftime("%Y%m%d")
            except ValueError as exc:
                raise ValueError(f"invalid target timestamp in key {key}") from exc
    raise ValueError(f"invalid target date in key {key}")


def paired_date_cluster_bootstrap_rmse_difference(
    records: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    control_arm: str,
    treatment_arm: str,
    seeds: Sequence[int],
    stations: Sequence[str],
    resamples: int = 10_000,
    random_seed: int = 20261013,
    block_length_days: int = 1,
) -> dict[str, Any]:
    """Paired calendar-day block bootstrap of equal-station/equal-seed RMSE.

    Each record requires `keys` (init, target, lead), `truth_wm2`, and
    `prediction_wm2`. Every arm/seed/station must have identical keys and truth.
    For block_length_days > 1, target dates must partition into complete,
    non-overlapping consecutive calendar-day blocks, and each station must
    have support in every block. Unsupported calendars return an explicit
    unavailable status; they are never silently dropped from the bootstrap.
    """
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not isinstance(block_length_days, int) or block_length_days < 1:
        raise ValueError("block_length_days must be a positive integer")
    if not seeds or not stations:
        raise ValueError("seeds and stations must be nonempty")
    arms = (control_arm, treatment_arm)
    prepared: dict[tuple[int, str, str], tuple[list[tuple[Any, ...]], np.ndarray, np.ndarray, list[str]]] = {}
    all_dates: set[str] = set()
    for seed in seeds:
        for station in stations:
            for arm in arms:
                key = (int(seed), str(station), str(arm))
                if key not in records:
                    raise ValueError(f"missing scored group: {key}")
                rec = records[key]
                keys = [tuple(k) for k in rec["keys"]]
                if not keys or len(keys) != len(set(keys)):
                    raise ValueError(f"empty or duplicate keys in {key}")
                truth = _float_vector(f"{key}.truth_wm2", rec["truth_wm2"])
                pred = _float_vector(f"{key}.prediction_wm2", rec["prediction_wm2"])
                if len(keys) != len(truth) or truth.shape != pred.shape:
                    raise ValueError(f"array/key length mismatch in {key}")
                if (truth < 0).any() or (pred < 0).any():
                    raise ValueError(f"negative GHI in {key}")
                dates = [_target_date_from_key(k) for k in keys]
                prepared[key] = (keys, truth, pred, dates)
                all_dates.update(dates)

    for station in stations:
        ref = prepared[(int(seeds[0]), str(station), str(control_arm))]
        for seed in seeds:
            control = prepared[(int(seed), str(station), str(control_arm))]
            treatment = prepared[(int(seed), str(station), str(treatment_arm))]
            if control[0] != treatment[0] or not np.array_equal(control[1], treatment[1]):
                raise ValueError(f"paired keys/truth differ for seed={seed}, station={station}")
            if control[0] != ref[0] or not np.array_equal(control[1], ref[1]):
                raise ValueError(f"seed support/truth differs for station={station}")

    date_list = sorted(all_dates)
    if len(date_list) < 2:
        raise ValueError("at least two target dates are required for cluster bootstrap")
    eligible_blocks: list[list[str]] = []
    if block_length_days == 1:
        eligible_blocks = [[day] for day in date_list]
    else:
        i = 0
        while i < len(date_list):
            block = date_list[i:i + block_length_days]
            if len(block) != block_length_days:
                return {
                    "status": "unavailable_dates_not_partitionable_into_complete_blocks",
                    "target_dates": date_list,
                    "block_length_days": int(block_length_days),
                    "uncovered_dates": block,
                }
            expected = [
                (datetime.strptime(block[0], "%Y%m%d").date() + timedelta(days=offset)).strftime("%Y%m%d")
                for offset in range(block_length_days)
            ]
            if block != expected:
                return {
                    "status": "unavailable_dates_not_partitionable_into_complete_blocks",
                    "target_dates": date_list,
                    "block_length_days": int(block_length_days),
                    "uncovered_dates": block,
                }
            eligible_blocks.append(block)
            i += block_length_days
        for station in stations:
            station_dates = set(prepared[(int(seeds[0]), str(station), str(control_arm))][3])
            unsupported = [block for block in eligible_blocks if not station_dates.intersection(block)]
            if unsupported:
                return {
                    "status": "unavailable_station_missing_support_in_block",
                    "target_dates": date_list,
                    "block_length_days": int(block_length_days),
                    "station": str(station),
                    "unsupported_blocks": unsupported,
                }
    date_index = {day: idx for idx, day in enumerate(date_list)}
    rng = np.random.default_rng(random_seed)
    diff = np.empty(resamples, dtype=np.float64)
    raw_scores: dict[str, list[float]] = {arm: [] for arm in arms}
    for arm in arms:
        for seed in seeds:
            station_scores = [
                _metrics_core(
                    prepared[(int(seed), str(station), str(arm))][1],
                    prepared[(int(seed), str(station), str(arm))][2],
                )["rmse_wm2"]
                for station in stations
            ]
            raw_scores[arm].append(float(np.mean(station_scores, dtype=np.float64)))
    point_difference = float(np.mean(raw_scores[treatment_arm]) - np.mean(raw_scores[control_arm]))
    for b in range(resamples):
        n_blocks = len(eligible_blocks)
        selected = rng.integers(0, len(eligible_blocks), size=n_blocks)
        sampled_dates = [day for block_index in selected
                         for day in eligible_blocks[int(block_index)]]
        multiplicity = np.zeros(len(date_list), dtype=np.int64)
        for day in sampled_dates:
            multiplicity[date_index[day]] += 1
        arm_score: dict[str, list[float]] = {arm: [] for arm in arms}
        for arm in arms:
            for seed in seeds:
                station_scores: list[float] = []
                for station in stations:
                    _, truth, pred, dates = prepared[(int(seed), str(station), str(arm))]
                    weights = np.asarray([multiplicity[date_index[day]] for day in dates], dtype=np.float64)
                    station_scores.append(_weighted_rmse(truth, pred, weights))
                arm_score[arm].append(float(np.mean(station_scores, dtype=np.float64)))
        diff[b] = float(np.mean(arm_score[treatment_arm]) - np.mean(arm_score[control_arm]))
        if not np.isfinite(diff[b]):
            raise ValueError(f"non-finite paired bootstrap replicate {b}")
    return {
        "estimand": f"{treatment_arm}-{control_arm} equal-station then equal-seed RMSE difference (W/m2)",
        "target_dates": date_list,
        "block_length_days": int(block_length_days),
        "eligible_block_count": len(eligible_blocks),
        "resamples": int(resamples),
        "random_seed": int(random_seed),
        "invalid_resamples_skipped": 0,
        "estimate_difference_wm2": point_difference,
        "difference_mean_wm2": float(np.mean(diff, dtype=np.float64)),
        "ci95_wm2": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
    }
