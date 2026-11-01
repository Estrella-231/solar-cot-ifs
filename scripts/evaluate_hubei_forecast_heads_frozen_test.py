#!/usr/bin/env python3
"""Synthetic-only pre-release scorer prototype; not approved for live test data.

This prototype verifies an independently hashed, station-keyed truth manifest,
keeps truth arrays separate from prediction bundles, and consumes a one-use
ledger before hashing/opening truth or prediction payloads. It permits sparse
valid leads only when prediction and independent-truth keys exactly match the
pre-frozen per-station key count/hash; missing observations do not require
inventing labels or discarding otherwise valid leads. It still does not
authenticate the actual test pack, selected checkpoints, source labels, or
inference receipt, and it has no test-pack builder or inference entrypoint. It
therefore refuses any non-synthetic scope. Keep real test/buffer payloads
sealed until a complete independently audited builder -> inference -> scorer
pipeline replaces this scaffold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from hubei_forecast_metrics import (
    forecast_metrics,
    metrics_by_lead,
    paired_date_cluster_bootstrap_rmse_difference,
)
from hubei_test_ledger import claim_one_use, ledger_sha256

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)
ARMS = ("FA", "FAC", "FAP")
PAIRED_COMPARISONS = (("FA", "FAC"), ("FA", "FAP"), ("FAC", "FAP"))
LEADS = tuple(range(15, 241, 15))
# A single canonical ledger per frozen experiment. Do not derive this path from
# caller-controlled split/output paths: copying the split must not create a new
# independent test-read allowance.
CANONICAL_TEST_LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "refine-logs" / "hubei_cot_independent_20261014"
    / "audit_receipts" / "hubei_frozen_test_once.ledger.json"
)
SYNTHETIC_ONLY_ROOT = Path(__file__).resolve().parent / "synthetic_hubei_test_fixtures"
KEY_FIELDS = ("init_time_bjt", "target_time_bjt", "lead_minutes")
REQUIRED_FREEZE_FIELDS = (
    "split_sha256", "test_pack_manifest_sha256", "model_manifest_sha256",
    "test_target_dates_bjt", "test_target_dates_sha256", "evaluator_sha256",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_synthetic_fixture_paths(paths: list[Path]) -> None:
    """Reject real/external paths before reading even a release receipt."""
    root = SYNTHETIC_ONLY_ROOT.resolve()
    for path in paths:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"synthetic-only evaluator refuses path outside fixture root: {resolved}"
            ) from exc


def canonical_dates_sha256(dates: list[str]) -> str:
    payload = ("\n".join(sorted(set(dates))) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def has_consecutive_calendar_dates(dates: list[str]) -> bool:
    date_set = set(dates)
    return any(
        (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d") in date_set
        for day in date_set
    )


def canonical_keys_sha256(keys: list[tuple[str, str, int]]) -> str:
    lines = [f"{init},{target},{lead}" for init, target, lead in sorted(set(keys))]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def validate_freeze_receipt(receipt: dict[str, Any], evaluator_sha256: str,
                            split_sha256: str) -> list[str]:
    """Validate synthetic fixture evidence without opening prediction files."""
    if receipt.get("state") != "HUBEI_TEST_EVALUATION_RELEASE_PASS":
        raise ValueError("Test release receipt is absent or not PASS")
    if receipt.get("evaluation_scope") != "SYNTHETIC_ONLY":
        raise ValueError("This prototype is synthetic-only and refuses live test evaluation")
    if receipt.get("test_unseal_verdict") != "PASS":
        raise ValueError("Independent test-freeze verdict is not PASS")
    if receipt.get("test_payloads_read") != 0 or receipt.get("buffer_payloads_read") != 0:
        raise ValueError("Release preflight must certify zero test/buffer payload reads")
    missing = [key for key in REQUIRED_FREEZE_FIELDS if key not in receipt]
    if missing:
        raise ValueError(f"Freeze receipt missing required bindings: {missing}")
    if receipt["evaluator_sha256"] != evaluator_sha256:
        raise ValueError("Freeze receipt does not bind this exact evaluator source")
    if receipt["split_sha256"] != split_sha256:
        raise ValueError("Freeze receipt split SHA does not match the requested frozen split")
    for field in ("split_sha256", "test_pack_manifest_sha256", "model_manifest_sha256",
                  "test_target_dates_sha256"):
        value = receipt[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Invalid SHA-256 binding in {field}")
    dates = receipt["test_target_dates_bjt"]
    if not isinstance(dates, list) or not dates or not all(
        isinstance(d, str) and len(d) == 8 and d.isdigit() for d in dates
    ):
        raise ValueError("Freeze receipt must list YYYYMMDD test target dates")
    if len(set(dates)) != len(dates) or canonical_dates_sha256(dates) != receipt["test_target_dates_sha256"]:
        raise ValueError("Test target-date list is duplicated or its hash does not match")
    key_counts = receipt.get("expected_key_count_by_station")
    key_hashes = receipt.get("expected_keys_sha256_by_station")
    if not isinstance(key_counts, dict) or not isinstance(key_hashes, dict):
        raise ValueError("Freeze receipt must bind exact expected keys per station")
    if set(key_counts) != set(STATIONS) or set(key_hashes) != set(STATIONS):
        raise ValueError("Freeze receipt expected-key station set mismatch")
    for station in STATIONS:
        if not isinstance(key_counts[station], int) or key_counts[station] <= 0:
            raise ValueError(f"Invalid expected key count for {station}")
        value = key_hashes[station]
        if not isinstance(value, str) or len(value) != 64 or any(
            c not in "0123456789abcdef" for c in value.lower()
        ):
            raise ValueError(f"Invalid expected-key SHA-256 for {station}")
    if tuple(receipt.get("stations", ())) != STATIONS:
        raise ValueError("Freeze receipt station set/order mismatch")
    if tuple(receipt.get("seeds", ())) != SEEDS:
        raise ValueError("Freeze receipt seed set/order mismatch")
    if tuple(receipt.get("arms", ())) != ARMS:
        raise ValueError("Freeze receipt must freeze FA/FAC/FAP before test access")
    return dates


def read_frozen_test_dates(split_path: Path) -> list[str]:
    """Derive the allowed test dates from the authoritative frozen split CSV."""
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"date_BJT", "split"}.issubset(reader.fieldnames):
            raise ValueError("Frozen split must contain date_BJT and split columns")
        all_dates: set[str] = set()
        test_dates: list[str] = []
        for row in reader:
            raw = str(row.get("date_BJT", "")).strip()
            compact = raw.replace("-", "")
            try:
                day = datetime.strptime(compact, "%Y%m%d").strftime("%Y%m%d")
            except ValueError as exc:
                raise ValueError(f"Invalid frozen split BJT date: {raw!r}") from exc
            if day in all_dates:
                raise ValueError(f"Duplicate date in frozen split: {day}")
            all_dates.add(day)
            if str(row.get("split", "")).strip().lower() == "test":
                test_dates.append(day)
    if not test_dates:
        raise ValueError("Frozen split contains no test dates")
    return sorted(test_dates)


def parse_bjt_time(value: Any) -> datetime:
    s = str(value)
    for fmt in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported BJT timestamp: {s!r}")


def load_prediction(path: Path, station: str, allowed_dates: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"station", "pred_ghi_wm2", "clear_sky_ghi_wm2", *KEY_FIELDS}
        if "truth_ghi_wm2" in z.files:
            raise ValueError(f"Prediction bundle must not contain ground truth: {path}")
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        out = {key: np.asarray(z[key]) for key in required}
    n = len(out["pred_ghi_wm2"])
    if n == 0 or any(v.ndim != 1 or len(v) != n for v in out.values()):
        raise ValueError(f"Empty or inconsistent prediction bundle: {path}")
    if not np.all(out["station"] == station):
        raise ValueError(f"Station mismatch in {path}")
    for field in ("pred_ghi_wm2", "clear_sky_ghi_wm2"):
        if not np.issubdtype(out[field].dtype, np.number):
            raise ValueError(f"Nonnumeric {field} in {path}")
        if not np.isfinite(out[field]).all():
            raise ValueError(f"Nonfinite {field} in {path}")
    if (out["clear_sky_ghi_wm2"] <= 0).any():
        raise ValueError(f"Invalid daylight support in {path}")
    raw_lead = out["lead_minutes"]
    if not np.issubdtype(raw_lead.dtype, np.number) or not np.isfinite(raw_lead).all():
        raise ValueError(f"Nonnumeric or nonfinite lead in {path}")
    if not np.equal(raw_lead, np.floor(raw_lead)).all():
        raise ValueError(f"Lossy non-integer lead in {path}")
    lead = raw_lead.astype(np.int64)
    if not np.isin(lead, LEADS).all():
        raise ValueError(f"Noncanonical forecast lead in {path}")
    keys: list[tuple[str, str, int]] = []
    for init_raw, target_raw, lead_min in zip(out["init_time_bjt"], out["target_time_bjt"], lead):
        init_dt, target_dt = parse_bjt_time(init_raw), parse_bjt_time(target_raw)
        init_s, target_s = init_dt.strftime("%Y%m%d%H%M"), target_dt.strftime("%Y%m%d%H%M")
        if init_dt.minute % 15 or target_dt.minute % 15:
            raise ValueError(f"Non-quarter-hour timestamp in {path}: {init_s}->{target_s}")
        if (target_dt - init_dt).total_seconds() != int(lead_min) * 60:
            raise ValueError(f"Lead does not equal target-init in {path}: {init_s}->{target_s}")
        if target_dt.strftime("%Y%m%d") not in allowed_dates:
            raise ValueError(f"Prediction target outside frozen test dates in {path}: {target_s}")
        keys.append((init_s, target_s, int(lead_min)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate forecast key in {path}")
    out["lead_minutes"] = lead
    out["keys"] = np.asarray(keys, dtype=object)
    return out


def _load_keyed_truth(path: Path, station: str, allowed_dates: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"station", "truth_ghi_wm2", "clear_sky_ghi_wm2", *KEY_FIELDS}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        out = {key: np.asarray(z[key]) for key in required}
    n = len(out["truth_ghi_wm2"])
    if n == 0 or any(v.ndim != 1 or len(v) != n for v in out.values()):
        raise ValueError(f"Empty or inconsistent independent truth bundle: {path}")
    if not np.all(out["station"] == station):
        raise ValueError(f"Truth station mismatch in {path}")
    for field in ("truth_ghi_wm2", "clear_sky_ghi_wm2"):
        if not np.issubdtype(out[field].dtype, np.number) or not np.isfinite(out[field]).all():
            raise ValueError(f"Invalid {field} in independent truth bundle: {path}")
    if (out["truth_ghi_wm2"] < 0).any() or (out["clear_sky_ghi_wm2"] <= 0).any():
        raise ValueError(f"Invalid daylight truth support in {path}")
    raw_lead = out["lead_minutes"]
    if not np.issubdtype(raw_lead.dtype, np.number) or not np.isfinite(raw_lead).all():
        raise ValueError(f"Invalid truth lead in {path}")
    if not np.equal(raw_lead, np.floor(raw_lead)).all():
        raise ValueError(f"Lossy non-integer truth lead in {path}")
    leads = raw_lead.astype(np.int64)
    keys: list[tuple[str, str, int]] = []
    for init_raw, target_raw, lead in zip(out["init_time_bjt"], out["target_time_bjt"], leads):
        init_dt, target_dt = parse_bjt_time(init_raw), parse_bjt_time(target_raw)
        init_s, target_s = init_dt.strftime("%Y%m%d%H%M"), target_dt.strftime("%Y%m%d%H%M")
        if target_dt.strftime("%Y%m%d") not in allowed_dates:
            raise ValueError(f"Truth target outside frozen test dates in {path}: {target_s}")
        if init_dt.minute % 15 or target_dt.minute % 15 or (target_dt - init_dt).total_seconds() != int(lead) * 60:
            raise ValueError(f"Invalid truth time/lead key in {path}: {init_s}->{target_s}/{lead}")
        if int(lead) not in LEADS:
            raise ValueError(f"Noncanonical truth lead in {path}: {lead}")
        keys.append((init_s, target_s, int(lead)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate truth key in {path}")
    out["lead_minutes"] = leads
    out["keys"] = np.asarray(keys, dtype=object)
    return out


def load_truth_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Path], dict[str, str]]:
    """Verify a station-to-truth-file manifest before opening truth payloads."""
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("Independent truth manifest is absent or does not match the frozen SHA-256")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("state") != "HUBEI_INDEPENDENT_TRUTH_MANIFEST_V1":
        raise ValueError("Unexpected independent truth manifest state")
    entries = manifest.get("stations")
    if not isinstance(entries, dict) or set(entries) != set(STATIONS):
        raise ValueError("Independent truth manifest must bind exactly the three frozen stations")
    root = path.parent.resolve()
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for station in STATIONS:
        entry = entries[station]
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"Malformed truth manifest entry for {station}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"Invalid truth SHA-256 for {station}")
        unresolved = root / entry["path"]
        if unresolved.is_symlink():
            raise ValueError(f"Truth path must not be a symlink: {station}")
        candidate = unresolved.resolve()
        if candidate.parent != root:
            raise ValueError(f"Truth path must be a non-symlink direct child of its manifest directory: {station}")
        if not candidate.is_file():
            raise ValueError(f"Independent truth file is absent for {station}")
        paths[station], digests[station] = candidate, digest
    return paths, digests


def verify_truth_file_hashes(paths: dict[str, Path], digests: dict[str, str]) -> None:
    """Hash truth payloads only after the one-use ledger has been claimed."""
    for station in STATIONS:
        if sha256_file(paths[station]) != digests[station]:
            raise ValueError(f"Independent truth file hash mismatch for {station}")


def rmse_mae(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    y64, p64 = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y64.ndim != 1 or p64.ndim != 1 or y64.shape != p64.shape:
        raise ValueError("Metric inputs must be exact matching 1-D arrays")
    error = p64 - y64
    if weights is None:
        return float(np.sqrt(np.mean(error ** 2))), float(np.mean(np.abs(error)))
    w = np.asarray(weights, dtype=np.float64)
    denominator = float(w.sum())
    if denominator <= 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.sum(w * error ** 2) / denominator)), float(np.sum(w * np.abs(error)) / denominator)


def evaluate(run_root: Path, output: Path, receipt_path: Path, split_path: Path,
             truth_manifest_path: Path, bootstrap_reps: int = 2000,
             bootstrap_seed: int = 20261013) -> dict[str, Any]:
    require_synthetic_fixture_paths(
        [run_root, output, receipt_path, split_path, truth_manifest_path]
    )
    if output.exists():
        raise FileExistsError(output)
    if bootstrap_reps < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    # The canonical path is independent of caller-selected split/output paths.
    ledger_path = CANONICAL_TEST_LEDGER_PATH

    # Gate before opening any prediction bundle. The release receipt is metadata only.
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evaluator_sha = sha256_file(Path(__file__).resolve())
    split_sha256 = sha256_file(split_path)
    allowed_dates = set(validate_freeze_receipt(receipt, evaluator_sha, split_sha256))
    split_test_dates = set(read_frozen_test_dates(split_path))
    if split_test_dates != allowed_dates:
        raise ValueError("Release receipt test dates do not match dates derived from frozen split")
    truth_paths, truth_hashes = load_truth_manifest(truth_manifest_path, receipt["test_pack_manifest_sha256"])
    ledger_record = claim_one_use(ledger_path, {
        "release_receipt_sha256": sha256_file(receipt_path),
        "split_sha256": split_sha256,
        "truth_manifest_sha256": receipt["test_pack_manifest_sha256"],
        "model_manifest_sha256": receipt["model_manifest_sha256"],
        "evaluator_sha256": evaluator_sha,
        "test_target_dates_sha256": receipt["test_target_dates_sha256"],
    })
    # From here onward, even a crash or data-integrity failure leaves the
    # one-use slot consumed and cannot silently trigger a second test run.
    verify_truth_file_hashes(truth_paths, truth_hashes)

    bundles: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    input_hashes: dict[str, str] = {}
    truth_by_station = {
        station: _load_keyed_truth(truth_paths[station], station, allowed_dates)
        for station in STATIONS
    }
    for seed in SEEDS:
        for station in STATIONS:
            for arm in ARMS:
                path = run_root / f"seed{seed}" / station / f"{arm}_test_predictions_with_keys.npz"
                obj = load_prediction(path, station, allowed_dates)
                truth = truth_by_station[station]
                if not np.array_equal(obj["keys"], truth["keys"]):
                    raise ValueError(f"Prediction keys do not exactly match independent truth for {station}/{seed}/{arm}")
                if not np.array_equal(obj["clear_sky_ghi_wm2"], truth["clear_sky_ghi_wm2"]):
                    raise ValueError(f"Prediction clear-sky values differ from independent truth support for {station}/{seed}/{arm}")
                obj["truth_ghi_wm2"] = truth["truth_ghi_wm2"]
                obj["lead_minutes"] = truth["lead_minutes"]
                key = (seed, station, arm)
                bundles[key] = obj
                input_hashes[f"seed{seed}/{station}/{arm}"] = sha256_file(path)

    for station in STATIONS:
        reference = bundles[(SEEDS[0], station, ARMS[0])]
        reference_keys = [tuple(k) for k in reference["keys"].tolist()]
        if len(reference_keys) != receipt["expected_key_count_by_station"][station]:
            raise ValueError(f"Prediction count differs from frozen expected support for {station}")
        if canonical_keys_sha256(reference_keys) != receipt["expected_keys_sha256_by_station"][station]:
            raise ValueError(f"Prediction keys differ from frozen expected support for {station}")
        for seed in SEEDS:
            for arm in ARMS:
                current = bundles[(seed, station, arm)]
                if not np.array_equal(reference["keys"], current["keys"]):
                    raise ValueError(f"Test keys differ across arms/seeds for {station}/{seed}/{arm}")
                for field in ("truth_ghi_wm2", "clear_sky_ghi_wm2"):
                    if not np.array_equal(reference[field], current[field]):
                        raise ValueError(f"Test {field} differs across arms/seeds for {station}/{seed}/{arm}")

    per_seed: dict[str, Any] = {}
    per_group: list[dict[str, Any]] = []
    bootstrap_records: dict[tuple[int, str, str], dict[str, Any]] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for arm in ARMS:
            scores = []
            for station in STATIONS:
                obj = bundles[(seed, station, arm)]
                metrics = forecast_metrics(
                    obj["truth_ghi_wm2"], obj["pred_ghi_wm2"],
                    clear_sky_wm2=obj["clear_sky_ghi_wm2"],
                    clear_sky_threshold_wm2=100.0,
                )
                scores.append(metrics["rmse_wm2"])
                lead_metrics = metrics_by_lead(
                    obj["truth_ghi_wm2"], obj["pred_ghi_wm2"], obj["lead_minutes"],
                    clear_sky_wm2=obj["clear_sky_ghi_wm2"],
                    clear_sky_threshold_wm2=100.0,
                )
                per_group.append({"seed": seed, "station": station, "arm": arm,
                                  **metrics,
                                  "per_lead": {str(lead): values for lead, values in lead_metrics.items()}})
                bootstrap_records[(seed, station, arm)] = {
                    "keys": [tuple(k) for k in obj["keys"].tolist()],
                    "truth_wm2": obj["truth_ghi_wm2"],
                    "prediction_wm2": obj["pred_ghi_wm2"],
                }
            per_seed[str(seed)][arm] = float(np.mean(scores))
    primary = {arm: float(np.mean([per_seed[str(seed)][arm] for seed in SEEDS])) for arm in ARMS}

    dates = sorted({str(k[1])[:8] for obj in bundles.values() for k in obj["keys"]})
    if len(dates) < 2 or not set(dates).issubset(allowed_dates):
        raise ValueError("Observed test target dates do not match frozen allowlist")
    bootstraps: dict[str, dict[str, Any]] = {}
    two_day_bootstraps: dict[str, dict[str, Any]] = {}
    point_deltas: dict[str, float] = {}
    for comparison_index, (control, treatment) in enumerate(PAIRED_COMPARISONS):
        comparison = f"{treatment}_minus_{control}"
        result = paired_date_cluster_bootstrap_rmse_difference(
            bootstrap_records,
            control_arm=control, treatment_arm=treatment,
            seeds=SEEDS, stations=STATIONS,
            resamples=bootstrap_reps, random_seed=bootstrap_seed + comparison_index,
        )
        delta = primary[treatment] - primary[control]
        if not np.isclose(delta, result["estimate_difference_wm2"], rtol=0.0, atol=1e-10):
            raise ValueError(f"{comparison} differs from paired-bootstrap point estimate")
        bootstraps[comparison] = result
        two_day_bootstraps[comparison] = (
            paired_date_cluster_bootstrap_rmse_difference(
                bootstrap_records,
                control_arm=control, treatment_arm=treatment,
                seeds=SEEDS, stations=STATIONS,
                resamples=bootstrap_reps, random_seed=bootstrap_seed + 100 + comparison_index,
                block_length_days=2,
            ) if has_consecutive_calendar_dates(dates)
            else {"status": "unavailable_no_consecutive_2day_test_dates"}
        )
        point_deltas[comparison] = delta

    station_key_sets = [
        {tuple(k) for k in bundles[(SEEDS[0], station, ARMS[0])]["keys"].tolist()}
        for station in STATIONS
    ]
    common_keys = sorted(set.intersection(*station_key_sets))
    common_support: dict[str, Any] = {
        "n_keys_per_station": {station: len(common_keys) for station in STATIONS},
        "target_dates_bjt": sorted({key[1][:8] for key in common_keys}),
    }
    if common_keys:
        common_records: dict[tuple[int, str, str], dict[str, Any]] = {}
        common_per_seed: dict[str, dict[str, float]] = {}
        common_groups: list[dict[str, Any]] = []
        for seed in SEEDS:
            common_per_seed[str(seed)] = {}
            for arm in ARMS:
                station_rmse = []
                for station in STATIONS:
                    obj = bundles[(seed, station, arm)]
                    row_by_key = {tuple(k): i for i, k in enumerate(obj["keys"].tolist())}
                    indices = np.asarray([row_by_key[key] for key in common_keys], dtype=np.int64)
                    truth = obj["truth_ghi_wm2"][indices]
                    prediction = obj["pred_ghi_wm2"][indices]
                    clear_sky = obj["clear_sky_ghi_wm2"][indices]
                    lead = obj["lead_minutes"][indices]
                    metrics = forecast_metrics(
                        truth, prediction, clear_sky_wm2=clear_sky,
                        clear_sky_threshold_wm2=100.0,
                    )
                    station_rmse.append(metrics["rmse_wm2"])
                    lead_metrics = metrics_by_lead(
                        truth, prediction, lead, clear_sky_wm2=clear_sky,
                        clear_sky_threshold_wm2=100.0,
                    )
                    common_groups.append({"seed": seed, "station": station, "arm": arm,
                                          **metrics,
                                          "per_lead": {str(k): v for k, v in lead_metrics.items()}})
                    common_records[(seed, station, arm)] = {
                        "keys": common_keys,
                        "truth_wm2": truth,
                        "prediction_wm2": prediction,
                    }
                common_per_seed[str(seed)][arm] = float(np.mean(station_rmse))
        common_primary = {
            arm: float(np.mean([common_per_seed[str(seed)][arm] for seed in SEEDS]))
            for arm in ARMS
        }
        common_support["primary_rmse_wm2"] = common_primary
        common_support["paired_rmse_differences_wm2"] = {
            f"{treatment}_minus_{control}": common_primary[treatment] - common_primary[control]
            for control, treatment in PAIRED_COMPARISONS
        }
        common_support["per_seed_station_arm_metrics"] = common_groups
        if len(common_support["target_dates_bjt"]) >= 2:
            common_bootstraps = {}
            common_two_day_bootstraps = {}
            for comparison_index, (control, treatment) in enumerate(PAIRED_COMPARISONS):
                comparison = f"{treatment}_minus_{control}"
                result = paired_date_cluster_bootstrap_rmse_difference(
                    common_records, control_arm=control, treatment_arm=treatment,
                    seeds=SEEDS, stations=STATIONS,
                    resamples=bootstrap_reps, random_seed=bootstrap_seed + comparison_index,
                )
                common_bootstraps[comparison] = {
                    "estimate_wm2": result["estimate_difference_wm2"],
                    "ci95_wm2": result["ci95_wm2"],
                    "resamples": result["resamples"],
                    "invalid_resamples_skipped": result["invalid_resamples_skipped"],
                }
                if has_consecutive_calendar_dates(common_support["target_dates_bjt"]):
                    two_day = paired_date_cluster_bootstrap_rmse_difference(
                        common_records, control_arm=control, treatment_arm=treatment,
                        seeds=SEEDS, stations=STATIONS,
                        resamples=bootstrap_reps,
                        random_seed=bootstrap_seed + 100 + comparison_index,
                        block_length_days=2,
                    )
                    if "estimate_difference_wm2" in two_day:
                        common_two_day_bootstraps[comparison] = {
                            "estimate_wm2": two_day["estimate_difference_wm2"],
                            "ci95_wm2": two_day["ci95_wm2"],
                            "resamples": two_day["resamples"],
                            "random_seed": two_day["random_seed"],
                            "block_length_days": two_day["block_length_days"],
                            "eligible_block_count": two_day["eligible_block_count"],
                            "invalid_resamples_skipped": two_day["invalid_resamples_skipped"],
                        }
                    else:
                        common_two_day_bootstraps[comparison] = two_day
                else:
                    common_two_day_bootstraps[comparison] = {
                        "status": "unavailable_no_consecutive_2day_common_dates"
                    }
            common_support["paired_date_cluster_bootstrap_by_comparison"] = common_bootstraps
            common_support["paired_two_day_block_bootstrap_by_comparison"] = common_two_day_bootstraps
        else:
            common_support["paired_date_cluster_bootstrap_by_comparison"] = None
            common_support["paired_two_day_block_bootstrap_by_comparison"] = None
            common_support["bootstrap_status"] = "unavailable_fewer_than_two_common_target_dates"
    else:
        common_support["primary_rmse_wm2"] = None
        common_support["per_seed_station_arm_metrics"] = []
        common_support["paired_rmse_differences_wm2"] = None
        common_support["paired_date_cluster_bootstrap_by_comparison"] = None
        common_support["paired_two_day_block_bootstrap_by_comparison"] = None
        common_support["bootstrap_status"] = "unavailable_no_three_station_common_keys"

    report = {
        "state": "SYNTHETIC_CONTRACT_TEST_PASSED",
        "evaluation_type": "synthetic_proxy",
        "split_sha256": split_sha256,
        "release_receipt_sha256": sha256_file(receipt_path),
        "one_use_ledger_sha256": ledger_sha256(ledger_path),
        "one_use_claim_id": ledger_record["claim_id"],
        "evaluator_sha256": evaluator_sha,
        "test_pack_manifest_sha256": receipt["test_pack_manifest_sha256"],
        "model_manifest_sha256": receipt["model_manifest_sha256"],
        "test_target_dates_bjt": dates,
        "stations": list(STATIONS), "seeds": list(SEEDS), "arms": list(ARMS),
        "primary_endpoint": "mean station RMSE within seed, then equal mean across seeds",
        "primary_rmse_wm2": primary,
        "paired_rmse_differences_wm2": point_deltas,
        "paired_date_cluster_bootstrap_by_comparison": {
            comparison: {
            "resamples": result["resamples"], "seed": result["random_seed"],
            "shared_target_date_weights_across_stations_arms_and_seeds": True,
            "invalid_resamples_skipped": result["invalid_resamples_skipped"],
            "estimate_wm2": result["estimate_difference_wm2"],
            "ci95_wm2": result["ci95_wm2"],
            } for comparison, result in bootstraps.items()
        },
        "paired_two_day_block_bootstrap_by_comparison": {
            comparison: ({
                "resamples": result["resamples"],
                "random_seed": result["random_seed"],
                "block_length_days": result["block_length_days"],
                "eligible_block_count": result["eligible_block_count"],
                "invalid_resamples_skipped": result["invalid_resamples_skipped"],
                "estimate_wm2": result["estimate_difference_wm2"],
                "ci95_wm2": result["ci95_wm2"],
            } if "estimate_difference_wm2" in result else result)
            for comparison, result in two_day_bootstraps.items()
        },
        "clear_sky_sensitivity_threshold_wm2": 100.0,
        "normalized_metric_denominator": "mean observed GHI on the evaluated support",
        "three_station_common_support_sensitivity": common_support,
        "per_seed_station_arm_metrics": per_group,
        "prediction_bundle_sha256": input_hashes,
        "independent_truth_sha256": truth_hashes,
        "test_payloads_read": len(input_hashes) + len(truth_hashes),
        "buffer_payloads_read": 0,
        "interpretation": "Report the frozen test result as observed; do not use it for model selection or further tuning.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp_output.write_text(payload, encoding="utf-8")
    tmp_output.replace(output)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release-receipt", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True,
                   help="Authoritative frozen day_assignment.csv; test dates are derived and SHA-checked")
    p.add_argument("--truth-manifest", type=Path, required=True,
                   help="Independently sourced station truth manifest; synthetic-only until release audit passes")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261013)
    a = p.parse_args()
    report = evaluate(a.run_root, a.output, a.release_receipt, a.split, a.truth_manifest,
                      a.bootstrap_reps, a.bootstrap_seed)
    print(json.dumps({"state": report["state"], "primary_rmse_wm2": report["primary_rmse_wm2"],
                      "paired_rmse_differences_wm2": report["paired_rmse_differences_wm2"],
                      "test_payloads_read": report["test_payloads_read"],
                      "buffer_payloads_read": report["buffer_payloads_read"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
