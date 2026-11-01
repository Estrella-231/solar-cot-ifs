#!/usr/bin/env python3
"""Independently rescore matched GHI-head predictions and paired date bootstrap.

Reads only the frozen train/validation pack rows and saved validation
predictions/checkpoints. It does not open test payloads or use training logs as
metric inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ARMS = ("A", "AC", "AP")
STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)
EXPECTED_PACK_SHA = "27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4"
EXPECTED_ROWS_SHA = "a3cac8b82fa0a9639ef1235da9b0a2b4010829982e075f5de8d714b17ba98756"
EXPECTED_SOURCE_HASHES = {
    42: {"training": "f633d1c290ebb6081a4dd0913d9cede973991d6f476c2a59187989ab8edab4b4",
         "head": "eab8a098541b3eab47b1972e0413a82b82746039b768e3479fa1c4f75732730b"},
    43: {"training": "ece9abfb04f01a1e89bf613ba76be0c0c32d753841ec60de12f942ca52c86f58",
         "head": "eab8a098541b3eab47b1972e0413a82b82746039b768e3479fa1c4f75732730b"},
    44: {"training": "ece9abfb04f01a1e89bf613ba76be0c0c32d753841ec60de12f942ca52c86f58",
         "head": "eab8a098541b3eab47b1972e0413a82b82746039b768e3479fa1c4f75732730b"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def rmse64(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float:
    err = np.asarray(p, np.float64) - np.asarray(y, np.float64)
    if weights is None:
        return float(np.sqrt(np.mean(err * err, dtype=np.float64)))
    w = np.asarray(weights, np.float64)
    den = float(w.sum())
    if den <= 0:
        raise ValueError("bootstrap draw has zero support")
    return float(np.sqrt(np.sum(w * err * err, dtype=np.float64) / den))


def parse_bjt_date(utc_text: str) -> str:
    utc = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=__import__("datetime").timezone.utc)
    return (utc.astimezone(__import__("datetime").timezone.utc) + timedelta(hours=8)).date().isoformat()


def season_for_bjt_date(date_text: str) -> str:
    month = int(date_text[5:7])
    return "DJF" if month in (12, 1, 2) else "MAM" if month in (3, 4, 5) else "JJA" if month in (6, 7, 8) else "SON"


def weighted_rmse_pair(y: np.ndarray, pred1: np.ndarray, pred0: np.ndarray,
                       dates: np.ndarray, multiplicity: Counter[str] | None = None) -> float:
    if multiplicity is None:
        w = None
    else:
        w = np.asarray([multiplicity.get(d, 0) for d in dates], dtype=np.float64)
    return rmse64(y, pred1, w) - rmse64(y, pred0, w)


def date_resample_draws(dates: list[str], n_boot: int, rng: np.random.Generator,
                        two_day: bool) -> list[Counter[str]]:
    unique = sorted(set(dates))
    if not unique:
        raise ValueError("no validation date clusters")
    if not two_day:
        return [Counter(rng.choice(unique, size=len(unique), replace=True).tolist())
                for _ in range(n_boot)]
    day = [datetime.fromisoformat(d).date() for d in unique]
    pairs = [(unique[i], unique[i + 1]) for i in range(len(unique) - 1)
             if (day[i + 1] - day[i]).days == 1]
    if len(pairs) < 2:
        raise ValueError(f"too few adjacent BJT date pairs for 2-day bootstrap: {len(pairs)}")
    n_blocks = int(np.ceil(len(unique) / 2))
    draws = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(pairs), size=n_blocks)
        counts: Counter[str] = Counter()
        for i in chosen:
            counts.update(pairs[int(i)])
        draws.append(counts)
    return draws


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True, help="local copy of frozen GHI pack")
    p.add_argument("--results", type=Path, required=True, help="directory containing seed42/43/44 result dirs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261013)
    a = p.parse_args()

    pack_meta_path = a.pack / "pack_manifest.json"
    rows_path = a.pack / "rows.csv"
    if sha256(rows_path) != EXPECTED_ROWS_SHA:
        raise ValueError("downloaded rows.csv does not match frozen pack hash")
    pack_meta = json.loads(pack_meta_path.read_text(encoding="utf-8"))
    if pack_meta.get("state") != "COMPLETE" or pack_meta.get("test_payload_opened") is not False:
        raise ValueError("pack completion/test gate failed")
    if sha256(pack_meta_path) != EXPECTED_PACK_SHA:
        raise ValueError("pack manifest hash differs from training contract")
    rows = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8")))
    if len(rows) != int(pack_meta["rows"]):
        raise ValueError("row count does not match pack manifest")
    keys = [(r["station"], r["target_time_utc"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate station/time key in frozen pack")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("non-train/validation row found")

    data: dict[tuple[int, str], dict[str, object]] = {}
    per_seed: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    reference_val_keys = None
    reference_truth: dict[str, np.ndarray] = {}
    reference_dates: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        seed_dir = a.results / f"ghi_heads_new_r_seed{seed}_v1"
        data[f"seed{seed}"] = {}
        per_seed[str(seed)] = {}
        for station in STATIONS:
            station_rows = [r for r in rows if r["split"] == "val" and r["station"] == station]
            station_keys = [(r["station"], r["target_time_utc"]) for r in station_rows]
            y = np.asarray([float(r["ghi"]) for r in station_rows], dtype=np.float64)
            dates = np.asarray([parse_bjt_date(r["target_time_utc"]) for r in station_rows])
            if not np.isfinite(y).all():
                raise ValueError(f"nonfinite truth: seed={seed}, station={station}")
            if reference_val_keys is None:
                reference_val_keys = {}
            if station not in reference_val_keys:
                reference_val_keys[station] = station_keys
                reference_truth[station] = y
                reference_dates[station] = dates
            elif station_keys != reference_val_keys[station] or not np.array_equal(y, reference_truth[station]):
                raise ValueError(f"validation keys/truth differ across seeds for {station}")

            run_meta = json.loads((seed_dir / station / "run_metadata.json").read_text(encoding="utf-8"))
            if run_meta.get("seed") != seed or run_meta.get("station") != station:
                raise ValueError(f"run metadata identity mismatch: {seed}/{station}")
            if run_meta.get("pack_manifest_sha256") != EXPECTED_PACK_SHA:
                raise ValueError(f"run metadata pack hash mismatch: {seed}/{station}")
            if run_meta.get("split_id") != pack_meta.get("split_id") or run_meta.get("test_payload_opened") is not False:
                raise ValueError(f"run metadata split/test gate mismatch: {seed}/{station}")
            if run_meta.get("epochs") != 30 or run_meta.get("batch_size") != 2000:
                raise ValueError(f"training budget mismatch: {seed}/{station}")
            expected_hash = EXPECTED_SOURCE_HASHES[seed]
            for field, expected in (("training_source_sha256", expected_hash["training"]),
                                    ("head_source_sha256", expected_hash["head"])):
                recorded = run_meta.get(field)
                if recorded is not None and recorded != expected:
                    raise ValueError(f"source hash mismatch: {seed}/{station}/{field}")
            if run_meta.get("parameter_counts") != {"A": 22177, "AC": 24489, "AP": 24613}:
                raise ValueError(f"parameter-count contract mismatch: {seed}/{station}")
            sources[f"{seed}/{station}"] = {
                "training_source_sha256": run_meta.get("training_source_sha256") or expected_hash["training"],
                "head_source_sha256": run_meta.get("head_source_sha256") or expected_hash["head"],
                "pack_manifest_sha256": run_meta.get("pack_manifest_sha256"),
                "checkpoint_sha256": {},
                "prediction_sha256": {},
            }
            data[f"seed{seed}"][station] = {"truth": y, "dates": dates, "pred": {}}
            per_seed[str(seed)][station] = {}
            for arm in ARMS:
                d = seed_dir / station
                cp = d / f"{arm}_best.pt"
                pred_path = d / f"{arm}_val_prediction_ghi_wm2.npy"
                if not cp.is_file() or not pred_path.is_file():
                    raise FileNotFoundError(f"missing completed artifact: {cp} or {pred_path}")
                pred = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.float64).reshape(-1)
                if len(pred) != len(y) or not np.isfinite(pred).all():
                    raise ValueError(f"prediction shape/finite mismatch: {seed}/{station}/{arm}")
                val_meta = json.loads((d / f"{arm}_validation.json").read_text(encoding="utf-8"))
                if val_meta.get("n") != len(y) or val_meta.get("seed") != seed or val_meta.get("station") != station:
                    raise ValueError(f"validation metadata mismatch: {seed}/{station}/{arm}")
                if val_meta.get("test_payload_opened") is not False:
                    raise ValueError(f"test gate failed: {seed}/{station}/{arm}")
                actual_cp_sha, actual_pred_sha = sha256(cp), sha256(pred_path)
                if val_meta.get("checkpoint_sha256") != actual_cp_sha:
                    raise ValueError(f"checkpoint SHA mismatch: {seed}/{station}/{arm}")
                if "prediction_sha256" in val_meta and val_meta["prediction_sha256"] != actual_pred_sha:
                    raise ValueError(f"prediction SHA mismatch: {seed}/{station}/{arm}")
                sources[f"{seed}/{station}"]["checkpoint_sha256"][arm] = actual_cp_sha
                sources[f"{seed}/{station}"]["prediction_sha256"][arm] = actual_pred_sha
                data[f"seed{seed}"][station]["pred"][arm] = pred
                value = {
                    "n": int(len(y)),
                    "rmse_wm2": rmse64(y, pred),
                    "mae_wm2": float(np.mean(np.abs(pred - y), dtype=np.float64)),
                    "bias_wm2": float(np.mean(pred - y, dtype=np.float64)),
                    "checkpoint_sha256": actual_cp_sha,
                    "prediction_sha256": actual_pred_sha,
                }
                per_seed[str(seed)][station][arm] = value

    # Seed-level and three-seed matched estimands.
    seed_summary = {}
    for seed in SEEDS:
        stations = per_seed[str(seed)]
        arm_mean = {arm: float(np.mean([stations[s][arm]["rmse_wm2"] for s in STATIONS])) for arm in ARMS}
        seed_summary[str(seed)] = {
            "station_equal_rmse_wm2": arm_mean,
            "AC_minus_A_wm2": arm_mean["AC"] - arm_mean["A"],
            "AC_minus_AP_wm2": arm_mean["AC"] - arm_mean["AP"],
            "AP_minus_A_wm2": arm_mean["AP"] - arm_mean["A"],
            "per_station_deltas": {
                s: {
                    "AC_minus_A_wm2": stations[s]["AC"]["rmse_wm2"] - stations[s]["A"]["rmse_wm2"],
                    "AC_minus_AP_wm2": stations[s]["AC"]["rmse_wm2"] - stations[s]["AP"]["rmse_wm2"],
                } for s in STATIONS
            },
        }
    mean_deltas = {
        name: float(np.mean([seed_summary[str(s)][name] for s in SEEDS]))
        for name in ("AC_minus_A_wm2", "AC_minus_AP_wm2", "AP_minus_A_wm2")
    }

    season_summary = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        season_summary[season] = {}
        for seed in SEEDS:
            season_summary[season][str(seed)] = {}
            for station in STATIONS:
                d = data[f"seed{seed}"][station]
                mask = np.asarray([season_for_bjt_date(x) == season for x in d["dates"]])
                if not mask.any():
                    continue
                y = d["truth"][mask]
                scores = {arm: rmse64(y, d["pred"][arm][mask]) for arm in ARMS}
                season_summary[season][str(seed)][station] = {
                    "n": int(mask.sum()),
                    "rmse_wm2": scores,
                    "AC_minus_A_wm2": scores["AC"] - scores["A"],
                    "AC_minus_AP_wm2": scores["AC"] - scores["AP"],
                }

    all_dates = sorted(set().union(*(set(reference_dates[s].tolist()) for s in STATIONS)))
    rng = np.random.default_rng(a.seed)
    bootstrap = {}
    for two_day, name in ((False, "target_bjt_date"), (True, "consecutive_2day_block")):
        draws = date_resample_draws(all_dates, a.bootstrap, rng, two_day)
        values = {"AC_minus_A_wm2": [], "AC_minus_AP_wm2": [], "AP_minus_A_wm2": []}
        for counts in draws:
            estimates = {k: [] for k in values}
            for seed in SEEDS:
                for comp, a1, a0 in (("AC_minus_A_wm2", "AC", "A"),
                                     ("AC_minus_AP_wm2", "AC", "AP"),
                                     ("AP_minus_A_wm2", "AP", "A")):
                    station_deltas = []
                    for station in STATIONS:
                        d = data[f"seed{seed}"][station]
                        delta = weighted_rmse_pair(d["truth"], d["pred"][a1], d["pred"][a0], d["dates"], counts)
                        station_deltas.append(delta)
                    estimates[comp].append(float(np.mean(station_deltas)))
            for comp in values:
                values[comp].append(float(np.mean(estimates[comp])))
        bootstrap[name] = {
            comp: {
                "percentile_95_ci_wm2": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "bootstrap_mean_wm2": float(np.mean(v)),
            } for comp, v in values.items()
        }

    summary = {
        "state": "PASS_THREE_SEED_FP64_COMMON_VALIDATION_AUDIT",
        "split_id": pack_meta["split_id"],
        "pack_manifest_sha256": sha256(pack_meta_path),
        "rows_sha256": sha256(rows_path),
        "rows_total": len(rows),
        "validation_rows": {s: len(reference_truth[s]) for s in STATIONS},
        "validation_unique_bjt_dates": len(all_dates),
        "bjt_dates": all_dates,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "per_station_metrics": per_seed,
        "per_seed": seed_summary,
        "descriptive_season_strata": season_summary,
        "three_seed_mean_delta_wm2": mean_deltas,
        "three_seed_arm_rmse_mean_wm2": {
            arm: float(np.mean([seed_summary[str(s)]["station_equal_rmse_wm2"][arm] for s in SEEDS]))
            for arm in ARMS
        },
        "bootstrap": {
            "n_resamples": a.bootstrap,
            "rng_seed": a.seed,
            "interval": "percentile 95%; resample BJT target-date clusters; preserve all seed/station/method pairing; recompute station RMSE differences then average stations and seeds",
            "results": bootstrap,
        },
        "source_artifacts": sources,
        "test_payload_opened": False,
        "claim_scope": "three-seed validation result conditional on validation-based checkpoint selection; not independent-test evidence",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "state": summary["state"],
        "per_seed": seed_summary,
        "three_seed_mean_delta_wm2": mean_deltas,
        "bootstrap": bootstrap,
        "test_payload_opened": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
