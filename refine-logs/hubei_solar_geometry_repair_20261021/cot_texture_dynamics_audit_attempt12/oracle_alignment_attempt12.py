#!/usr/bin/env python3
"""Validation-only COT error attribution by oracle spatial registration.

For each matched attempt12 validation (init, target, lead), compare the
forecast COT map and history-persistence map with the frozen R applied to the
observed future AGRI. The future map is an offline proxy; CPP supplies only a
common valid-pixel mask. Per-case translations/scales are optimized against
the future proxy and are deliberately non-deployable upper-bound diagnostics.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
INP = HERE / "inputs"
LEADS = tuple(range(15, 241, 15))
RADIUS = 2
BOOTSTRAPS = 4000
SEED = 20261021


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 12:
        raise ValueError(f"Invalid BJT key: {value!r}")
    datetime.strptime(digits, "%Y%m%d%H%M")
    return digits


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rmse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def summarize(records: list[dict], date_draws: np.ndarray | None = None) -> dict:
    if date_draws is None:
        selected = records
        mult = None
    else:
        date_names = sorted({r["date"] for r in records})
        counts = defaultdict(int)
        for d in date_draws:
            counts[date_names[int(d)]] += 1
        selected = [r for r in records if r["date"] in counts]
        mult = counts
    names = ("forecast", "persistence", "oracle_shift", "oracle_scale", "oracle_shift_scale")
    sums = {name: 0.0 for name in names}
    n_total = 0
    for rec in selected:
        m = 1 if mult is None else mult[rec["date"]]
        n_total += m * rec["n"]
        for name in names:
            sums[name] += m * rec[f"{name}_sse"]
    return {name: rmse(sums[name], n_total) for name in names} | {"n_pixels": n_total}


def main() -> None:
    manifest = json.loads((INP / "source_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((INP / "forecast_cot_metrics.json").read_text(encoding="utf-8"))
    bank = json.loads((INP / "BANK_REPORT.json").read_text(encoding="utf-8"))
    expected = {
        "validation.npy": manifest["source_roles"][0]["COT_sha256"],
        "keys.csv": metrics["simvp_key_sha256"],
        "val_keys.csv": metrics["cpp_reference_keys_sha256"],
        "val_reference_cpp_cot.npy": metrics["cpp_reference_array_sha256"],
        "val_valid_mask.npy": metrics["cpp_mask_sha256"],
    }
    for name, digest in expected.items():
        if not digest:
            raise ValueError(f"Missing source digest for {name}")
        actual = sha256(INP / name)
        if actual.lower() != str(digest).lower():
            raise ValueError(f"Input hash mismatch for {name}: {actual} != {digest}")
    if manifest.get("test_payloads_read") != 0 or manifest.get("buffer_payloads_read") != 0:
        raise ValueError("Input source manifest does not keep test/buffer closed")
    if bank.get("test_payloads_read") != 0 or bank.get("buffer_payloads_read") != 0:
        raise ValueError("Validation bank does not keep test/buffer closed")
    if metrics.get("forecast_cot_bank_sha256", "").lower() != sha256(INP / "validation.npy").lower():
        raise ValueError("Forecast COT bank hash does not match its metric receipt")

    cot = np.load(INP / "validation.npy", mmap_mode="r", allow_pickle=False)
    rpred = np.load(INP / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(INP / "val_reference_cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(INP / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    if cot.shape != (407, 16, 1, 16, 16):
        raise ValueError(f"Unexpected forecast COT bank shape {cot.shape}")
    if rpred.shape != cpp.shape or rpred.shape != valid.shape or rpred.shape[1:] != (1, 16, 16):
        raise ValueError("R prediction, CPP reference, and mask shapes differ")
    if not (np.isfinite(cot).all() and np.isfinite(rpred).all() and np.isfinite(cpp).all()):
        raise ValueError("Non-finite values found in a source array; do not silently mask them")

    rrows = read_rows(INP / "val_keys.csv")
    srows = read_rows(INP / "keys.csv")
    rindex: dict[str, int] = {}
    for i, row in enumerate(rrows):
        key = stamp(row["time_bjt"])
        if key in rindex or row.get("split") != "val":
            raise ValueError(f"Duplicate/non-val R key: {row}")
        rindex[key] = i
    if len(srows) != cot.shape[0] or len(rrows) != rpred.shape[0]:
        raise ValueError("Key table length does not match arrays")

    # Keep the exact same CPP-valid target pixels for every candidate shift.
    # Eroding by the shift radius avoids wraparound and support changes.
    lo, hi = RADIUS, 16 - RADIUS
    center = np.s_[lo:hi, lo:hi]
    records_by_lead: dict[int, list[dict]] = defaultdict(list)
    full_scores: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: {"sse": 0.0, "n": 0})
    official_method_names = {
        "forecast": "forecast_simvp_to_R",
        "persistence": "history_COT_persistence",
        "offline_future": "offline_observed_future_AGRI_to_R",
    }
    omitted = {"target_missing": 0, "init_missing": 0, "no_valid_pixels": 0}
    for si, row in enumerate(srows):
        if row.get("split") != "val":
            raise ValueError("Unexpected non-validation row in forecast bank")
        init = stamp(row["init_time_bjt"])
        targets = [stamp(x) for x in row["target_times_bjt"].split("|")]
        if len(targets) != 16:
            raise ValueError("Expected 16 canonical target keys")
        init_dt = datetime.strptime(init, "%Y%m%d%H%M")
        for li, lead in enumerate(LEADS):
            target = targets[li]
            if target != (init_dt + timedelta(minutes=lead)).strftime("%Y%m%d%H%M"):
                raise ValueError(f"init/lead/target mismatch: {init}, {lead}, {target}")
            if target not in rindex:
                omitted["target_missing"] += 1
                continue
            if init not in rindex:
                omitted["init_missing"] += 1
                continue
            ti, hi_idx = rindex[target], rindex[init]
            f = np.asarray(cot[si, li, 0], dtype=np.float64)
            h = np.asarray(rpred[hi_idx, 0], dtype=np.float64)
            y = np.asarray(rpred[ti, 0], dtype=np.float64)
            cpp_map = np.asarray(cpp[ti, 0], dtype=np.float64)
            full_mask = np.asarray(valid[ti, 0], dtype=bool).copy()
            full_mask &= np.isfinite(f) & np.isfinite(h) & np.isfinite(y) & np.isfinite(cpp_map)
            full_arrays = {"forecast": f, "persistence": h, "offline_future": y}
            for method, array in full_arrays.items():
                error = array[full_mask] - cpp_map[full_mask]
                acc = full_scores[(lead, method)]
                acc["sse"] += float(np.dot(error, error))
                acc["n"] += int(error.size)

            m = full_mask[center].copy()
            f0 = f[center]
            h0 = h[center]
            y0 = y[center]
            # All candidate source cells must be finite on the same fixed mask.
            for dy in range(-RADIUS, RADIUS + 1):
                for dx in range(-RADIUS, RADIUS + 1):
                    m &= np.isfinite(f[lo + dy:hi + dy, lo + dx:hi + dx])
            m &= np.isfinite(f0) & np.isfinite(h0) & np.isfinite(y0)
            n = int(m.sum())
            if not n:
                omitted["no_valid_pixels"] += 1
                continue

            # Compute per-case oracle upper bounds without changing the mask.
            base_sse = float(np.square(f0[m] - y0[m]).sum(dtype=np.float64))
            persist_sse = float(np.square(h0[m] - y0[m]).sum(dtype=np.float64))
            best_shift_sse = float("inf")
            best_scale_sse = float("inf")
            best_joint_sse = float("inf")
            best_shift = (0, 0)
            best_joint_scale = float("nan")
            zero_shift_scale = float("nan")
            for dy in range(-RADIUS, RADIUS + 1):
                for dx in range(-RADIUS, RADIUS + 1):
                    src = f[lo + dy:hi + dy, lo + dx:hi + dx][m]
                    tgt = y0[m]
                    err = src - tgt
                    sse = float(np.dot(err, err))
                    if sse < best_shift_sse:
                        best_shift_sse, best_shift = sse, (dy, dx)
                    denom = float(np.dot(src, src))
                    scale = max(0.0, float(np.dot(src, tgt)) / denom) if denom > 0 else 0.0
                    scaled_sse = float(np.square(scale * src - tgt).sum(dtype=np.float64))
                    if scaled_sse < best_joint_sse:
                        best_joint_sse = scaled_sse
                        best_joint_scale = scale
                    if dy == 0 and dx == 0:
                        best_scale_sse = scaled_sse
                        zero_shift_scale = scale
            records_by_lead[lead].append({
                "date": target[:8], "init": init, "target": target, "lead": lead,
                "n": n, "forecast_sse": base_sse, "persistence_sse": persist_sse,
                "oracle_shift_sse": best_shift_sse, "oracle_scale_sse": best_scale_sse,
                "oracle_shift_scale_sse": best_joint_sse,
                "best_dy": best_shift[0], "best_dx": best_shift[1],
                "zero_shift_scale": zero_shift_scale,
                "best_joint_scale": best_joint_scale,
                "forecast_mean": float(f0[m].mean()), "future_mean": float(y0[m].mean()),
            })

    # Independently reproduce the archived full-patch scores before using the
    # staged observed-future R array for the registration upper-bound analysis.
    official_path = HERE / "official_recalc" / "forecast_cot_metrics.csv"
    official_rows = read_rows(official_path)
    for lead in LEADS:
        for method, official_name in official_method_names.items():
            candidates = [
                row for row in official_rows
                if int(row["lead_minutes"]) == lead
                and row["scope"] == "all_patch"
                and row["station"] == "ALL_PATCH"
                and row["method"] == official_name
            ]
            if len(candidates) != 1:
                raise ValueError(f"Expected one archived full-patch row for {lead}/{official_name}, got {len(candidates)}")
            official = candidates[0]
            actual = full_scores[(lead, method)]
            actual_rmse = rmse(actual["sse"], int(actual["n"]))
            if int(official["cpp_common_support_pixels"]) != int(actual["n"]):
                raise ValueError(f"Common-support count mismatch at lead {lead}/{method}: {actual['n']} vs {official['cpp_common_support_pixels']}")
            if abs(float(official["rmse_cot"]) - actual_rmse) > 2e-6:
                raise ValueError(f"Archived score reproduction mismatch at lead {lead}/{method}: {actual_rmse} vs {official['rmse_cot']}")

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for lead in LEADS:
        recs = records_by_lead.get(lead, [])
        if not recs:
            continue
        point = summarize(recs)
        dates = sorted({r["date"] for r in recs})
        if len(dates) < 2:
            raise ValueError(f"Too few date clusters at lead {lead}")
        draws = rng.integers(0, len(dates), size=(BOOTSTRAPS, len(dates)))
        delta = {name: [] for name in ("persistence", "oracle_shift", "oracle_scale", "oracle_shift_scale")}
        for draw in draws:
            s = summarize(recs, draw)
            for name in delta:
                delta[name].append(s[name] - s["forecast"])
        row_out = {
            "lead_min": lead, "n_init_target_pairs": len(recs), "n_target_dates": len(dates),
            "n_common_cpp_pixels": point["n_pixels"],
            "forecast_RMSE_vs_future_R": point["forecast"],
            "persistence_RMSE_vs_future_R": point["persistence"],
            "oracle_shift_RMSE_vs_future_R": point["oracle_shift"],
            "oracle_scale_RMSE_vs_future_R": point["oracle_scale"],
            "oracle_shift_scale_RMSE_vs_future_R": point["oracle_shift_scale"],
            "mean_oracle_dy": float(np.mean([r["best_dy"] for r in recs])),
            "mean_oracle_dx": float(np.mean([r["best_dx"] for r in recs])),
            "fraction_nonzero_oracle_shift": float(np.mean([r["best_dy"] != 0 or r["best_dx"] != 0 for r in recs])),
            "median_zero_shift_oracle_scale": float(np.median([r["zero_shift_scale"] for r in recs])),
            "median_joint_oracle_scale": float(np.median([r["best_joint_scale"] for r in recs])),
        }
        for name, values in delta.items():
            row_out[f"{name}_minus_forecast_CI95_low"] = float(np.quantile(values, 0.025))
            row_out[f"{name}_minus_forecast_CI95_high"] = float(np.quantile(values, 0.975))
        rows.append(row_out)

    out_csv = HERE / "oracle_alignment_by_lead_attempt12.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_records = [r for recs in records_by_lead.values() for r in recs]
    pair_csv = HERE / "oracle_alignment_pairs_attempt12.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(all_records[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(all_records, key=lambda r: (r["target"], r["init"], r["lead"])))
    overall = summarize(all_records)
    date_names = sorted({r["date"] for r in all_records})
    draws = rng.integers(0, len(date_names), size=(BOOTSTRAPS, len(date_names)))
    overall_delta = {}
    for name in ("persistence", "oracle_shift", "oracle_scale", "oracle_shift_scale"):
        vals = [summarize(all_records, draw)[name] - summarize(all_records, draw)["forecast"] for draw in draws]
        overall_delta[name] = {
            "rmse_difference_vs_forecast": overall[name] - overall["forecast"],
            "date_bootstrap_95ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        }
    result = {
        "state": "VALIDATION_ONLY_ORACLE_ALIGNMENT_DIAGNOSTIC_NOT_DEPLOYABLE",
        "region_model": "Hubei attempt12 SimVP-to-frozen-R COT bank",
        "target": "frozen R applied to observed future AGRI (offline proxy); CPP only supplies common target mask",
        "interpretation": "per-case spatial shift and scale are fitted against future proxy and are unattainable at forecast issue time; they diagnose error type, not forecast skill",
        "fixed_support": f"CPP-valid pixels in central {(16-2*RADIUS)}x{(16-2*RADIUS)}; support eroded by shift radius {RADIUS} pixels so all shifts use identical target pixels without wraparound",
        "max_integer_shift_pixels_each_axis": RADIUS,
        "matched_pair_count": len(all_records),
        "target_date_count": len(date_names),
        "target_dates_bjt": date_names,
        "unique_initializations": len({r["init"] for r in all_records}),
        "omitted_pair_counts": omitted,
        "overall_common_pixel_RMSE": overall,
        "overall_paired_target_date_bootstrap": overall_delta,
        "n_bootstrap": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "input_sha256": {name: sha256(INP / name) for name in [*expected, "val_prediction_cot.npy"]},
        "prediction_array_provenance_note": "val_prediction_cot.npy has no digest field in the original source manifest; its local SHA is recorded here, and all 16 full-patch forecast/persistence/offline-future RMSEs and common-support counts were independently reproduced against the archived official_recalc CSV before oracle analysis.",
        "reproduced_archived_full_patch_rows": len(LEADS) * len(official_method_names),
        "pair_level_output_csv": str(pair_csv),
        "pair_level_output_sha256": sha256(pair_csv),
        "output_csv": str(out_csv),
    }
    out_json = HERE / "oracle_alignment_attempt12.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md = HERE / "ORACLE_ALIGNMENT_DIAGNOSTIC_ATTEMPT12.md"
    lines = [
        "# Attempt12 COT spatial-alignment upper-bound diagnostic",
        "",
        "This is a validation-only attribution diagnostic, not an operational score. The reference is the same frozen R applied to observed future AGRI; that future image is unavailable at initialization. CPP only defines the common valid-pixel mask and is not independent COT truth.",
        "",
        f"Matched init-target-lead pairs: {len(all_records)}; target BJT dates: {len(date_names)}; test/buffer reads: 0/0.",
        "Before this diagnostic, an independent full-patch re-score reproduced all 48 archived forecast/persistence/offline-future RMSE and support-count rows (16 exact leads × 3 methods). The R-output array lacks a source-manifest digest; its local SHA and this score-reproduction check are recorded in the JSON receipt.",
        f"All methods use the same CPP-valid central {(16-2*RADIUS)}x{(16-2*RADIUS)} pixels; shifts are bounded by ±{RADIUS} pixels per axis and no wraparound is used.",
        "The per-case shift/scale sees the future R proxy, so it is an unattainable oracle. It only indicates whether position or magnitude correction could plausibly recover error.",
        "",
        "## Pooled validation-proxy results",
        "",
        "| Method | RMSE vs future R proxy | Paired date-block 95% CI of RMSE difference vs forecast |",
        "|---|---:|---:|",
        f"| Forecast COT | {overall['forecast']:.4f} | reference |",
    ]
    for name, label in (("persistence", "History persistence"), ("oracle_shift", "Oracle spatial shift"), ("oracle_scale", "Oracle amplitude scale"), ("oracle_shift_scale", "Oracle shift + scale")):
        ci = overall_delta[name]["date_bootstrap_95ci"]
        lines.append(f"| {label} | {overall[name]:.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += [
        "",
        "## By exact lead",
        "",
        "See `oracle_alignment_by_lead_attempt12.csv` for exact +15…+240 min rows and date-cluster intervals; `oracle_alignment_pairs_attempt12.csv` contains each paired case, its selected oracle shift/scale, and per-case errors.",
        "",
        "## Claim boundary",
        "",
        "A lower oracle-shift error would show that spatial registration has recoverable headroom; a lower oracle-scale error would show amplitude-calibration headroom. Neither proves a deployable method. The future AGRI→R target and CPP mask are retrieval-chain references, not independent physical COT truth. A new learned transport/source-sink model still requires train-only development, untouched validation selection rules, and a fresh paired evaluation against forecast and persistence.",
        "",
        "Reproduce with `python oracle_alignment_attempt12.py`. Input SHA-256 values, exact target dates, masks, bootstrap settings, and the pair-level output hash are saved in `oracle_alignment_attempt12.json`.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "pairs": len(all_records), "dates": date_names, "overall": overall, "delta": overall_delta, "by_lead": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
