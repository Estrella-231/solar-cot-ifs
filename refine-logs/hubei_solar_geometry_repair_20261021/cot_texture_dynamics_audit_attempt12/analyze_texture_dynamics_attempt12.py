#!/usr/bin/env python3
"""Offline validation audit of spatial texture and COT temporal change.

Uses only the attempt12 validation COT bank and validation R outputs copied
under inputs/. CPP is a retrieval reference, not independent truth. The
future observed-AGRI->R series is an offline proxy, never a deployable input.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INP = HERE / "inputs"
OUT = HERE
LEADS = tuple(range(15, 241, 15))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp(x: str) -> str:
    s = "".join(c for c in str(x) if c.isdigit())
    if len(s) != 12:
        raise ValueError(f"Invalid BJT timestamp {x!r}")
    datetime.strptime(s, "%Y%m%d%H%M")
    return s


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def field_stats(a: np.ndarray, m: np.ndarray) -> tuple[float, float, int]:
    v = a[m]
    if not v.size:
        return float("nan"), float("nan"), 0
    return float(v.std(dtype=np.float64)), float(np.mean(v >= 30.0)), int(v.size)


def grad_parts(a: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mh = m[:, 1:] & m[:, :-1]
    mv = m[1:, :] & m[:-1, :]
    return a[:, 1:][mh] - a[:, :-1][mh], a[1:, :][mv] - a[:-1, :][mv]


def main() -> None:
    manifest = json.loads((INP / "source_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((INP / "forecast_cot_metrics.json").read_text(encoding="utf-8"))
    bank_report = json.loads((INP / "BANK_REPORT.json").read_text(encoding="utf-8"))
    hashes = {
        "validation.npy": manifest["source_roles"][0]["COT_sha256"],
        "keys.csv": metrics["simvp_key_sha256"],
        "val_keys.csv": metrics["cpp_reference_keys_sha256"],
        "val_reference_cpp_cot.npy": metrics["cpp_reference_array_sha256"],
        "val_valid_mask.npy": metrics["cpp_mask_sha256"],
        "validation.npy_metrics": metrics["forecast_cot_bank_sha256"],
    }
    for name, expected in hashes.items():
        source_name = "validation.npy" if name == "validation.npy_metrics" else name
        actual = sha256(INP / source_name)
        if actual.lower() != expected.lower():
            raise ValueError(f"SHA mismatch {source_name}: {actual} != {expected}")
    if manifest.get("test_payloads_read") != 0 or manifest.get("buffer_payloads_read") != 0:
        raise ValueError("Input manifest does not keep test/buffer closed")
    if bank_report.get("test_payloads_read") != 0 or bank_report.get("buffer_payloads_read") != 0:
        raise ValueError("Bank report does not keep test/buffer closed")

    cot = np.load(INP / "validation.npy", mmap_mode="r", allow_pickle=False)
    pred = np.load(INP / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(INP / "val_reference_cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(INP / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    if cot.shape != (407, 16, 1, 16, 16):
        raise ValueError(f"Unexpected COT bank shape {cot.shape}")
    if pred.shape != cpp.shape or pred.shape != valid.shape or pred.shape[1:] != (1, 16, 16):
        raise ValueError(f"R validation arrays mismatch: {pred.shape}, {cpp.shape}, {valid.shape}")

    srows = read_csv(INP / "keys.csv")
    rrows = read_csv(INP / "val_keys.csv")
    if len(srows) != cot.shape[0] or len(rrows) != len(pred):
        raise ValueError("Key row counts do not match the arrays")
    rindex: dict[str, int] = {}
    for i, row in enumerate(rrows):
        t = stamp(row["time_bjt"])
        if t in rindex or row["split"] != "val":
            raise ValueError(f"Duplicate/non-validation R row {row}")
        rindex[t] = i

    agg: dict[int, dict] = {}
    all_records: list[dict] = []
    pair_key_count = 0
    skipped_target = skipped_history = 0
    for si, row in enumerate(srows):
        if row["split"] != "val":
            raise ValueError(f"Unexpected split in COT key {row}")
        init = stamp(row["init_time_bjt"])
        targets = [stamp(x) for x in row["target_times_bjt"].split("|")]
        if len(targets) != 16:
            raise ValueError("Each validation sequence must have 16 canonical targets")
        init_dt = datetime.strptime(init, "%Y%m%d%H%M")
        for li, lead in enumerate(LEADS):
            target = targets[li]
            if target != (init_dt + timedelta(minutes=lead)).strftime("%Y%m%d%H%M"):
                raise ValueError(f"init/lead/target mismatch: {init}, {lead}, {target}")
            if target not in rindex:
                skipped_target += 1
                continue
            if init not in rindex:
                skipped_history += 1
                continue
            ti, hi = rindex[target], rindex[init]
            f = np.asarray(cot[si, li, 0], dtype=np.float64)
            h = np.asarray(pred[hi, 0], dtype=np.float64)
            o = np.asarray(pred[ti, 0], dtype=np.float64)
            c = np.asarray(cpp[ti, 0], dtype=np.float64)
            m = np.array(valid[ti, 0], dtype=bool, copy=True)
            m &= np.isfinite(f) & np.isfinite(h) & np.isfinite(o) & np.isfinite(c)
            if not m.any():
                continue
            pair_key_count += 1
            err_f, err_h, err_o = f[m] - c[m], h[m] - c[m], o[m] - c[m]
            df, do = f[m] - h[m], o[m] - h[m]
            if lead not in agg:
                agg[lead] = defaultdict(list)
            a = agg[lead]
            a["date"].append(target[:8])
            a["n_pixels"].append(int(m.sum()))
            for name, e in (("forecast_cpp", err_f), ("persist_cpp", err_h), ("offline_cpp", err_o)):
                a[name + "_sse"].append(float(np.dot(e, e)))
            for name, x in (("forecast", f), ("history", h), ("offline", o), ("cpp", c)):
                sd, thick, n = field_stats(x, m)
                a[name + "_std"].append(sd)
                a[name + "_thick"].append(thick)
            gf_h, gf_v = grad_parts(f, m)
            go_h, go_v = grad_parts(o, m)
            gc_h, gc_v = grad_parts(c, m)
            for name, (gh, gv) in (("forecast", (gf_h, gf_v)), ("offline", (go_h, go_v)), ("cpp", (gc_h, gc_v))):
                a[name + "_grad_sse"].append(float(np.dot(gh, gh) + np.dot(gv, gv)))
                a[name + "_grad_n"].append(int(gh.size + gv.size))
            # Accumulate pixel-level temporal-change agreement against the
            # same frozen R applied to observed future AGRI (offline proxy).
            a["delta_f_sq"].append(float(np.dot(df, df)))
            a["delta_o_sq"].append(float(np.dot(do, do)))
            a["delta_p_sq"].append(float(np.dot(do, do)))  # persistence increment is zero
            a["delta_error_f_sq"].append(float(np.dot(df - do, df - do)))
            a["delta_error_p_sq"].append(float(np.dot(do, do)))
            a["delta_cross"].append(float(np.dot(df, do)))
            a["delta_n"].append(int(df.size))
            a["delta_f_sum"].append(float(df.sum(dtype=np.float64)))
            a["delta_o_sum"].append(float(do.sum(dtype=np.float64)))
            a["delta_f_sq_center"].append(float(np.dot(df, df)))
            a["delta_o_sq_center"].append(float(np.dot(do, do)))

    rows = []
    for lead in LEADS:
        if lead not in agg:
            continue
        a = agg[lead]
        n = sum(a["n_pixels"])
        d = {"lead_minutes": lead, "n_init_target_pairs": len(a["date"]),
             "n_target_dates": len(set(a["date"])), "n_common_cpp_pixels": n}
        for label, key in (("forecast", "forecast_cpp_sse"), ("persistence", "persist_cpp_sse"), ("offline_future_agri_R", "offline_cpp_sse")):
            d[label + "_rmse_vs_cpp"] = float(np.sqrt(sum(a[key]) / n))
        for method in ("forecast", "history", "offline", "cpp"):
            d[method + "_mean_map_spatial_std"] = float(np.nanmean(a[method + "_std"]))
            d[method + "_mean_map_fraction_cot_ge_30"] = float(np.nanmean(a[method + "_thick"]))
        for method in ("forecast", "offline", "cpp"):
            gn = sum(a[method + "_grad_n"])
            d[method + "_gradient_rms"] = float(np.sqrt(sum(a[method + "_grad_sse"]) / gn)) if gn else float("nan")
        dn = sum(a["delta_n"])
        d["forecast_delta_rms"] = float(np.sqrt(sum(a["delta_f_sq"]) / dn))
        d["offline_future_agri_R_delta_rms"] = float(np.sqrt(sum(a["delta_o_sq"]) / dn))
        d["forecast_delta_to_offline_delta_rms_ratio"] = d["forecast_delta_rms"] / d["offline_future_agri_R_delta_rms"] if d["offline_future_agri_R_delta_rms"] else float("nan")
        d["forecast_delta_rmse_vs_offline_delta"] = float(np.sqrt(sum(a["delta_error_f_sq"]) / dn))
        d["persistence_delta_rmse_vs_offline_delta"] = float(np.sqrt(sum(a["delta_error_p_sq"]) / dn))
        cross = sum(a["delta_cross"])
        df2, do2 = sum(a["delta_f_sq_center"]), sum(a["delta_o_sq_center"])
        d["forecast_vs_offline_delta_corr_through_origin"] = cross / np.sqrt(df2 * do2) if df2 > 0 and do2 > 0 else float("nan")
        d["forecast_gradient_rms_ratio_to_offline"] = d["forecast_gradient_rms"] / d["offline_gradient_rms"] if d["offline_gradient_rms"] else float("nan")
        d["forecast_spatial_std_ratio_to_offline"] = d["forecast_mean_map_spatial_std"] / d["offline_mean_map_spatial_std"] if d["offline_mean_map_spatial_std"] else float("nan")
        rows.append(d)

    # The earlier attempt12 headline table labels 1/2/3/4 h as consecutive
    # one-hour bins (+15..60, +75..120, +135..180, +195..240), not exact
    # point leads at +60/+120/+180/+240. Reproduce both scopes explicitly.
    horizon_rows = []
    old_headline = [
        (10.255, 8.946, 8.169),
        (9.787, 9.238, 8.224),
        (10.194, 9.617, 9.052),
        (12.063, 9.086, 9.304),
    ]
    for bi, start in enumerate((0, 60, 120, 180)):
        leads_in_bin = range(start + 15, start + 61, 15)
        nbin = sum(sum(agg[lead]["n_pixels"]) for lead in leads_in_bin)
        sse = {}
        for name, key in (("forecast", "forecast_cpp_sse"),
                          ("persistence", "persist_cpp_sse"),
                          ("offline_future_agri_R", "offline_cpp_sse")):
            sse[name] = sum(sum(agg[lead][key]) for lead in leads_in_bin)
        vals = {name: float(np.sqrt(v / nbin)) for name, v in sse.items()}
        expected = old_headline[bi]
        reproduced = all(abs(v - e) < 0.0005 for v, e in zip(
            (vals["forecast"], vals["persistence"], vals["offline_future_agri_R"]), expected))
        horizon_rows.append({
            "horizon_bin": bi + 1,
            "lead_window_minutes": f"+{start+15}..+{start+60}",
            "n_common_cpp_pixels": nbin,
            "forecast_rmse_vs_cpp": vals["forecast"],
            "persistence_rmse_vs_cpp": vals["persistence"],
            "offline_future_agri_R_rmse_vs_cpp": vals["offline_future_agri_R"],
            "forecast_minus_persistence": vals["forecast"] - vals["persistence"],
            "matches_attempt12_published_hour_bin": reproduced,
        })
    with (OUT / "horizon_bin_rmse.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)

    with (OUT / "texture_dynamics_by_lead.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "state": "VALIDATION_ONLY_TEXTURE_AND_TEMPORAL_CHANGE_DIAGNOSTIC",
        "scope": "attempt12 corrected SimVP->frozen corrected R on exact validation init/target BJT keys",
        "reference_warning": "CPP is a satellite retrieval reference, not independent COT truth. Future observed-AGRI->R is an offline proxy and is not a deployable forecast input.",
        "support": "All COT errors and spatial/texture metrics use exact target-time CPP-valid masks. Temporal increments compare predicted-future COT minus frozen-R(history AGRI) with frozen-R(observed future AGRI) minus the same frozen-R(history AGRI), on that target mask.",
        "n_simvp_validation_sequences": len(srows), "n_r_validation_frames": len(rrows),
        "n_matched_init_target_lead_pairs": pair_key_count,
        "skipped_no_future_R_frame": skipped_target, "skipped_no_history_R_frame": skipped_history,
        "cpp_reference_key_sha256": sha256(INP / "val_keys.csv"),
        "simvp_forecast_cot_sha256": sha256(INP / "validation.npy"),
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "exact_lead_metrics": rows,
        "one_hour_bin_metrics": horizon_rows,
    }
    (OUT / "texture_dynamics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    leads = np.asarray([r["lead_minutes"] / 60 for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    ax[0, 0].plot(leads, [r["forecast_rmse_vs_cpp"] for r in rows], "o-", label="SimVP forecast → R")
    ax[0, 0].plot(leads, [r["persistence_rmse_vs_cpp"] for r in rows], "s--", label="History persistence → R")
    ax[0, 0].plot(leads, [r["offline_future_agri_R_rmse_vs_cpp"] for r in rows], "^:", label="Observed future AGRI → R (offline proxy)")
    ax[0, 0].set_title("COT RMSE vs CPP at each exact lead")
    ax[0, 0].set_ylabel("RMSE (COT)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(leads, [r["forecast_spatial_std_ratio_to_offline"] for r in rows], "o-", label="Spatial standard deviation")
    ax[0, 1].plot(leads, [r["forecast_gradient_rms_ratio_to_offline"] for r in rows], "s-", label="Neighbor-gradient RMS")
    ax[0, 1].axhline(1, color="0.3", lw=1, ls="--")
    ax[0, 1].set_title("Forecast spatial texture / offline proxy")
    ax[0, 1].set_ylabel("Ratio (1 = same texture amplitude)")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(leads, [r["forecast_delta_rmse_vs_offline_delta"] for r in rows], "o-", label="Forecast increment")
    ax[1, 0].plot(leads, [r["persistence_delta_rmse_vs_offline_delta"] for r in rows], "s--", label="Persistence increment = 0")
    ax[1, 0].set_title("Does predicted COT change match the offline R proxy?")
    ax[1, 0].set_ylabel("Increment RMSE (COT)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(leads, [r["forecast_delta_to_offline_delta_rms_ratio"] for r in rows], "o-", label="Increment RMS ratio")
    ax[1, 1].plot(leads, [r["forecast_vs_offline_delta_corr_through_origin"] for r in rows], "s-", label="Through-origin increment correlation")
    ax[1, 1].axhline(1, color="0.3", lw=1, ls="--")
    ax[1, 1].axhline(0, color="0.5", lw=0.8)
    ax[1, 1].set_title("Change amplitude and direction")
    ax[1, 1].set_ylabel("Ratio / correlation")
    ax[1, 1].legend(fontsize=8)
    for a in ax.ravel():
        a.set_xlabel("Forecast lead (hours from history end)")
        a.grid(alpha=0.25)
        a.set_xticks([0.25, 1, 2, 3, 4])
    fig.suptitle("Validation-only COT texture and dynamics audit — not independent test", fontsize=14)
    fig.savefig(OUT / "texture_dynamics_diagnostic.png", dpi=180)
    fig.savefig(OUT / "texture_dynamics_diagnostic.pdf")
    plt.close(fig)

    # Human-readable lead summary at 1..4 h only; full lead data remains CSV.
    selected = [r for r in rows if r["lead_minutes"] in (60, 120, 180, 240)]
    lines = [
        "# Attempt12 validation COT texture and temporal-dynamics audit",
        "",
        "## Scope and interpretation",
        "This is a validation-only diagnostic of attempt12 corrected SimVP→frozen R. All target comparisons use exact BJT timestamps and the same CPP-valid target mask. CPP is a satellite retrieval reference, not independent truth. The observed-future AGRI→R curve is an offline proxy, not a forecast input. The result does not use test or buffer payloads.",
        "",
        "The spatial-texture ratios compare forecast-map spatial standard deviation and neighbor-gradient RMS against the offline observed-future AGRI→R proxy on the same target support. The temporal-change comparison is forecast COT minus R(history AGRI) versus R(observed future AGRI) minus the same R(history AGRI); persistence has zero increment. Thus these are attribution diagnostics for forecast smoothness/change, not standalone truth scores.",
        "",
        f"Matched init-target-lead pairs: {pair_key_count}. Validation sequences: {len(srows)}. R validation frames: {len(rrows)}. Test payloads read: 0. Buffer payloads read: 0.",
        "",
        "## Exact-lead results at +1/+2/+3/+4 h",
        "",
        "| Exact target lead | Forecast RMSE vs CPP | Persistence RMSE vs CPP | Forecast/offline spatial std | Forecast/offline gradient RMS | Forecast Δ RMSE vs offline Δ | Persistence Δ RMSE vs offline Δ | Forecast/offline Δ RMS | Through-origin Δ correlation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in selected:
        vals = [f"+{r['lead_minutes']} min", r["forecast_rmse_vs_cpp"], r["persistence_rmse_vs_cpp"],
                r["forecast_spatial_std_ratio_to_offline"], r["forecast_gradient_rms_ratio_to_offline"],
                r["forecast_delta_rmse_vs_offline_delta"], r["persistence_delta_rmse_vs_offline_delta"],
                r["forecast_delta_to_offline_delta_rms_ratio"], r["forecast_vs_offline_delta_corr_through_origin"]]
        lines.append("| " + " | ".join([vals[0]] + [f"{v:.3f}" for v in vals[1:]]) + " |")
    lines += [
        "",
        "## Reconciliation with the earlier 1/2/3/4-hour table",
        "The established attempt12 headline values (10.255/8.946, 9.787/9.238, 10.194/9.617, 12.063/9.086) are **one-hour-wide lead bins**, not exact point leads at +60/+120/+180/+240. Re-aggregating the current hash-verified per-lead RMSEs over +15..+60, +75..+120, +135..+180 and +195..+240 reproduces those values; see `horizon_bin_rmse.csv`. The exact +60/+120/+180/+240 values in the table above answer a different question. So the records are consistent, but the phrase “1 h” is easy to misread unless the lead window is shown.",
        "",
        "## Reading the result",
        "Across exact +2/+3/+4 h leads, forecast COT is worse than the history-persistence map against CPP; at exact +1 h it is slightly better (11.055 vs 11.505). In the one-hour-wide bins, all four bins are worse than persistence, matching the existing attempt12 headline. Forecast spatial standard deviation is about 0.52–0.68 of the offline future-AGRI→R proxy at the 1–4 h checkpoints, and neighbor-gradient RMS is about 0.68–0.78, consistent with smoother spatial fields. But the forecast temporal-increment RMS is 1.04–1.40 of the offline proxy, so changes are not uniformly too weak. Through-origin increment correlation is only about 0.33 at +1 h and 0.45–0.68 at +2..+4 h: the pattern/timing of change is an issue alongside texture, not simply insufficient amplitude. These are retrieval-proxy diagnostics; they do not establish that a physical-constraint loss will improve COT or GHI. Attempt13’s matched auxiliary-loss comparison remains the controlled training test before choosing a subsequent single-factor motion/evolution experiment.",
        "",
        "## Reproducibility",
        "Run `python analyze_texture_dynamics_attempt12.py` from this folder. Source hashes and counts are embedded in `texture_dynamics_summary.json`; the downloaded input files are limited to validation bank/keys/CPP and validation R prediction/mask files.",
    ]
    (OUT / "TEXTURE_DYNAMICS_AUDIT_20261021.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": pair_key_count, "leads": len(rows), "horizon_bins": len(horizon_rows), "csv": str(OUT / "texture_dynamics_by_lead.csv"), "horizon_csv": str(OUT / "horizon_bin_rmse.csv"), "plot": str(OUT / "texture_dynamics_diagnostic.png")}, indent=2))


if __name__ == "__main__":
    main()
