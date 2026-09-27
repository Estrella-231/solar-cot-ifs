"""Diagnose how the learned residual changes Sili station COT errors.

Input is the compact validation-only station-track NPZ from the pilot audit.
CPP is treated as a retrieval reference. The bootstrap resamples initialization
days to reduce inflation from overlapping windows within a day.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COT_BINS = [(0, 5, "0–<5"), (5, 10, "5–<10"), (10, 20, "10–<20"),
            (20, 30, "20–<30"), (30, 50, "30–<50"), (50, np.inf, "≥50")]
LEAD_BINS = [(15, 60, "+15–60"), (75, 120, "+75–120"),
             (135, 180, "+135–180"), (195, 240, "+195–240")]


def metrics(ref, valid, pred):
    e = pred[valid] - ref[valid]
    thick = valid & (ref >= 30)
    clear = valid & (ref < 5)
    et = pred[thick] - ref[thick]
    return {
        "n": int(valid.sum()),
        "bias": float(e.mean()) if e.size else None,
        "mae": float(np.abs(e).mean()) if e.size else None,
        "rmse": float(np.sqrt(np.mean(e * e))) if e.size else None,
        "thick_n": int(thick.sum()),
        "thick_bias": float(et.mean()) if et.size else None,
        "thick_mae": float(np.abs(et).mean()) if et.size else None,
        "thick_rmse": float(np.sqrt(np.mean(et * et))) if et.size else None,
        "thick_miss_lt10": int(np.sum(thick & (pred < 10))),
        "clear_n": int(clear.sum()),
        "clear_false_ge10_rate": float(np.mean(pred[clear] >= 10)) if clear.any() else None,
    }


def day_bootstrap(ref, valid, learned, transport, day_rows, reps=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique = np.unique(day_rows)
    aggregates = {}
    for day in unique:
        rows = day_rows == day
        v = valid[rows]
        r = ref[rows]
        l = learned[rows]
        t = transport[rows]
        thick = v & (r >= 30)
        clear = v & (r < 5)
        aggregates[str(day)] = {
            "all_n": int(v.sum()),
            "all_sq_l": float(np.square(l[v] - r[v]).sum()),
            "all_sq_t": float(np.square(t[v] - r[v]).sum()),
            "all_abs_l": float(np.abs(l[v] - r[v]).sum()),
            "all_abs_t": float(np.abs(t[v] - r[v]).sum()),
            "thick_n": int(thick.sum()),
            "thick_sq_l": float(np.square(l[thick] - r[thick]).sum()),
            "thick_sq_t": float(np.square(t[thick] - r[thick]).sum()),
            "thick_miss_l": int(np.sum(thick & (l < 10))),
            "thick_miss_t": int(np.sum(thick & (t < 10))),
            "clear_n": int(clear.sum()),
            "clear_false_l": int(np.sum(clear & (l >= 10))),
            "clear_false_t": int(np.sum(clear & (t >= 10))),
        }
    day_ids = list(aggregates)
    draws = np.empty((reps, 4), dtype=np.float64)
    for b in range(reps):
        chosen = rng.choice(day_ids, size=len(day_ids), replace=True)
        sums = {k: sum(aggregates[d][k] for d in chosen) for k in next(iter(aggregates.values()))}
        draws[b, 0] = np.sqrt(sums["all_sq_l"] / sums["all_n"]) - np.sqrt(sums["all_sq_t"] / sums["all_n"])
        draws[b, 1] = np.sqrt(sums["thick_sq_l"] / sums["thick_n"]) - np.sqrt(sums["thick_sq_t"] / sums["thick_n"])
        draws[b, 2] = sums["thick_miss_l"] / sums["thick_n"] - sums["thick_miss_t"] / sums["thick_n"]
        draws[b, 3] = sums["clear_false_l"] / sums["clear_n"] - sums["clear_false_t"] / sums["clear_n"]
    names = ["all_rmse_learned_minus_transport", "thick_rmse_learned_minus_transport",
             "thick_miss_rate_learned_minus_transport", "clear_false_rate_learned_minus_transport"]
    return {"grouping": "initialization BJT day; overlapping windows within a day stay together",
            "days": len(unique), "replicates": reps, "seed": seed,
            "intervals_95pct": {name: [float(x) for x in np.quantile(draws[:, i], [0.025, 0.975])]
                                for i, name in enumerate(names)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    a = np.load(args.tracks)
    ref = np.asarray(a["cpp"], dtype=float)
    valid = np.asarray(a["cpp_valid"], dtype=bool)
    learned = np.asarray(a["learned"], dtype=float)
    transport = np.asarray(a["transport"], dtype=float)
    leads = np.asarray(a["lead_minutes"], dtype=int)
    days = np.asarray(a["init_day_BJT"]).astype(str)
    if ref.shape != valid.shape or learned.shape != ref.shape or transport.shape != ref.shape:
        raise ValueError("track arrays have incompatible shapes")
    if not (ref.ndim == 2 and ref.shape[0] == 512 and ref.shape[1] == 16):
        raise ValueError(f"expected the fixed 512x16 validation tracks, got {ref.shape}")
    summary = json.loads(args.summary.read_text())
    if summary.get("source", {}).get("test_used") is not False or summary.get("source", {}).get("split") != "validation":
        raise RuntimeError("Only the closed validation audit may be analyzed here")
    if summary.get("source", {}).get("station_patch_pixel") != [32, 32]:
        raise RuntimeError("Station center contract mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    overall = {"transport": metrics(ref, valid, transport),
               "learned": metrics(ref, valid, learned)}
    overall["paired_change"] = {
        "mean_residual_adjustment_learned_minus_transport": float((learned[valid] - transport[valid]).mean()),
        "mean_abs_error_change": float((np.abs(learned[valid]-ref[valid]) - np.abs(transport[valid]-ref[valid])).mean()),
        "fraction_lower_absolute_error": float(np.mean(np.abs(learned[valid]-ref[valid]) < np.abs(transport[valid]-ref[valid]))),
    }
    cot_rows = []
    cot_effect = np.full((len(COT_BINS), len(LEAD_BINS)), np.nan)
    cot_counts = np.zeros_like(cot_effect, dtype=int)
    methods_by_bin = {"transport": [], "learned": []}
    for bi, (lo, hi, label) in enumerate(COT_BINS):
        m = valid & (ref >= lo) & (ref < hi)
        row = {"cot_bin": label, "n": int(m.sum()),
               "mean_cpp_reference": float(ref[m].mean()) if m.any() else None,
               "mean_residual_adjustment_learned_minus_transport": float((learned[m]-transport[m]).mean()) if m.any() else None,
               "fraction_adjustment_negative": float(np.mean((learned[m]-transport[m]) < 0)) if m.any() else None,
               "transport": metrics(ref, m, transport),
               "learned": metrics(ref, m, learned)}
        cot_rows.append(row)
        methods_by_bin["transport"].append(row["transport"]["rmse"])
        methods_by_bin["learned"].append(row["learned"]["rmse"])
        for li, (l0, l1, llabel) in enumerate(LEAD_BINS):
            lm = (leads >= l0) & (leads <= l1)
            q = m & lm[None, :]
            cot_counts[bi, li] = int(q.sum())
            if q.any():
                cot_effect[bi, li] = float((learned[q]-transport[q]).mean())

    lead_cot = []
    for li, (l0, l1, label) in enumerate(LEAD_BINS):
        lead_mask = (leads >= l0) & (leads <= l1)
        row = {"lead_bin_min": label, "n": int((valid & lead_mask[None, :]).sum()), "cot_bins": []}
        for bi, (lo, hi, cot_label) in enumerate(COT_BINS):
            row["cot_bins"].append({"cot_bin": cot_label, "n": int(cot_counts[bi, li]),
                                    "mean_residual_adjustment_learned_minus_transport":
                                        float(cot_effect[bi, li]) if np.isfinite(cot_effect[bi, li]) else None})
        lead_cot.append(row)

    result = {
        "state": "COMPLETE_VALIDATION_RESIDUAL_BIAS_DIAGNOSTIC",
        "test_used": False,
        "reference": "CPP COT retrieval; not independent in-situ cloud truth",
        "support": "same site-pixel CPP mask and C13 transport in-domain mask as the station audit",
        "overlapping_sequences": True,
        "overall": overall,
        "by_cpp_magnitude": cot_rows,
        "residual_adjustment_by_cpp_and_lead": lead_cot,
        "day_block_bootstrap": day_bootstrap(ref, valid, learned, transport, days),
        "decision_note": "descriptive follow-up; CPP-conditioned bins do not constitute deploy-time routing",
    }
    (output / "residual_bias.json").write_text(json.dumps(result, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im = axes[0].imshow(cot_effect, aspect="auto", cmap="RdBu_r", vmin=-12, vmax=12)
    axes[0].set_xticks(range(len(LEAD_BINS)), [x[2] for x in LEAD_BINS])
    axes[0].set_yticks(range(len(COT_BINS)), [x[2] for x in COT_BINS])
    axes[0].set_xlabel("Forecast lead range (min)")
    axes[0].set_ylabel("CPP station COT bin")
    axes[0].set_title("Mean residual change: learned − transport")
    for i in range(len(COT_BINS)):
        for j in range(len(LEAD_BINS)):
            val = cot_effect[i, j]
            label = f"{val:.1f}\nn={cot_counts[i,j]}" if np.isfinite(val) else "n=0"
            axes[0].text(j, i, label, ha="center", va="center", fontsize=8,
                         color="white" if np.isfinite(val) and abs(val) > 7 else "black")
    fig.colorbar(im, ax=axes[0], label="COT adjustment")
    x = np.arange(len(COT_BINS))
    width = 0.36
    axes[1].bar(x-width/2, methods_by_bin["transport"], width, label="C13 transport")
    axes[1].bar(x+width/2, methods_by_bin["learned"], width, label="Transport + residual")
    axes[1].set_xticks(x, [z[2] for z in COT_BINS])
    axes[1].set_xlabel("CPP station COT bin")
    axes[1].set_ylabel("Station COT RMSE within reference bin")
    axes[1].set_title("Error by reference cloud thickness")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Sili residual bias audit — fixed validation tracks; CPP retrieval reference")
    fig.tight_layout()
    fig.savefig(output / "residual_bias_by_cot_and_lead.png", dpi=180)
    fig.savefig(output / "residual_bias_by_cot_and_lead.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(output), "overall": overall,
                      "day_block_bootstrap": result["day_block_bootstrap"],
                      "by_cpp_magnitude": [{k: r[k] for k in ("cot_bin", "n", "mean_residual_adjustment_learned_minus_transport", "fraction_adjustment_negative")}
                                           for r in cot_rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
