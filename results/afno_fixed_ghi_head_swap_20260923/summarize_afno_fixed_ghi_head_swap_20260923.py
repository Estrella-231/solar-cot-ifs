import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / "complete.json").read_text())
assert result["state"] == "COMPLETE_PAIRED_FROZEN_GHI_HEAD_AFNO_COT_SWAP"
assert result["test_used"] is False

rows = []
for station in ("Sili", "Zhujia"):
    for group in ("AC_F", "AC_ST"):
        arm = result["arms"][station][group]
        low, high = arm["paired_day_bootstrap"]["delta_AFNO_minus_SimVP_95pct_Wm2"]
        rows.append({
            "station": station,
            "head": group,
            "n_rows": arm["rows"],
            "days": arm["days"],
            "SimVP_RMSE_Wm2": arm["SimVP_RMSE_Wm2"],
            "AFNO_RMSE_Wm2": arm["AFNO_RMSE_Wm2"],
            "delta_AFNO_minus_SimVP_Wm2": arm["delta_AFNO_minus_SimVP_RMSE_Wm2"],
            "CI95_low": low,
            "CI95_high": high,
            "p_bootstrap_AFNO_lower": arm["paired_day_bootstrap"]["bootstrap_probability_AFNO_RMSE_lower"],
            "max_abs_same_head_replay_kt": arm["fixed_head_replay_max_abs_kt"],
        })

with (ROOT / "paired_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

pooled = {}
for group in ("AC_F", "AC_ST"):
    batches = [np.load(ROOT / f"{station}_{group}_paired_predictions.npz", allow_pickle=False)
               for station in ("Sili", "Zhujia")]
    truth = np.concatenate([item["true_ghi"].astype(np.float64) for item in batches])
    simvp = np.concatenate([item["simvp_ghi"].astype(np.float64) for item in batches])
    afno = np.concatenate([item["afno_ghi"].astype(np.float64) for item in batches])
    days = np.concatenate([item["init_day"] for item in batches])
    unique_days = np.unique(days)
    simvp_sse, afno_sse, counts = [], [], []
    for day in unique_days:
        chosen = days == day
        simvp_sse.append(np.square(simvp[chosen] - truth[chosen]).sum(dtype=np.float64))
        afno_sse.append(np.square(afno[chosen] - truth[chosen]).sum(dtype=np.float64))
        counts.append(int(chosen.sum()))
    simvp_sse, afno_sse = np.asarray(simvp_sse), np.asarray(afno_sse)
    counts = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(unique_days), size=(2000, len(unique_days)))
    denom = counts[draws].sum(axis=1)
    delta_boot = np.sqrt(afno_sse[draws].sum(axis=1) / denom) - np.sqrt(simvp_sse[draws].sum(axis=1) / denom)
    pooled[group] = {
        "rows": len(truth), "days": len(unique_days),
        "simvp_rmse": float(np.sqrt(np.mean((simvp - truth) ** 2))),
        "afno_rmse": float(np.sqrt(np.mean((afno - truth) ** 2))),
        "delta": float(np.sqrt(np.mean((afno - truth) ** 2)) - np.sqrt(np.mean((simvp - truth) ** 2))),
        "ci": np.quantile(delta_boot, [0.025, 0.975]).tolist(),
    }

fig, ax = plt.subplots(figsize=(10.0, 4.8), layout="constrained")
colors = {"Sili": "#0072B2", "Zhujia": "#D55E00"}
markers = {"AC_F": "o", "AC_ST": "s"}
y_positions = [3, 2, 1, 0]
for y, row in zip(y_positions, rows):
    delta, low, high = row["delta_AFNO_minus_SimVP_Wm2"], row["CI95_low"], row["CI95_high"]
    ax.errorbar(delta, y, xerr=[[delta - low], [high - delta]], fmt=markers[row["head"]],
                color=colors[row["station"]], ecolor=colors[row["station"]], capsize=4,
                markersize=7, linewidth=1.8)
ax.axvline(0, color="#333333", lw=1.2, ls="--")
ax.set_yticks(y_positions, [f'{row["station"]} · {row["head"].replace("_", "-")}' for row in rows])
ax.set_xlim(-0.9, 0.48)
ax.set_ylim(-0.7, 3.7)
ax.set_xlabel("Paired change in GHI RMSE (AFNO-COT − SimVP-COT; W m⁻²)")
ax.set_title("Same frozen GHI head, same validation rows")
ax.grid(axis="x", color="#D9DEE5", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(ROOT / "paired_rmse_delta.png", dpi=200, facecolor="white")
plt.close(fig)

lines = [
    "# AFNO COT paired frozen-GHI-head comparison (2026-09-23)",
    "",
    "AFNO COT replaces only future forecast COT. For each station and head, the exact same frozen GHI-head checkpoint, history COT, train-only COT normalization, satellite image and solar metadata, causal mask, row keys, observed GHI, and clear-sky denominator were held fixed.",
    "",
    "Negative ΔRMSE favors AFNO. Intervals are 2,000-replicate paired bootstrap intervals resampling BJT initialization days, with all leads for a day kept together.",
    "",
    "| Station | Head | Rows | SimVP RMSE | AFNO RMSE | Δ (AFNO−SimVP) | 95% day-bootstrap CI |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f'| {row["station"]} | {row["head"]} | {row["n_rows"]:,} | '
        f'{row["SimVP_RMSE_Wm2"]:.3f} | {row["AFNO_RMSE_Wm2"]:.3f} | '
        f'{row["delta_AFNO_minus_SimVP_Wm2"]:+.3f} | '
        f'[{row["CI95_low"]:+.3f}, {row["CI95_high"]:+.3f}] |'
    )
lines += [
    "",
    "Pooled over both stations (each station retains its own trained head; resampling BJT days jointly across stations):",
]
for group in ("AC_F", "AC_ST"):
    p = pooled[group]
    lines.append(f'- {group}: SimVP {p["simvp_rmse"]:.3f}, AFNO {p["afno_rmse"]:.3f}, '
                 f'Δ {p["delta"]:+.3f} W m⁻²; 95% CI [{p["ci"][0]:+.3f}, {p["ci"][1]:+.3f}] '
                 f'(n={p["rows"]:,}, {p["days"]} days).')
lines += [
    "",
    "All four intervals include zero. The direction is mixed at Sili (AC_F improves slightly; AC_ST is slightly worse) and AFNO is lower at Zhujia for both heads, but none of these validation-only differences is statistically resolved by this bootstrap.",
    "",
    "Integrity: 99,849 validation rows total (49,806 Sili; 50,043 Zhujia), 72 initialization days; test data were not accessed. Replaying the original SimVP-COT head predictions differs by at most 0.00264 kt. The SimVP COT bank matched the frozen cache exactly (maximum absolute difference 0).",
    "",
    "Scope: exploratory post-hoc input-swap diagnostic. The seed-42 GHI heads were trained on SimVP COT and selected using validation, so the result is neither an AFNO-trained GHI head nor independent test evidence. AFNO has a COT-distribution shift relative to the head's training input; a fair AFNO comparison for a paper claim requires retraining the same GHI-head architecture on AFNO COT using the same train/validation splits and then evaluating the held-out test once.",
    "",
    "Raw result: `complete.json`; per-row predictions: `*_paired_predictions.npz`; summary data: `paired_summary.csv`.",
    "",
    "Visualization workflow reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065.",
    "",
]
(ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote figure and summaries to {ROOT}")
