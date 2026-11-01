"""Source-bound figure for the fixed date-OOF conditional COT probes.

No fitting, sample selection, or payload beyond saved development scores occurs.
Run with Python + numpy + matplotlib from any working directory.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SRC = ROOT / "refine-logs/hubei_cot_conditional_information_20261023"
SOURCE_FILES = [SRC / "RESULTS.json", SRC / "DATE_SUMS.npz",
                SRC / "cross_source_attempt1/RESULTS.json",
                SRC / "cross_source_attempt1/DATE_SUMS.npz",
                SRC / "RESULT_REPORT.md", SRC / "score.py", SRC / "cross_source.py"]
J = json.loads(SOURCE_FILES[0].read_text(encoding="utf-8"))
C = json.loads(SOURCE_FILES[2].read_text(encoding="utf-8"))
assert J["state"] == "SCORED" and C["state"] == "COLD_CROSS_SOURCE_COMPLETE"
assert J["test_payloads_read"] == C["test_payloads_read"] == 0
all_ = J["summary"]["all"]
cross = C["summary"]["all"]
assert all_["dates"] == cross["dates"]
assert all_["rows"] == cross["rows"] == 32838

# Independent verification from the archived daily sufficient statistics.
verification = {}
for tag, path, result in [("main", SOURCE_FILES[1], J), ("cross", SOURCE_FILES[3], C)]:
    with np.load(path) as z:
        arms = z["arms"].tolist()
        stats = z["all"]
    total = stats.sum(0)
    rmse = np.sqrt(total[..., 0] / total[..., 1]).mean((0, 1))
    ref = np.array([result["summary"]["all"]["rmse"][arm] for arm in arms])
    err = float(np.max(np.abs(rmse - ref)))
    assert err < 1e-10
    weights = np.random.default_rng(20261023).multinomial(
        len(stats), np.ones(len(stats)) / len(stats), 5000)
    den = weights @ stats[..., 1].reshape(len(stats), -1)
    boot = np.sqrt((weights @ stats[..., 0].reshape(len(stats), -1)) / den)
    boot = boot.reshape(-1, 3, 3, len(arms)).mean((1, 2))
    ci_error = 0.0
    for key, item in result["summary"]["all"]["comparisons"].items():
        lhs, rhs = key.split("_minus_")
        ci = np.quantile(boot[:, arms.index(lhs)] - boot[:, arms.index(rhs)], [.025, .975])
        ci_error = max(ci_error, float(np.max(np.abs(ci - item["ci95"]))))
    assert ci_error < 1e-10
    verification[tag] = {"max_rmse_difference": err, "max_ci_difference": ci_error,
                         "dates": len(stats), "arms": arms}

GRAY, ORANGE, PURPLE = "#586976", "#C06100", "#80549A"
COLORS = [GRAY, GRAY, ORANGE, PURPLE]
ARMS = ["G1", "context", "forecast_COT", "observed_R_OFFLINE"]
LABELS = ["Frozen baseline", "Context probe", "+ predicted COT", "+ actual-image COT*"]
MARKERS = ["o", "D", "s", "^"]
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
         "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 8,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
         "mathtext.fontset": "stix", "savefig.dpi": 300}

def tag(ax, letter, phrase):
    ax.text(0, 1.06, f"({letter})  {phrase}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold")

rows = []
with plt.rc_context(STYLE):
    fig = plt.figure(figsize=(177 / 25.4, 133 / 25.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.04, h_pad=.06, wspace=.12, hspace=.18)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.20], width_ratios=[1, 1.05])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=axa)
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    yy = np.arange(4)[::-1]
    for y, arm, color, marker in zip(yy, ARMS, COLORS, MARKERS):
        val = all_["rmse"][arm]
        axa.plot(val, y, marker=marker, color=color, ms=5.7, linestyle="none")
        axa.text(val + 1.2, y, f"{val:.2f}", va="center", fontsize=8.5, color=color)
        rows.append({"panel": "a", "comparison": arm, "point": val,
                     "ci_low": "", "ci_high": "", "unit": "W m-2"})
    axa.set(yticks=yy, yticklabels=LABELS, ylim=(-.45, 3.5), xlim=(92, 136),
            xlabel=r"GHI RMSE (W m$^{-2}$)")
    axa.set_xticks([100, 110, 120, 130])
    axa.tick_params(axis="y", length=0, pad=5)
    axa.grid(axis="x", color=".90", lw=.6)
    axa.spines["left"].set_visible(False)
    tag(axa, "a", "Absolute error")
    axb.axvline(0, color=".45", lw=.8)
    for y, arm, color, marker in zip(yy[1:], ARMS[1:], COLORS[1:], MARKERS[1:]):
        v = all_["comparisons"][f"{arm}_minus_G1"]
        lo, hi = v["ci95"]
        p = v["delta_rmse"]
        axb.errorbar(p, y, xerr=[[p-lo], [hi-p]], color=color, fmt=marker,
                     ms=5.7, capsize=2.5, lw=1.15, markeredgewidth=.8)
        axb.text(23, y, f"{p:+.2f}", ha="right", va="center", fontsize=8.5, color=color)
        rows.append({"panel": "b", "comparison": f"{arm}_minus_G1", "point": p,
                     "ci_low": lo, "ci_high": hi, "unit": "W m-2"})
    axb.plot(0, yy[0], "o", color=GRAY, ms=5.7)
    axb.text(23, yy[0], "reference", ha="right", va="center", color=GRAY, fontsize=8)
    axb.set(xlim=(-35, 25), xlabel=r"$\Delta$RMSE vs. frozen baseline (W m$^{-2}$)")
    axb.set_xticks([-30, -20, -10, 0, 10])
    axb.tick_params(axis="y", length=0, labelleft=False)
    axb.spines["left"].set_visible(False)
    axb.grid(axis="x", color=".90", lw=.6)
    tag(axb, "b", "Paired difference; negative is better")

    # One cell is one source-pair, not a pooled score from another cohort.
    keys = [["forecast_train_forecast_infer", "forecast_train_observed_infer_OFFLINE"],
            ["observed_train_forecast_infer", "observed_train_observed_infer_OFFLINE"]]
    values = np.array([[cross["rmse"][key] for key in row] for row in keys])
    im = axc.imshow(values, cmap="Blues", norm=Normalize(95, 130), interpolation="nearest",
                     aspect="auto", extent=(-.5, 1.5, 1.5, -.5))
    for i in range(2):
        for j in range(2):
            value = values[i, j]
            delta = value - cross["rmse"]["G1"]
            color = "white" if value > 115 else "#172530"
            axc.text(j, i-.08, f"{value:.2f}", va="center", ha="center", color=color,
                     fontsize=12, fontweight="bold")
            axc.text(j, i+.20, f"({delta:+.2f})", va="center", ha="center", color=color,
                     fontsize=8.5)
            rows.append({"panel": "c", "comparison": keys[i][j], "point": value,
                         "ci_low": "", "ci_high": "", "unit": "W m-2"})
    axc.set_xticks([0, 1], ["Predicted", "Actual-image*"], fontsize=8.5)
    axc.set_yticks([0, 1], ["Predicted", "Actual-image*"], fontsize=8.5)
    axc.set_xlabel("Inference COT source")
    axc.set_ylabel("Training COT source")
    axc.set_xticks([.5], minor=True)
    axc.set_yticks([.5], minor=True)
    axc.grid(which="minor", color="white", lw=1.2)
    axc.tick_params(which="both", length=0)
    axc.spines["left"].set_visible(False)
    axc.spines["bottom"].set_visible(False)
    # Dash outline redundantly identifies inference unavailable in deployment.
    axc.add_patch(Rectangle((.5, -.5), 1, 2, fill=False, ec="#303030", lw=1.1,
                           ls=(0, (4, 2)), clip_on=False))
    cb = fig.colorbar(im, ax=axc, orientation="horizontal", fraction=.09, pad=.20,
                      aspect=28, ticks=[100, 110, 120, 130])
    cb.set_label(r"GHI RMSE (W m$^{-2}$); parentheses: $\Delta$ vs. baseline", fontsize=8)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(.5)
    tag(axc, "c", "COT-source substitution")

    axd.axhline(0, color=".45", lw=.8)
    for arm, color, marker, offset, label in [
            ("forecast_COT", ORANGE, "s", -.04, "Predicted COT"),
            ("observed_R_OFFLINE", PURPLE, "^", .04, "Actual-image COT*")]:
        p, low, high = [], [], []
        for h in range(1, 5):
            val = J["summary"][f"{h}h"]["comparisons"][f"{arm}_minus_context"]
            p.append(val["delta_rmse"])
            low.append(val["ci95"][0]); high.append(val["ci95"][1])
            rows.append({"panel": "d", "comparison": f"{h}h:{arm}_minus_context",
                         "point": p[-1], "ci_low": low[-1], "ci_high": high[-1], "unit": "W m-2"})
        p, low, high = map(np.array, (p, low, high))
        axd.errorbar(np.arange(1, 5)+offset, p, yerr=[p-low, high-p], color=color,
                     marker=marker, ms=4.7, lw=1.05, capsize=2.1,
                     linestyle="-" if arm == "forecast_COT" else "--", label=label)
    axd.set_xticks([1, 2, 3, 4], ["0–1", "1–2", "2–3", "3–4"])
    axd.set(xlabel="Forecast lead interval (h)", ylabel=r"$\Delta$RMSE vs. context (W m$^{-2}$)",
            xlim=(.7, 4.3), ylim=(-56, 5))
    axd.set_yticks([-50, -40, -30, -20, -10, 0])
    axd.grid(axis="y", color=".90", lw=.6)
    axd.legend(loc="lower left", frameon=False, handlelength=1.7,
               borderpad=.2, labelspacing=.35)
    tag(axd, "d", "Increment across forecast horizons")
    # No in-figure title; status and details belong to the bilingual caption.
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"cot_conditional_information.{ext}", dpi=300)
    fig.savefig(OUT / "cot_conditional_information_print_preview.png", dpi=150)
    plt.close(fig)

with (OUT / "plotted_values.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
manifest = {
    "figure": "cot_conditional_information", "physical_size_mm": [177, 133],
    "scope": "Hubei development-date three-fold OOF probes; not independent validation/test",
    "rows": all_["rows"], "target_dates": len(all_["dates"]), "first_date": min(all_["dates"]),
    "last_date": max(all_["dates"]), "horizons_minutes": [15, 240],
    "aggregation": "RMSE separately for station x original GHI model source; equal-weight mean across 3 x 3",
    "uncertainty": "Saved paired target-date cluster percentile bootstrap CI, 5000 replicates, seed20261023",
    "source_substitution": "Posthoc cold reuse of original fitted probes, no new fits",
    "COT_reference": "Frozen R applied to actual future AGRI; not independent CPP truth",
    "inference_availability": "Actual-image COT column is offline; predicted column has causal inputs only",
    "transformations": "Read stored scores; verify point estimates from saved daily SSE/counts; no refitting or sample filtering",
    "journal_requirements": "Not yet selected; provisional print dimensions, not a compliance claim",
    "semantic_colors": {"reference_and_context": GRAY, "predicted_future_COT": ORANGE,
                        "actual_future_COT_offline": PURPLE},
    "verification": verification,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_FILES},
    "software": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"output": str(OUT), "verification": verification}, indent=2))
