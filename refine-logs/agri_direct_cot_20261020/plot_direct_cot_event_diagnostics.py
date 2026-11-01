from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SCORE = ROOT / "results" / "direct_cot_ghi_validation_score_v1_local4060"
EVENT = SCORE / "event_diagnostics"
CSV = EVENT / "selected_event_rows_seed42_counterfactual.csv"
FIG = EVENT / "direct_cot_b2_selected_events.png"
PDF = EVENT / "direct_cot_b2_selected_events.pdf"

with CSV.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("No selected event rows")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "figure.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.2), constrained_layout=True)
panels = [
    ("sili", "signal", "Sili · selected thick-COT event"),
    ("sili", "control", "Sili · matched thick-COT control"),
    ("zhujia", "signal", "Zhujia · selected near-zero-COT degradation"),
    ("zhujia", "control", "Zhujia · matched near-zero-COT control"),
]
pred_specs = [
    ("ghi_none_seed42_wm2", "No COT", "#667f8c", "-"),
    ("ghi_simvp_r_seed42_wm2", "SimVP→R COT", "#4c78a8", "--"),
    ("ghi_p_c_seed42_wm2", "Direct COT (P-C)", "#f2a541", "-."),
    ("ghi_p_xc_seed42_wm2", "AGRI+COT direct (P-XC)", "#d45a4c", "-"),
    ("same_p_xc_head_neutral_cot_ghi_wm2", "P-XC head, neutral COT", "#777777", (0, (2, 1))),
]

for ax, (station, case, title) in zip(axes.flat, panels):
    rr = sorted((r for r in rows if r["station"] == station and r["case_type"] == case),
                key=lambda r: int(r["lead_minutes"]))
    if not rr:
        ax.set_visible(False)
        continue
    lead = np.array([int(r["lead_minutes"]) for r in rr])
    obs = np.array([float(r["ghi_obs_wm2"]) for r in rr])
    for key, label, color, style in pred_specs:
        ax.plot(lead, [float(r[key]) for r in rr], marker="o", ms=3.5,
                lw=1.5, color=color, ls=style, label=label)
    ax.plot(lead, obs, color="black", marker="s", ms=3.7, lw=1.8,
            label="Observed 15-min GHI")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Forecast lead from history end (min)")
    ax.set_ylabel("GHI (W m$^{-2}$)")
    ax.grid(alpha=.23, linewidth=.6)
    ax.set_xticks(lead[::max(1, len(lead)//6)])
    ax2 = ax.twinx()
    ax2.plot(lead, [float(r["cpp_center_cot"]) for r in rr], color="#7251a2",
             marker="^", ms=3.5, lw=1.1, ls=":", label="CPP reference COT")
    ax2.plot(lead, [float(r["p_xc_predicted_cot_center"]) for r in rr], color="#008b72",
             marker="D", ms=3, lw=1.2, ls="--", label="P-XC predicted COT")
    ax2.set_ylabel("Station-center COT")
    ax2.set_ylim(bottom=0)
    meta = rr[0]
    delta = float(meta["event_delta_rmse_pxc_minus_none_wm2"])
    cf_delta = float(np.sqrt(np.mean((np.array([float(r['ghi_p_xc_seed42_wm2']) for r in rr]) - obs) ** 2))
                     - np.sqrt(np.mean((np.array([float(r['same_p_xc_head_neutral_cot_ghi_wm2']) for r in rr]) - obs) ** 2)))
    ax.text(.02, .97,
            f"init {meta['init_time_utc']} UTC | {meta['bjt_month']} BJT\n"
            f"CPP stratum: {meta['event_bin']} | ΔRMSE(P-XC−noCOT)={delta:+.1f}\n"
            f"P-XC vs same head, neutral COT: {cf_delta:+.1f} W m$^{{-2}}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
            bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "alpha": .82, "edgecolor": ".8"})

handles, labels = axes.flat[0].get_legend_handles_labels()
h2, l2 = axes.flat[0].twinx().get_legend_handles_labels()
fig.legend(handles + h2, labels + l2, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle("B2 validation event diagnostics: COT forecast and station GHI trajectories\n"
             "Seed 42 examples selected by predeclared event rule; post-hoc illustration, not causal proof",
             fontweight="bold")
fig.savefig(FIG, dpi=220, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")
plt.close(fig)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "state": "COMPLETE_DIRECT_COT_B2_EVENT_FIGURES",
    "source_event_selection_manifest_sha256": sha(EVENT / "event_selection_manifest.json"),
    "source_event_csv_sha256": sha(CSV),
    "png_sha256": sha(FIG),
    "pdf_sha256": sha(PDF),
    "claim_scope": "validation-only seed-42 examples selected by predeclared diagnostic rule; not causal proof or independent test",
    "uses_future_observed_agri": False,
    "cpp_is_inference_input": False,
    "test_payload_opened": False,
    "buffer_payload_opened": False,
}
(EVENT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
