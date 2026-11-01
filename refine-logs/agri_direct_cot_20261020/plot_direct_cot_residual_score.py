from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SCORE = ROOT / "results" / "direct_cot_ghi_residual_v1_local4060" / "score" / "direct_cot_residual_metrics.json"
OUT = SCORE.parent
PNG = OUT / "direct_cot_residual_b3_validation.png"
PDF = OUT / "direct_cot_residual_b3_validation.pdf"
data = json.loads(SCORE.read_text(encoding="utf-8"))

items = [
    ("Residual − no COT", "all_16_leads", "residual_minus_none", "#bb3e03"),
    ("Residual − no COT", "lead_15_60", "residual_minus_none", "#bb3e03"),
    ("Residual − SimVP→R", "all_16_leads", "residual_minus_simvp_r", "#256d85"),
    ("Residual − SimVP→R", "lead_15_60", "residual_minus_simvp_r", "#256d85"),
]
ys = np.arange(len(items))[::-1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(10.8, 5.6), constrained_layout=True)
for y, (label, band, contrast, color) in zip(ys, items):
    r = data["contrasts"][band][contrast]
    point = r["station_equal_seed_equal_point_delta_rmse_wm2"]
    lo, hi = r["paired_date_block_bootstrap_ci95_wm2"]
    ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", ms=6,
                capsize=4, lw=1.8, color=color)
    ax.text(hi + .25, y,
            f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}] W m$^{{-2}}$; seeds {r['seeds_improved']}/3 improve",
            va="center", fontsize=8)
ax.axvline(0, color="black", lw=1)
ax.set_yticks(ys, [f"{a} · {'all 16 leads' if b == 'all_16_leads' else '+15–60 min'}" for a,b,_,_ in items])
ax.set_xlabel("Candidate RMSE − comparator RMSE (W m$^{-2}$); positive = candidate worse")
ax.set_title("B3 zero-initialized conditional residual: no reproducible GHI gain", fontweight="bold")
ax.grid(axis="x", alpha=.24)
ax.set_xlim(-5.0, 8.5)
fig.text(.01, -.02,
         "95% CI: 5,000 paired resamples of 26 frozen monthly two-day validation blocks. "
         "Three matched seeds; V is development validation, not independent test.", fontsize=8)
fig.savefig(PNG, dpi=220, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")
plt.close(fig)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

manifest = {"state": "COMPLETE_DIRECT_COT_RESIDUAL_B3_FIGURE",
            "score_sha256": sha(SCORE), "png_sha256": sha(PNG), "pdf_sha256": sha(PDF),
            "test_payload_opened": False, "buffer_payload_opened": False,
            "future_observed_agri_used_as_input": False, "cpp_used_as_input": False}
(OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
