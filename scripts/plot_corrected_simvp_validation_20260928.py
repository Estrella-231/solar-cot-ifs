"""Plot corrected lead-wise frozen SimVP->R validation metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results/regional_station_cot_20260927/simvp_axis_corrected_val_20260928/corrected_simvp_metrics.json"
out = src.parent
d = json.loads(src.read_text(encoding="utf-8"))
leads = np.arange(15, 241, 15)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
for name, label, ax in (("rmse", "All CPP-valid pixels", axes[0]),
                        ("thick_rmse", "CPP COT ≥ 30", axes[1])):
    for scope, color, marker in (("patch", "#0072B2", "o"), ("station", "#D55E00", "s")):
        vals = [x[name] for x in d["per_lead"][scope]]
        ax.plot(leads, vals, marker=marker, color=color, label=scope)
    ax.set_title(label)
    ax.set_xlabel("Lead from history end (min)")
    ax.set_ylabel("COT RMSE")
    ax.grid(alpha=.25)
    ax.legend()
fig.suptitle("Corrected SimVP→R validation: lead axis preserved, same support mask")
fig.savefig(out / "corrected_simvp_lead_rmse.png", dpi=180)
fig.savefig(out / "corrected_simvp_lead_rmse.pdf")
print(out / "corrected_simvp_lead_rmse.png")
