"""Validation-only spatial error audit for the saved staged COT retriever."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cot_staged_diagnostic_20260924/source_cases"
RUN = ROOT / "results/cot_real_agri_staged_hurdle_seed42_20260924"
OUT = SOURCE / "border_context_audit_20260925"


def rate(select, event):
    count = int(select.sum())
    return {"n": count, "events": int((select & event).sum()),
            "rate": float((select & event).sum() / count) if count else None}


def main():
    OUT.mkdir(exist_ok=True)
    rows = pd.read_csv(SOURCE / "pack_readonly/rows.csv")
    val = rows[rows.split == "validation"]
    cls = np.load(RUN / "stage1_classifier/best_validation_probabilities.npz")
    final = np.load(RUN / "stage2_conditional_regressors/best_validation_predictions.npz")
    assert np.array_equal(cls["row_indices"], val["index"].to_numpy())
    assert np.array_equal(final["row_indices"], cls["row_indices"])
    ids = val.index.to_numpy()
    reference = np.asarray(np.load(SOURCE / "pack_readonly/target.npy", mmap_mode="r")[ids, 0], np.float32) * 100
    valid = np.asarray(np.load(SOURCE / "pack_readonly/mask.npy", mmap_mode="r")[ids, 0], bool)
    cossoz = np.asarray(np.load(SOURCE / "pack_readonly/x_raw.npy", mmap_mode="r")[ids, 13], np.float32)
    prob = cls["cloud_probability"][:, 0]
    predicted = final["pred_cot"][:, 0]
    assert reference.shape == prob.shape == predicted.shape == valid.shape == (6536, 16, 16)
    assert np.array_equal(cls["mask"][:, 0], valid)
    assert np.allclose(final["reference_cot"][:, 0][valid], reference[valid], atol=1e-5)
    yy, xx = np.indices((16, 16))
    dist = np.minimum.reduce((yy, xx, 15 - yy, 15 - xx))[None]
    nearzero = valid & (reference < .1)
    false = nearzero & (prob >= .5)
    severe = nearzero & (predicted >= 10)
    # Only within each source patch, without wrapping or reaching another time.
    adjacent = maximum_filter(valid & (reference >= .1), size=(1, 3, 3), mode="constant", cval=0)
    adjacent &= reference < .1
    station = val.station_id.to_numpy()
    entries = []
    for dname, dsel in [("outer", dist == 0), ("ring1", dist == 1),
                         ("ring2_3", (dist >= 2) & (dist <= 3)), ("inner4plus", dist >= 4)]:
        for cloud_name, cloud_sel in [("all", np.ones_like(adjacent, bool)),
                                       ("adjacent_cloud", adjacent), ("no_adjacent_cloud", ~adjacent)]:
            select = nearzero & dsel & cloud_sel
            entry = {"distance": dname, "cloud_context": cloud_name,
                     "false_cloud": rate(select, false), "severe_cot_ge10": rate(select, severe)}
            entries.append(entry)
    # Directly standardize the border/interior rates over the shared strata:
    # station x cosSOZ bin x adjacent CPP-cloud indicator.
    sun_edges = [-1, .2, .4, .6, .8, 1.01]
    strata = []
    for st in sorted(set(station)):
        for lo, hi in zip(sun_edges[:-1], sun_edges[1:]):
            sun = (cossoz >= lo) & (cossoz < hi)
            for edge in (False, True):
                ctx = (station[:, None, None] == st) & sun & (adjacent == edge) & nearzero
                outer = ctx & (dist == 0)
                inner = ctx & (dist >= 1)
                if outer.sum() < 20 or inner.sum() < 20:
                    continue
                strata.append({"station": st, "sun_bin": [lo, hi], "adjacent_cloud": edge,
                               "outer": rate(outer, false), "inner": rate(inner, false),
                               "outer_severe": rate(outer, severe), "inner_severe": rate(inner, severe)})
    denom = sum(x["outer"]["n"] + x["inner"]["n"] for x in strata)
    weights = [(x["outer"]["n"] + x["inner"]["n"]) / denom for x in strata]
    standardized = {
        "false_cloud": {name: sum(w * x[name]["rate"] for w, x in zip(weights, strata))
                        for name in ("outer", "inner")},
        "severe_cot_ge10": {name: sum(w * x[name]["rate"] for w, x in zip(weights, strata))
                            for name in ("outer_severe", "inner_severe")},
        "shared_strata": len(strata), "shared_nearzero_pixels": denom,
        "context": "standardized for station, broad cosSOZ bins, and immediate CPP cloud adjacency; descriptive, not causal",
    }
    summary = {"scope": "saved real-AGRI validation predictions only; no training or test",
               "reference": "CPP COT in physical units; near-zero <0.1; false-cloud P>=0.5",
               "cohort_rows": len(val), "nearzero_pixels": int(nearzero.sum()),
               "by_distance_and_context": entries, "matched_context_standardized": standardized,
               "strata": strata}
    (OUT / "border_context_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([{ "distance": x["distance"], "context": x["cloud_context"],
                    "n": x["false_cloud"]["n"], "false_cloud_rate": x["false_cloud"]["rate"],
                    "severe_cot_ge10_rate": x["severe_cot_ge10"]["rate"]} for x in entries]).to_csv(OUT / "border_context_rates.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ["outer", "ring1", "ring2-3", "inner4+"]
    for axis, metric, title in zip(ax, ("false_cloud", "severe_cot_ge10"),
                                   ("CPP near-zero → P(cloud) ≥ 0.5", "CPP near-zero → predicted COT ≥ 10")):
        for name, marker in (("all", "o"), ("adjacent_cloud", "s"), ("no_adjacent_cloud", "^")):
            vals = [next(x[metric]["rate"] for x in entries if x["distance"] == d and x["cloud_context"] == name)
                    for d in ("outer", "ring1", "ring2_3", "inner4plus")]
            axis.plot(labels, np.array(vals) * 100, marker=marker, label=name.replace("_", " "))
        axis.set(title=title, ylabel="Rate among valid CPP near-zero pixels (%)")
        axis.grid(alpha=.2)
    ax[0].legend(frameon=False)
    fig.savefig(OUT / "border_context_rates.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"nearzero_pixels": summary["nearzero_pixels"], "standardized": standardized,
                      "outer": entries[0], "inner4plus": entries[9]}, indent=2))


if __name__ == "__main__":
    main()
