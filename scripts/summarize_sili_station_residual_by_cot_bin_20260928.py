"""Summarize saved validation station tracks by CPP-COT regime.

Input is the validation-only `station_tracks.npz` emitted by
audit_sili_transport_thickness_val512_20260928.py. CPP is a retrieval reference.
Bootstrap unit is BJT initialization date; adjacent windows can still overlap.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(source: Path, output: Path, repeats: int = 5000, seed: int = 20260928):
    z = np.load(source)
    y = z["cpp"].astype(np.float64)
    valid = z["cpp_valid"].astype(bool)
    base = z["transport"].astype(np.float64)
    learned = z["learned"].astype(np.float64)
    day = z["init_day_BJT"].astype(str)
    bins = [("0-5", 0.0, 5.0), ("5-10", 5.0, 10.0),
            ("10-30", 10.0, 30.0), ("30-60", 30.0, 60.0),
            ("60-100", 60.0, 100.00001)]
    rng = np.random.default_rng(seed)
    out = {"source": str(source), "reference": "CPP retrieval, not independent cloud truth",
           "split": "validation", "test_used": False,
           "paired_bootstrap": {"unit": "BJT initialization day", "repetitions": repeats,
                                "seed": seed, "interval": "percentile 95%"},
           "all_valid": {}, "by_reference_cot_bin": {}}

    def stats(q):
        result = {"n": int(q.sum())}
        for name, p in (("transport", base), ("learned", learned)):
            e = p[q] - y[q]
            result[name] = {"rmse": float(np.mean(e**2)**0.5),
                            "mae": float(np.mean(np.abs(e))),
                            "bias": float(np.mean(e))}
        d = learned[q] - base[q]
        result["learned_minus_transport_correction"] = {
            "mean": float(np.mean(d)), "median": float(np.median(d)),
            "fraction_negative": float(np.mean(d < 0))}
        return result

    def bootstrap(q):
        eligible = [d for d in np.unique(day) if np.any((day == d)[:, None] & q)]
        groups = [np.flatnonzero((day == d)[:, None].repeat(y.shape[1], axis=1) & q)
                  for d in eligible]
        delta = []
        for _ in range(repeats):
            draw = rng.integers(0, len(groups), size=len(groups))
            ix = np.concatenate([groups[i] for i in draw])
            yy, bb, pp = y.ravel()[ix], base.ravel()[ix], learned.ravel()[ix]
            delta.append(float(np.mean((pp - yy)**2)**0.5 - np.mean((bb - yy)**2)**0.5))
        return {"days": len(groups),
                "delta_rmse_learned_minus_transport": stats(q)["learned"]["rmse"] - stats(q)["transport"]["rmse"],
                "percentile_95pct": [float(v) for v in np.quantile(delta, [0.025, 0.975])]}

    out["all_valid"] = stats(valid)
    out["all_valid"]["paired_day_bootstrap"] = bootstrap(valid)
    for label, lo, hi in bins:
        q = valid & (y >= lo) & (y < hi)
        if not q.any():
            continue
        item = stats(q)
        item["paired_day_bootstrap"] = bootstrap(q)
        out["by_reference_cot_bin"][label] = item
    thick = valid & (y >= 30)
    out["thick_cot_ge30"] = stats(thick)
    out["thick_cot_ge30"]["paired_day_bootstrap"] = bootstrap(thick)
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260928)
    a = p.parse_args()
    main(a.input, a.output, a.repeats, a.seed)
