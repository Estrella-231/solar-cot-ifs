"""Paired validation audit for the 2026-09-27 Zhujia COT residual pilot."""
import json
from pathlib import Path

import numpy as np


def metrics(y, p):
    e = p - y
    return {"n": int(len(y)), "rmse": float(np.mean(e**2) ** 0.5),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))}


def main(run: Path, bank: Path, base: Path, out: Path):
    names = ("zero", "fusion", "residual")
    arr = {k: np.load(run / f"{k}_val_predictions.npz") for k in names}
    ref = arr["zero"]
    for k in names[1:]:
        assert np.array_equal(arr[k]["row_ids"], ref["row_ids"]), f"row IDs differ: {k}"
        assert np.array_equal(arr[k]["lead"], ref["lead"]), f"leads differ: {k}"
        assert np.array_equal(arr[k]["true_ghi"], ref["true_ghi"]), f"truth differs: {k}"
    row = ref["row_ids"]
    seq_all = np.load(base / "sequence.npy", mmap_mode="r")
    station_all = np.load(base / "station.npy", mmap_mode="r")
    assert np.all(station_all[row] == 1), "non-Zhujia rows found"
    contract = json.loads((bank / "complete.json").read_text())
    day_map = contract["initialization_BJT_days"]["val"]
    seq = seq_all[row]
    days = np.asarray([day_map[str(int(s))] for s in seq])
    unique_days = np.unique(days)
    assert len(unique_days) >= 2
    y = ref["true_ghi"]
    preds = {k: arr[k]["pred_ghi"] for k in names}
    results = {k: metrics(y, preds[k]) for k in names}
    lead_results = {}
    for li in sorted(np.unique(ref["lead"])):
        mask = ref["lead"] == li
        lead_results[str(int(li) * 15 + 15)] = {
            k: metrics(y[mask], preds[k][mask])["rmse"] for k in names
        }
    # Paired day-cluster bootstrap: resample initialization days, retaining every
    # station/lead row from a selected day. Candidate-minus-zero delta is negative
    # when candidate RMSE is better.
    rng = np.random.default_rng(20260927)
    day_rows = [np.flatnonzero(days == d) for d in unique_days]
    boot = {}
    for k in ("fusion", "residual"):
        deltas = []
        for _ in range(5000):
            draw = rng.integers(0, len(day_rows), size=len(day_rows))
            ix = np.concatenate([day_rows[i] for i in draw])
            deltas.append(metrics(y[ix], preds[k][ix])["rmse"] - metrics(y[ix], preds["zero"][ix])["rmse"])
        boot[k] = {"unit": "BJT initialization day", "days": int(len(unique_days)),
                   "repetitions": len(deltas), "seed": 20260927,
                   "delta_rmse_candidate_minus_zero": results[k]["rmse"] - results["zero"]["rmse"],
                   "percentile_95pct": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
                   "fraction_candidate_better": float(np.mean(np.asarray(deltas) < 0))}
    output = {
        "state": "PAIRED_VALIDATION_AUDIT_COMPLETE",
        "scope": "single-seed 512-train-sequence/128-validation-sequence Zhujia pilot; validation only",
        "test_used": False, "paired_rows": int(len(row)), "initialization_days": int(len(unique_days)),
        "initialization_day_range_BJT": [str(min(unique_days)), str(max(unique_days))],
        "row_ids_truth_leads_identical": True,
        "results": results, "lead_rmse_W_m2": lead_results,
        "paired_day_bootstrap": boot,
        "residual_equals_zero_predictions_exactly": bool(np.array_equal(preds["residual"], preds["zero"])),
        "source_run": str(run), "validation_contract": contract["state"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    main(a.run, a.bank, a.base, a.out)
