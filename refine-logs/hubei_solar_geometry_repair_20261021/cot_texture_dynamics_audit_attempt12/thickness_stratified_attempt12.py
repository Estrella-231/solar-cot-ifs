"""Validation-only CPP-thickness error decomposition on archived common keys."""
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
INP = HERE / "inputs"
OUT = HERE / "thickness_stratified"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stamp(s):
    s = "".join(c for c in s if c.isdigit())
    datetime.strptime(s, "%Y%m%d%H%M")
    return s


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    receipt = json.loads((INP / "forecast_cot_metrics.json").read_text())
    manifest = json.loads((INP / "source_manifest.json").read_text())
    assert manifest["test_payloads_read"] == manifest["buffer_payloads_read"] == 0
    bindings = {
        "validation.npy": "forecast_cot_bank_sha256",
        "keys.csv": "simvp_key_sha256",
        "val_keys.csv": "cpp_reference_keys_sha256",
        "val_reference_cpp_cot.npy": "cpp_reference_array_sha256",
        "val_valid_mask.npy": "cpp_mask_sha256",
    }
    for name, field in bindings.items():
        assert sha(INP / name) == receipt[field], name
    fcast, real_r, cpp, masks = [np.load(INP / p, mmap_mode="r", allow_pickle=False)
        for p in ("validation.npy", "val_prediction_cot.npy",
                  "val_reference_cpp_cot.npy", "val_valid_mask.npy")]
    sr, rr = rows(INP / "keys.csv"), rows(INP / "val_keys.csv")
    assert fcast.shape == (len(sr), 16, 1, 16, 16)
    assert real_r.shape == cpp.shape == masks.shape == (len(rr), 1, 16, 16)
    ri = {stamp(r["time_bjt"]): i for i, r in enumerate(rr)}
    assert len(ri) == len(rr) and all(r["split"] == "val" for r in rr)
    # Each date row stores n, forecast SSE/SAE/bias, persistence SSE/SAE/bias.
    stats = defaultdict(lambda: defaultdict(lambda: np.zeros(7, dtype=np.float64)))
    legacy = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    key_set = set()
    for si, row in enumerate(sr):
        assert row["split"] == "val"
        init = stamp(row["init_time_bjt"])
        ts = [stamp(t) for t in row["target_times_bjt"].split("|")]
        assert len(ts) == 16
        for li, target in enumerate(ts):
            lead = 15 * (li + 1)
            assert target == (datetime.strptime(init, "%Y%m%d%H%M") + timedelta(minutes=lead)).strftime("%Y%m%d%H%M")
            if init not in ri or target not in ri:
                continue
            hi, ti = ri[init], ri[target]
            f, h, o, c = [np.asarray(x, dtype=np.float64) for x in
                          (fcast[si, li, 0], real_r[hi, 0], real_r[ti, 0], cpp[ti, 0])]
            m = np.asarray(masks[ti, 0], dtype=bool).copy()
            m &= np.isfinite(f) & np.isfinite(h) & np.isfinite(o) & np.isfinite(c)
            if not m.any():
                continue
            assert np.all(c[m] >= 0)
            key = (init, target, lead)
            assert key not in key_set
            key_set.add(key)
            ef, eh, eo = f - c, h - c, o - c
            legacy[lead] += [m.sum(), np.square(ef[m]).sum(), np.square(eh[m]).sum(), np.square(eo[m]).sum()]
            bins = {"all": m, "COT_lt_1": m & (c < 1),
                    "COT_1_to_10": m & (c >= 1) & (c < 10),
                    "COT_10_to_30": m & (c >= 10) & (c < 30),
                    "COT_ge_30": m & (c >= 30)}
            for horizon in ("all_leads", f"hour_{li // 4 + 1}"):
                for label, bm in bins.items():
                    if bm.any():
                        a, b = ef[bm], eh[bm]
                        stats[(horizon, label)][target[:8]] += [bm.sum(), a @ a,
                            np.abs(a).sum(), a.sum(), b @ b, np.abs(b).sum(), b.sum()]
    assert len(key_set) == receipt["n_paired_init_target_lead_keys"] == 1164
    # Independently reproduce all archived full-patch method RMSE/counts first.
    max_drift = 0.0
    for lead, vals in legacy.items():
        archived = receipt["metrics"][f"{lead}|all_patch|ALL_PATCH"]
        for j, name in enumerate(("forecast", "history_persistence", "offline_future_observed_agri_to_R"), 1):
            assert int(vals[0]) == archived[name]["n_pixels"]
            drift = abs(np.sqrt(vals[j] / vals[0]) - archived[name]["rmse_cot"])
            max_drift = max(max_drift, drift)
            assert drift < 1e-10, (lead, name, drift)
    output = []
    date_rows = []
    for idx, ((horizon, label), bydate) in enumerate(sorted(stats.items())):
        dates = sorted(bydate)
        mat = np.stack([bydate[d] for d in dates])
        total = mat.sum(axis=0)
        n, sf, af, bf, sh, ah, bh = total
        rf, rh = np.sqrt(sf / n), np.sqrt(sh / n)
        rng = np.random.default_rng(42000 + idx)
        draws = rng.integers(0, len(dates), size=(5000, len(dates)))
        boot = mat[draws].sum(axis=1)
        diff = np.sqrt(boot[:, 1] / boot[:, 0]) - np.sqrt(boot[:, 4] / boot[:, 0])
        lo, up = np.quantile(diff, [.025, .975])
        output.append(dict(horizon=horizon, cpp_thickness_bin=label,
            pixel_forecast_records=int(n), target_dates=len(dates),
            forecast_rmse=float(rf), persistence_rmse=float(rh),
            forecast_minus_persistence_rmse=float(rf-rh), ci95_low=float(lo), ci95_high=float(up),
            relative_rmse_improvement_percent=float(100 * (rh-rf) / rh),
            forecast_mae=float(af/n), persistence_mae=float(ah/n),
            forecast_bias=float(bf/n), persistence_bias=float(bh/n)))
        for date, vals in zip(dates, mat):
            date_rows.append(dict(horizon=horizon, cpp_thickness_bin=label, target_date=date,
                **dict(zip(("n", "forecast_sse", "forecast_sae", "forecast_bias_sum",
                            "persistence_sse", "persistence_sae", "persistence_bias_sum"), vals.tolist()))))
    OUT.mkdir()
    for name, values in (("metrics.csv", output), ("date_sufficient_statistics.csv", date_rows)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]))
            w.writeheader(); w.writerows(values)
    result = dict(state="VALIDATION_ONLY_COMMON_CPP_THICKNESS_DECOMPOSITION",
        n_common_keys=len(key_set), archived_48_full_patch_scores_max_abs_drift=max_drift,
        input_sha256={p.name: sha(p) for p in INP.iterdir() if p.is_file()},
        source_script_sha256=sha(Path(__file__)),
        metric_scope="full 16x16; pooled pixel-forecast records; no spatial averaging input",
        uncertainty="5000 paired resamples of BJT target dates within each thickness/horizon stratum",
        reference="CPP retrieval reference; conditional exploratory validation; not independent test",
        test_payloads_read=0, buffer_payloads_read=0, metrics=output)
    (OUT / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps([r for r in output if r["horizon"] == "all_leads"], indent=2))


if __name__ == "__main__":
    main()
