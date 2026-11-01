from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cot_value_first_20261013"
OUT = Path(__file__).resolve().parent


def main() -> None:
    pred = np.load(SRC / "validation_predictions_monthly_R_seed42.npz")
    clp = np.load(SRC / "clp_reference_monthly_R_seed42.npz")
    assert np.array_equal(pred["sample_indices"], clp["sample_indices"])
    assert np.array_equal(clp["common_mask"], pred["mask"] & clp["clp_valid"])
    assert np.all(clp["clp_valid"][clp["common_mask"]])

    y = pred["reference_cot"].astype(np.float64)[:, 0]
    p = np.expm1(pred["pred_log1p_cot"].astype(np.float64)[:, 0])
    m = clp["common_mask"][:, 0]
    c = clp["clp"][:, 0]
    stations = clp["station"].astype(str)
    times = clp["target_time_utc"].astype(str)
    labels = {0: "clear", 1: "water", 2: "ice", 3: "undetermined", 255: "fill"}
    cot_bins = [(0.0, 0.1, "[0,0.1)"), (0.1, 5.0, "[0.1,5)"),
                (5.0, 30.0, "[5,30)"), (30.0, np.inf, "[30,inf)")]
    cloud = m & np.isin(c, (1, 2))
    clear = m & (c == 0)

    def metrics(sel: np.ndarray) -> dict:
        n = int(sel.sum())
        if not n:
            return {"n": 0}
        e = p[sel] - y[sel]
        return {"n": n, "rmse": float(np.sqrt(np.mean(e * e))),
                "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
                "sse": float(np.sum(e * e)), "mean_ref": float(np.mean(y[sel])),
                "mean_pred": float(np.mean(p[sel]))}

    rows: list[dict] = []
    cross: dict[str, dict[str, int]] = {}
    scopes: dict[str, dict] = {}
    for s in sorted(set(stations)):
        sm1 = stations == s
        sm = sm1[:, None, None]
        cross[s] = {}
        for cl in (0, 1, 2, 3):
            for lo, hi, bname in cot_bins:
                key = f"CLP={cl}|COT={bname}"
                sel = m & sm & (c == cl) & (y >= lo) & (y < hi)
                n = int(sel.sum())
                cross[s][key] = n
                rows.append({"station": s, "clp": cl, "clp_name": labels.get(cl, str(cl)),
                             "reference_cot_bin": bname, "n": n})

        scope = {"all_valid": metrics(m & sm), "clear_CLP0": metrics(m & sm & (c == 0)),
                 "cloud_CLP1or2": metrics(m & sm & np.isin(c, (1, 2)))}
        for lo, hi, bname in cot_bins:
            scope[f"reference_cot_{bname}"] = metrics(m & sm & (y >= lo) & (y < hi))
        se = scope["all_valid"]["sse"]
        scope["sse_share_clear"] = scope["clear_CLP0"]["sse"] / se
        scope["sse_share_cloud"] = scope["cloud_CLP1or2"]["sse"] / se
        scopes[s] = scope

    # Combine station metrics equally for RMSE/MAE/bias; separately give pooled pixel counts/SSE.
    stvals = list(scopes.values())
    for key in ("all_valid", "clear_CLP0", "cloud_CLP1or2"):
        scopes["station_equal_2site_aggregate_" + key] = {
            "rmse_mean_of_station_rmse": float(np.mean([q[key]["rmse"] for q in stvals])),
            "mae_mean_of_station_mae": float(np.mean([q[key]["mae"] for q in stvals])),
            "bias_mean_of_station_bias": float(np.mean([q[key]["bias"] for q in stvals])),
            "pooled_n": int(sum(q[key]["n"] for q in stvals)),
            "pooled_sse": float(sum(q[key]["sse"] for q in stvals)),
        }

    # Diagnostic-only perfect clear reference gate: not deployable and not a
    # candidate score. The cloudy pixels are unchanged by construction.
    p_gate = np.where(c == 0, 0.0, p)
    gate_scopes = {}
    for s in sorted(set(stations)):
        sm = (stations == s)[:, None, None]
        def gate_metric(sel: np.ndarray) -> dict:
            n = int(sel.sum())
            if not n:
                return {"n": 0}
            e = p_gate[sel] - y[sel]
            return {"n": n, "rmse": float(np.sqrt(np.mean(e * e))),
                    "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
                    "sse": float(np.sum(e * e))}
        gate_scopes[s] = {"all_valid": gate_metric(m & sm),
                          "clear_CLP0": gate_metric(m & sm & (c == 0)),
                          "cloud_CLP1or2": gate_metric(m & sm & np.isin(c, (1, 2)))}
    gate_scopes["station_equal_all_valid_rmse"] = float(np.mean(
        [gate_scopes[s]["all_valid"]["rmse"] for s in sorted(set(stations))]))
    gate_scopes["station_equal_baseline_rmse"] = float(np.mean(
        [scopes[s]["all_valid"]["rmse"] for s in sorted(set(stations))]))
    gate_scopes["oracle_gate_gain_wm_cot_rmse"] = (
        gate_scopes["station_equal_all_valid_rmse"] - gate_scopes["station_equal_baseline_rmse"])

    monthly: dict[str, dict] = {}
    for month in sorted({(np.datetime64(t.replace(" ", "T"), "m") + np.timedelta64(8, "h")).astype("datetime64[M]").astype(str) for t in times}):
        # Input target_time_utc is a canonical timestamp. Convert to BJT for monthly label.
        idx = np.array([i for i, t in enumerate(times)
                        if (np.datetime64(t.replace(" ", "T"), "m") + np.timedelta64(8, "h")).astype("datetime64[M]").astype(str) == month])
        monthly[month] = {}
        for s in sorted(set(stations)):
            sm = stations[idx] == s
            mm = m[idx][sm]
            cc = c[idx][sm]
            yy = y[idx][sm]
            pp = p[idx][sm]
            def local_metric(mask: np.ndarray) -> dict:
                n = int(mask.sum())
                if not n:
                    return {"n": 0}
                err = pp[mask] - yy[mask]
                return {"n": n, "rmse": float(np.sqrt(np.mean(err**2))),
                        "sse": float(np.sum(err**2))}
            monthly[month][s] = {"clear": local_metric(mm & (cc == 0)),
                                  "cloud": local_metric(mm & np.isin(cc, (1, 2)))}

    out = {"source_prediction": str(SRC / "validation_predictions_monthly_R_seed42.npz"),
           "source_clp": str(SRC / "clp_reference_monthly_R_seed42.npz"),
           "n_samples": int(len(stations)), "common_valid_pixels": int(m.sum()),
           "clp_counts": {labels.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(c[m], return_counts=True))},
           "cloud_reference_definition": "CPP CLP in {1,2}; clear=CLP 0; other classes excluded",
           "cot_bins": {s: cross[s] for s in cross}, "metrics": scopes,
           "oracle_clear_gate_diagnostic": gate_scopes,
           "monthly_by_station": monthly,
           "limitations": ["CPP COT and CLP are from the same product chain, not independent truth",
                           "one R seed; validation only; no test payload opened",
                           "CLP class 3 and fill/other categories are not relabelled as clear"]}
    (OUT / "conditional_cot_metrics_20261014.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (OUT / "conditional_cot_metrics_20261014.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["station", "clp", "clp_name", "reference_cot_bin", "n"])
        w.writeheader(); w.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        sites = sorted(set(stations))
        names = ["Sili", "Zhujia"]
        xx = np.arange(2)
        width = 0.34
        vals_clear = [scopes[s]["clear_CLP0"]["rmse"] for s in sites]
        vals_cloud = [scopes[s]["cloud_CLP1or2"]["rmse"] for s in sites]
        axs[0].bar(xx - width/2, vals_clear, width, label="CLP clear (0)", color="#d88942")
        axs[0].bar(xx + width/2, vals_cloud, width, label="CLP cloudy (1/2)", color="#347e9a")
        axs[0].set_xticks(xx, names); axs[0].set_ylabel("COT RMSE"); axs[0].set_title("Conditional COT error")
        axs[0].legend(frameon=False)
        clr = [scopes[s]["clear_CLP0"]["sse"] / scopes[s]["all_valid"]["sse"] * 100 for s in sites]
        cld = [scopes[s]["cloud_CLP1or2"]["sse"] / scopes[s]["all_valid"]["sse"] * 100 for s in sites]
        axs[1].bar(xx, clr, label="Clear SSE share", color="#d88942")
        axs[1].bar(xx, cld, bottom=clr, label="Cloud SSE share", color="#347e9a")
        axs[1].set_xticks(xx, names); axs[1].set_ylim(0, 100); axs[1].set_ylabel("Share of total COT squared error (%)")
        axs[1].set_title("Both regimes contribute substantially"); axs[1].legend(frameon=False)
        fig.suptitle("New monthly R validation: clear and cloudy error decomposition")
        fig.savefig(OUT / "conditional_cot_error_20261014.png", dpi=180)
        plt.close(fig)
    except ImportError:
        out["figure_status"] = "skipped: matplotlib unavailable"
        (OUT / "conditional_cot_metrics_20261014.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"n_samples": out["n_samples"], "valid": out["common_valid_pixels"],
                      "clp_counts": out["clp_counts"], "metrics": scopes,
                      "cross_table": cross}, indent=2))


if __name__ == "__main__":
    main()
