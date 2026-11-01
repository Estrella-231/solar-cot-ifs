"""Pure NumPy metrics and train-OOF-only conservative gate selection.

Inputs share shape (samples, ...); stations has shape (samples,).  This module
does no I/O.  The caller must establish OOF provenance and pass split='train_oof'
to selection.  Evaluation may use another explicitly named split.

Cloud detection is C_gate >= 0.1, not the classifier decision by itself.
The gate zeros COT only when p_cloud < threshold.  Thresholds are in [0, 1].
"""
import numpy as np


STATIONS = ("sili", "zhujia")
CLOUD_COT = 0.1
THICK_COT = 30.0
_EPS = 1e-12


def _prepare(prob_cloud, raw_cot, reference_cot, clp, valid, stations):
    arrays = [np.asarray(x) for x in (prob_cloud, raw_cot, reference_cot, clp, valid)]
    shape = arrays[0].shape
    if not shape or any(a.shape != shape for a in arrays):
        raise ValueError("probability, COT, CLP, and valid arrays must share shape (samples, ...)")
    station = np.asarray(stations).astype(str)
    if station.shape != (shape[0],) or not np.isin(station, STATIONS).all():
        raise ValueError("stations must contain one sili/zhujia name per sample")
    mask = arrays[4].astype(bool, copy=False)
    p, raw, ref = [np.asarray(a[mask], dtype=np.float64) for a in arrays[:3]]
    labels = arrays[3][mask]
    if not all(np.isfinite(a).all() for a in (p, raw, ref)):
        raise ValueError("nonfinite probability/COT under common valid mask")
    if np.any((p < 0) | (p > 1)) or np.any(raw < 0) or np.any(ref < 0):
        raise ValueError("valid probabilities must be in [0,1] and physical COT nonnegative")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("common valid mask contains invalid CPP CLP")
    codes = (station == "zhujia").astype(np.uint8)
    codes = np.broadcast_to(codes.reshape((shape[0],) + (1,) * (len(shape) - 1)), shape)[mask]
    return p, raw, ref, labels.astype(np.uint8), codes


def _class_scores(pred, target):
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(np.count_nonzero(~pred & ~target))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "POD": tp / (tp + fn) if tp + fn else None,
            "FAR": fp / (tp + fp) if tp + fp else None,
            "CSI": tp / (tp + fp + fn) if tp + fp + fn else None}


def _cot_scores(pred, reference, labels):
    if not pred.size:
        return {"COT_RMSE": None, "SSE": 0.0,
                "SSE_by_CLP": {"clear": 0.0, "cloud": 0.0}}
    error = pred - reference
    squared = error * error
    return {"COT_RMSE": float(np.sqrt(np.mean(squared))), "SSE": float(squared.sum()),
            "SSE_by_CLP": {"clear": float(squared[labels == 0].sum()),
                           "cloud": float(squared[labels > 0].sum())}}


def _limits(max_cloud_pod_drop, max_thick_pod_drop, max_thick_csi_drop):
    values = (max_cloud_pod_drop, max_thick_pod_drop, max_thick_csi_drop)
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("recall/CSI loss limits must be finite fractions in [0,1]")
    return values


def _evaluate(prepared, threshold, split, limits):
    p, raw, ref, labels, codes = prepared
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability in [0,1]")
    by_station = {}
    for code, station in enumerate(STATIONS):
        sel = codes == code
        r, y, lab = raw[sel], ref[sel], labels[sel]
        gated = np.where(p[sel] < threshold, 0.0, r)
        target_cloud, target_thick = lab > 0, y >= THICK_COT
        arms = {}
        for name, prediction in (("raw", r), ("gated", gated)):
            arms[name] = {"cloud": _class_scores(prediction >= CLOUD_COT, target_cloud),
                          "thick": _class_scores(prediction >= THICK_COT, target_thick),
                          **_cot_scores(prediction, y, lab)}
        support = bool(r.size and target_cloud.any() and target_thick.any())
        # Undefined FAR / empty predicted class cannot silently pass a guard.
        support = support and all(arms[a][kind]["TP"] + arms[a][kind]["FP"] > 0
                                  for a in ("raw", "gated") for kind in ("cloud", "thick"))
        deltas = {}
        checks = {"support": support}
        for name, kind, metric, limit in (
                ("cloud_POD", "cloud", "POD", limits[0]),
                ("thick_POD", "thick", "POD", limits[1]),
                ("thick_CSI", "thick", "CSI", limits[2])):
            before, after = arms["raw"][kind][metric], arms["gated"][kind][metric]
            delta = None if before is None or after is None else after - before
            deltas[name] = delta
            checks[name + "_protected"] = bool(support and delta is not None and delta >= -limit - _EPS)
        by_station[station] = {"valid_pixels": int(r.size),
                               "reference_cloud_pixels": int(target_cloud.sum()),
                               "reference_thick_pixels": int(target_thick.sum()),
                               **arms, "delta_gated_minus_raw": deltas,
                               "guards": checks, "guards_pass": all(checks.values())}
    equal = {}
    for arm in ("raw", "gated"):
        rmses = [by_station[s][arm]["COT_RMSE"] for s in STATIONS]
        fars = [by_station[s][arm]["cloud"]["FAR"] for s in STATIONS]
        equal["COT_RMSE_" + arm] = None if any(x is None for x in rmses) else float(np.mean(rmses))
        equal["cloud_FAR_" + arm] = None if any(x is None for x in fars) else float(np.mean(fars))
    equal["delta_COT_RMSE"] = (None if equal["COT_RMSE_raw"] is None or equal["COT_RMSE_gated"] is None
                               else equal["COT_RMSE_gated"] - equal["COT_RMSE_raw"])
    return {"state": "EVALUATED", "split": split, "threshold": float(threshold),
            "cloud_cot_threshold": CLOUD_COT, "thick_cot_threshold": THICK_COT,
            "by_station": by_station, "station_equal_mean": equal,
            "guards_pass": all(by_station[s]["guards_pass"] for s in STATIONS)}


def evaluate_threshold(prob_cloud, raw_cot, reference_cot, clp, valid, stations,
                       threshold, *, split,
                       max_cloud_pod_drop=0.01, max_thick_pod_drop=0.01,
                       max_thick_csi_drop=0.01):
    """Evaluate a preselected threshold; no threshold fitting occurs here."""
    limits = _limits(max_cloud_pod_drop, max_thick_pod_drop, max_thick_csi_drop)
    return _evaluate(_prepare(prob_cloud, raw_cot, reference_cot, clp, valid, stations),
                     threshold, split, limits)


def select_threshold(prob_cloud, raw_cot, reference_cot, clp, valid, stations,
                     *, split="train_oof", max_cloud_pod_drop=0.01,
                     max_thick_pod_drop=0.01, max_thick_csi_drop=0.01):
    """Select the largest feasible threshold using only declared train OOF.

    Sort once and accumulate removal counts: O(N log N) time, O(N) memory.
    A candidate t=p retains every pixel with probability exactly p.  Every
    achievable zero-only gate has a representative among unique p values plus
    1.0; these representatives are the largest thresholds on their plateaus.
    No validation metric influences this selection.  Empty class support fails
    closed; no NaN comparison can make an unsupported threshold feasible.
    """
    if split != "train_oof":
        raise ValueError("threshold selection requires split='train_oof'; validation/test fitting forbidden")
    limits = _limits(max_cloud_pod_drop, max_thick_pod_drop, max_thick_csi_drop)
    prepared = _prepare(prob_cloud, raw_cot, reference_cot, clp, valid, stations)
    p, raw, ref, labels, codes = prepared
    baseline = _evaluate(prepared, 0.0, split, limits)
    result = {"state": "FAIL_CLOSED", "threshold": None, "selection_split": split,
              "thresholds_evaluated": 0, "feasible_thresholds": 0,
              "max_cloud_pod_drop": float(limits[0]), "max_thick_pod_drop": float(limits[1]),
              "max_thick_csi_drop": float(limits[2]),
              "baseline_metrics": baseline, "selected_metrics": None, "reasons": []}
    for station in STATIONS:
        if not baseline["by_station"][station]["guards"]["support"]:
            result["reasons"].append(station + ": missing valid/reference/predicted cloud or thick support")
    if result["reasons"]:
        return result
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    candidates, cut = np.unique(sorted_p, return_index=True)
    if candidates[-1] < 1.0:
        candidates = np.append(candidates, 1.0)
        cut = np.append(cut, len(p))
    # cut is exactly count(p < candidate), including ties as a whole.
    feasible = np.ones(len(candidates), dtype=bool)
    cloud_target, thick_target = labels > 0, ref >= THICK_COT
    cloud_pred, thick_pred = raw >= CLOUD_COT, raw >= THICK_COT

    def removed(event):
        prefix = np.empty(len(p) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(event[order], dtype=np.int64, out=prefix[1:])
        return prefix[cut]

    for code, station in enumerate(STATIONS):
        sel = codes == code
        before = baseline["by_station"][station]["raw"]
        c, thick = before["cloud"], before["thick"]
        ctp = c["TP"] - removed(sel & cloud_pred & cloud_target)
        cpred = c["TP"] + c["FP"] - removed(sel & cloud_pred)
        ttp = thick["TP"] - removed(sel & thick_pred & thick_target)
        tpred = thick["TP"] + thick["FP"] - removed(sel & thick_pred)
        cpod = ctp / (c["TP"] + c["FN"])
        tpod = ttp / (thick["TP"] + thick["FN"])
        tcsi = ttp / (thick["TP"] + thick["FN"] + tpred - ttp)
        feasible &= ((cpred > 0) & (tpred > 0)
                     & (cpod >= c["POD"] - limits[0] - _EPS)
                     & (tpod >= thick["POD"] - limits[1] - _EPS)
                     & (tcsi >= thick["CSI"] - limits[2] - _EPS))
    result["thresholds_evaluated"] = int(len(candidates))
    result["feasible_thresholds"] = int(feasible.sum())
    locations = np.flatnonzero(feasible)
    if not len(locations):
        result["reasons"] = ["no probability-event threshold passes every station's support and recall guards"]
        return result
    threshold = float(candidates[locations[-1]])
    selected = _evaluate(prepared, threshold, split, limits)
    if not selected["guards_pass"]:
        raise RuntimeError("cumulative selector disagrees with direct pixelwise evaluation")
    result.update(state="SELECTED", threshold=threshold, selected_metrics=selected)
    return result
