"""Paired train/val-only regional R diagnostics from saved predictions.

CPP is a retrieval reference. This script never reads test or trains a model.
Saved validation order is reconstructed using the exact PairDataset rule.
"""
import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


BINS = (("near_zero", 0, .1), ("thin", .1, 3), ("moderate", 3, 10),
        ("thick", 10, 30), ("very_thick", 30, 60), ("extreme", 60, 101))
STATIONS = {"sili": (163, 148), "zhujia": (73, 150)}


def unpack(run, work):
    target = work / run.name
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(run / "validation_predictions.npz") as archive:
        for name in ("pred_log1p_cot.npy", "reference_log1p_cot.npy", "valid_mask.npy"):
            dst = target / name
            if not dst.exists():
                with archive.open(name) as src, dst.with_suffix(".tmp").open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                dst.with_suffix(".tmp").replace(dst)
    return tuple(np.load(target / name, mmap_mode="r", allow_pickle=False)
                 for name in ("pred_log1p_cot.npy", "reference_log1p_cot.npy", "valid_mask.npy"))


def accumulate(store, key, prediction, reference, mask):
    error = (prediction - reference)[mask].astype(np.float64)
    if not error.size:
        return
    row = store[key]
    row += (error.size, np.abs(error).sum(), np.square(error).sum(), error.sum())


def finalize(store):
    return {key: {"n_pixels": int(v[0]), "mae": float(v[1]/v[0]),
                  "rmse": float(np.sqrt(v[2]/v[0])), "bias": float(v[3]/v[0])}
            for key, v in sorted(store.items()) if v[0]}


def rgb(raw, valid, limits):
    values = raw[[2, 1, 0]].transpose(1, 2, 0).copy()
    for ch, (low, high) in enumerate(limits):
        values[:, :, ch] = np.clip((values[:, :, ch] - low)/max(high-low, 1e-6), 0, 1)
    values[~valid] = .45
    return np.nan_to_num(values, nan=.45)


def main():
    p = argparse.ArgumentParser()
    for name in ("cache", "baseline", "candidate", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite diagnostic output")
    for run in (a.baseline, a.candidate):
        status = json.loads((run / "status.json").read_text())
        assert status["state"] == "COMPLETE_EXPLORATORY_REGIONAL_R" and status["test_used"] is False
    meta0, meta1 = [json.loads((run / "run_metadata.json").read_text())
                    for run in (a.baseline, a.candidate)]
    assert meta0["normalization_sha256"] == meta1["normalization_sha256"]
    assert meta0["batch"] == 32 and meta1["batch"] == 512
    for key in ("seed", "r_arch_sha256", "loss", "optimizer", "train_frames", "validation_frames"):
        assert meta0[key] == meta1[key], key
    index = json.loads((a.cache / "index.json").read_text())
    assert index["test_used"] is False and index["state"] == "COMPLETE_EXPLORATORY_REAL_AGRI_CPP_CACHE"
    a.output.mkdir(parents=True)
    work = a.output / "extracted_arrays"
    b, bt, bm = unpack(a.baseline, work)
    c, ct, cm = unpack(a.candidate, work)
    records = []
    for shard in index["shards"]:
        directory = a.cache / shard["directory"]
        for rec in json.loads((directory / "records.json").read_text()):
            if rec["split"] == "val" and rec["included_for_R"]:
                records.append({**rec, "directory": str(directory)})
    assert b.shape == c.shape == bt.shape == ct.shape == bm.shape == cm.shape == (len(records), 1, 256, 256)
    stores = [defaultdict(lambda: np.zeros(4, np.float64)) for _ in range(2)]
    station_rows = defaultdict(list)
    daily = defaultdict(lambda: np.zeros(3, np.float64))
    edge = np.zeros((2, 3), np.float64)
    for i, rec in enumerate(records):
        assert np.array_equal(bm[i], cm[i]) and np.array_equal(bt[i], ct[i])
        valid = np.asarray(bm[i, 0])
        truth = np.expm1(np.asarray(bt[i, 0]))
        predictions = [np.expm1(np.asarray(z[i, 0])) for z in (b, c)]
        d = Path(rec["directory"])
        row = rec["array_row"]
        ref = np.load(d / "cot.npy", mmap_mode="r")[row]
        cached_mask = np.load(d / "mask.npy", mmap_mode="r")[row]
        assert np.array_equal(valid, cached_mask)
        assert np.allclose(truth[valid], ref[valid], rtol=2e-6, atol=2e-5)
        raw = np.load(d / "x_raw.npy", mmap_mode="r")[row]
        altitude = np.rad2deg(np.arcsin(np.clip(raw[13], -1, 1)))
        bjt = datetime.strptime(rec["utc"], "%Y%m%d%H%M") + timedelta(hours=8)
        for arm, pred in enumerate(predictions):
            store = stores[arm]
            accumulate(store, "region/all", pred, truth, valid)
            accumulate(store, "month/"+bjt.strftime("%Y-%m"), pred, truth, valid)
            for name, low, high in BINS:
                accumulate(store, "cot/"+name, pred, truth, valid & (truth >= low) & (truth < high))
            for low, high in ((10, 20), (20, 40), (40, 60), (60, 91)):
                accumulate(store, f"solar_altitude/{low}_{high}", pred, truth,
                           valid & (altitude >= low) & (altitude < high))
            for name, (r, col) in STATIONS.items():
                sl = np.s_[r-2:r+3, col-2:col+3]
                accumulate(store, name+"/5x5", pred[sl], truth[sl], valid[sl])
                for group, low, high in BINS:
                    accumulate(store, name+"/cot/"+group, pred[sl], truth[sl],
                               valid[sl] & (truth[sl] >= low) & (truth[sl] < high))
            # Reference edge threshold is descriptive, not a physical cloud mask.
            for axis in (0, 1):
                gradient = np.diff(truth, axis=axis)
                pred_gradient = np.diff(pred, axis=axis)
                m = (valid[:-1, :] & valid[1:, :]) if axis == 0 else (valid[:, :-1] & valid[:, 1:])
                m &= np.abs(gradient) >= 3
                if m.any():
                    edge[arm] += (m.sum(), np.abs(pred_gradient[m]-gradient[m]).sum(dtype=np.float64),
                                  np.abs(pred_gradient[m]).sum(dtype=np.float64))
        err0, err1 = [(z-truth)[valid].astype(np.float64) for z in predictions]
        daily[bjt.strftime("%Y-%m-%d")] += (valid.sum(), np.square(err0).sum(), np.square(err1).sum())
        for name, (r, col) in STATIONS.items():
            sl = np.s_[r-2:r+3, col-2:col+3]
            if valid[sl].any():
                e = [float(np.sqrt(np.mean(np.square((z-truth)[sl][valid[sl]].astype(np.float64)))))
                     for z in predictions]
                station_rows[name].append({"index": i, "utc": rec["utc"], "bjt": str(bjt),
                    "rmse_baseline": e[0], "rmse_candidate": e[1], "reference_mean": float(truth[sl][valid[sl]].mean()),
                    "solar_altitude": float(altitude[r, col])})
        if i % 128 == 0:
            print("AUDIT", i, "/", len(records), flush=True)
    summary = {"state": "COMPLETE_PAIRED_CPP_REFERENCE_AUDIT", "test_used": False,
        "validation_frames": len(records), "baseline": str(a.baseline), "candidate": str(a.candidate),
        "baseline_metrics": finalize(stores[0]), "candidate_metrics": finalize(stores[1]),
        "edge_definition": "valid adjacent CPP pixels with absolute COT difference >=3; descriptive only",
        "edge_metrics": [{"n_pairs": int(v[0]), "gradient_mae": float(v[1]/v[0]),
                          "prediction_gradient_abs_mean": float(v[2]/v[0])} for v in edge],
        "note": "CPP retrieval reference; producer/physical QA unresolved; no GHI skill claim",
        "case_selection": "per station: worst candidate RMSE, largest improvement, largest worsening, median-RMSE reference case"}
    rng = np.random.default_rng(20260927)
    days = np.array(list(daily.values()))
    draws = rng.integers(0, len(days), (2000, len(days)))
    sums = days[draws].sum(axis=1)
    delta = np.sqrt(sums[:, 2]/sums[:, 0]) - np.sqrt(sums[:, 1]/sums[:, 0])
    summary["date_bootstrap"] = {"unit": "BJT validation day", "days": len(days), "repetitions": 2000,
        "seed": 20260927, "region_delta_RMSE_candidate_minus_baseline_95pct": np.quantile(delta, [.025, .975]).tolist()}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    stamps = {rec["utc"]: i for i, rec in enumerate(records)}
    selected = []
    for name, rows in station_rows.items():
        candidates = [("worst", max(rows, key=lambda x: x["rmse_candidate"])),
            ("improved", min(rows, key=lambda x: x["rmse_candidate"]-x["rmse_baseline"])),
            ("worsened", max(rows, key=lambda x: x["rmse_candidate"]-x["rmse_baseline"])),
            ("typical", sorted(rows, key=lambda x: x["rmse_candidate"])[len(rows)//2])]
        r, col = STATIONS[name]
        for kind, case in candidates:
            selected.append({"station": name, "kind": kind, **case})
            center = datetime.strptime(case["utc"], "%Y%m%d%H%M")
            ids = [stamps.get((center+timedelta(minutes=offset)).strftime("%Y%m%d%H%M")) for offset in (-15, 0, 15)]
            center_rec = records[case["index"]]
            raw_center = np.load(Path(center_rec["directory"])/"x_raw.npy", mmap_mode="r")[center_rec["array_row"]]
            limits = [np.nanpercentile(raw_center[ch][np.isfinite(raw_center[ch])], [2, 98]).tolist() for ch in (2, 1, 0)]
            sl = np.s_[r-32:r+32, col-32:col+32]
            fig, axes = plt.subplots(3, 5, figsize=(16, 9), layout="constrained")
            for j, idx in enumerate(ids):
                if idx is None:
                    for ax in axes[j]:
                        ax.text(.5, .5, "No paired valid frame", ha="center", transform=ax.transAxes)
                        ax.axis("off")
                    continue
                rec = records[idx]
                raw = np.load(Path(rec["directory"])/"x_raw.npy", mmap_mode="r")[rec["array_row"]]
                mask = np.asarray(bm[idx, 0])
                true = np.expm1(bt[idx, 0]); pb = np.expm1(b[idx, 0]); pc = np.expm1(c[idx, 0])
                fields = [rgb(raw, np.isfinite(raw[:13]).all(0), limits)[sl],
                          np.where(mask, true, np.nan)[sl], np.where(mask, pb, np.nan)[sl],
                          np.where(mask, pc, np.nan)[sl], np.where(mask, pc-true, np.nan)[sl]]
                for k, field in enumerate(fields):
                    ax = axes[j, k]
                    image = ax.imshow(field, origin="upper", interpolation="nearest", **({} if k == 0 else
                        {"cmap": "RdBu_r" if k == 4 else "magma", "vmin": -30 if k == 4 else 0,
                         "vmax": 30 if k == 4 else 100}))
                    ax.plot(32, 32, marker="+", color="#00bfff", markersize=10)
                    ax.set_title(["AGRI C03/C02/C01 proxy", "CPP retrieval reference", "R batch32", "R batch512", "batch512 minus CPP"][k])
                    ax.set_xlabel("local column (64-pixel crop)")
                    if k == 0:
                        ax.set_ylabel(rec["utc"]+" UTC\nlocal row")
                    if k in (3, 4):
                        fig.colorbar(image, ax=ax, shrink=.7, extend="both", label="COT (dimensionless)")
            fig.suptitle(f"{name} / {kind}: real AGRI retrieval diagnostic; center {case['bjt']} BJT\n"
                "+ marks the same station pixel; RGB uses center-frame channel 2-98% stretches shared across time; no smoothing", fontsize=11)
            fig.savefig(a.output/f"{name}_{kind}.png", dpi=160)
            fig.savefig(a.output/f"{name}_{kind}.pdf")
            plt.close(fig)
            np.savez_compressed(a.output/f"{name}_{kind}_source.npz", indices=np.array([-1 if v is None else v for v in ids]),
                                rgb_channel_limits=np.array(limits), station_rc=np.array([r, col]))
    (a.output/"selected_cases.json").write_text(json.dumps(selected, indent=2)+"\n")
    print("COMPLETE_PAIRED_CPP_REFERENCE_AUDIT", flush=True)


if __name__ == "__main__":
    main()
