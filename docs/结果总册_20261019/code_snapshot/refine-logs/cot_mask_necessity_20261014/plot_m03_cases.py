import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOG = ROOT / "refine-logs/cot_value_first_20261013"
PACK = ROOT / "experiments/monthly_R_station16_20261013_v1/results/ghi_head_pack_new_r_v1"
manifest = json.loads((HERE / "M03_case_manifest_20261014.json").read_text(encoding="utf-8"))
pred = np.load(LOG / "validation_predictions_monthly_R_seed42.npz")
refmeta = np.load(LOG / "clp_reference_monthly_R_seed42.npz")
pack_rows = list(csv.DictReader((PACK / "rows.csv").open(encoding="utf-8")))
idx = {(r["station"], r["target_time_utc"]): i for i, r in enumerate(pack_rows) if r["split"] == "val"}
clp_idx = {(s, t): i for i, (s, t) in enumerate(zip(refmeta["station"].astype(str), refmeta["target_time_utc"].astype(str)))}

loaded = []
for c in manifest["cases"]:
    key = (c["station"], c["time"])
    if key not in idx or key not in clp_idx or c["split"] != "val":
        raise ValueError(f"unaligned/non-validation case: {key}")
    k, j = idx[key], clp_idx[key]
    if int(pred["sample_indices"][j]) < 0 or not np.array_equal(pred["mask"][j], refmeta["common_mask"][j]):
        raise ValueError(f"support mismatch for {key}")
    frame = HERE / "agri_case_frames" / Path(c["agri_frame_relpath"]).name
    raw = np.load(frame, mmap_mode="r", allow_pickle=False)
    if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
        raise ValueError(f"unexpected AGRI frame {frame}: {raw.shape} {raw.dtype}")
    row, col = map(int, c["station_grid_rc"].split(","))
    r0, c0 = row - 8, col - 8
    rgb = np.stack([np.asarray(raw[ch, r0:r0+16, c0:c0+16]) for ch in (1, 2, 0)], axis=-1)
    loaded.append((c, j, rgb, raw.shape, frame))

# A single stretch across all selected patches preserves relative brightness.
allrgb = np.concatenate([x[2].reshape(-1, 3) for x in loaded], axis=0)
lo, hi = np.quantile(allrgb, [0.01, 0.99], axis=0)
rgb_images = [np.clip((x[2]-lo)/(hi-lo+1e-8), 0, 1) for x in loaded]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(len(loaded), 5, figsize=(15, 2.55*len(loaded)), constrained_layout=True)
fig.suptitle("M03 validation cases: AGRI appearance, CPP cloud reference, and frozen-R COT (station-centered 16×16)", fontsize=15)
im_cot = im_err = None
for n, ((case, j, rgb, shape, frame), axrow, rgb_show) in enumerate(zip(loaded, axes, rgb_images)):
    y = pred["reference_cot"][j, 0].astype(np.float64)
    yhat = np.expm1(pred["pred_log1p_cot"][j, 0].astype(np.float64))
    clp = refmeta["clp"][j, 0]
    valid = pred["mask"][j, 0] & refmeta["common_mask"][j, 0]
    y = np.where(valid, y, np.nan); yhat = np.where(valid, yhat, np.nan)
    err = yhat - y
    panels = [(rgb_show, "AGRI synthetic RGB\nR=C02, G=C03, B=C01", None),
              (np.ma.masked_where(~refmeta["clp_valid"][j, 0], clp), "CPP CLP\n0 clear, 1/2 cloudy", "clp"),
              (y, "CPP COT reference\nretrieval reference", "cot"),
              (yhat, "Frozen monthly-R COT\nlog1p output → COT", "cot"),
              (err, "R − CPP COT\nphysical-unit error", "err")]
    for ax, (arr, title, typ) in zip(axrow, panels):
        if typ == "clp":
            cm = plt.get_cmap("viridis", 4).copy(); cm.set_bad("#d9d9d9")
            im = ax.imshow(arr, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
        elif typ == "cot":
            im = ax.imshow(arr, cmap="magma", vmin=0, vmax=60, interpolation="nearest"); im_cot = im
        elif typ == "err":
            im = ax.imshow(arr, cmap="coolwarm", vmin=-60, vmax=60, interpolation="nearest"); im_err = im
        else:
            ax.imshow(arr, interpolation="nearest")
        ax.scatter([8], [8], marker="+", s=90, linewidths=1.6, c="cyan")
        ax.set_xticks([0, 4, 8, 12, 15]); ax.set_yticks([0, 4, 8, 12, 15])
        ax.set_title(title)
        if ax is axrow[0]:
            center = np.s_[7:10, 7:10]
            ax.set_ylabel(f"{case['station'].upper()} · {case['kind'].replace('_',' ')}\n{case['target_time_bjt']} BJT\n3×3 CPP/R mean: {np.nanmean(y[center]):.1f}/{np.nanmean(yhat[center]):.1f}\nkey: {case['time']} UTC")
    axrow[0].set_xlabel("col (station at 8)"); axrow[0].set_ylabel(axrow[0].get_ylabel()+"\nrow (station at 8)")
fig.colorbar(im_cot, ax=list(axes[:, 2]) + list(axes[:, 3]), shrink=.72, label="COT")
fig.colorbar(im_err, ax=list(axes[:, 4]), shrink=.72, label="COT error")
out = HERE / "M03_validation_case_panels_20261014.png"
fig.savefig(out, dpi=190, facecolor="white")
plt.close(fig)
audit = {"state": "PASS_ALIGNMENT_AND_PLOT", "cases": len(loaded), "rows": [
    {"station": c["station"], "case": c["kind"], "target_time_utc": c["time"], "target_time_bjt": c["target_time_bjt"],
     "key_exact": True, "split": "val", "agri_frame": str(f), "agri_shape": list(shape),
     "agri_source_remote": "/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/data/" + c["agri_frame_relpath"],
     "agri_frame_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
     "common_valid_pixels_16x16": int((pred["mask"][j,0] & refmeta["common_mask"][j,0]).sum()),
     "cot_mae_16x16": float(np.nanmean(np.abs(np.expm1(pred["pred_log1p_cot"][j,0].astype(np.float64))-pred["reference_cot"][j,0].astype(np.float64)))),
     "cot_reference_mean_center3x3": float(np.nanmean(pred["reference_cot"][j,0,7:10,7:10])),
     "cot_R_mean_center3x3": float(np.nanmean(np.expm1(pred["pred_log1p_cot"][j,0,7:10,7:10].astype(np.float64)))),
     "cot_mae_center3x3": float(np.nanmean(np.abs(np.expm1(pred["pred_log1p_cot"][j,0,7:10,7:10].astype(np.float64))-pred["reference_cot"][j,0,7:10,7:10].astype(np.float64))))}
    for c,j,rgb,shape,f in loaded], "rgb_channels": {"R": "AGRI channel index 1 / C02", "G": "index 2 / C03", "B": "index 0 / C01"},
    "rgb_stretch": "one global 1st–99th percentile per channel across selected patches", "station_pixel": [8,8],
    "test_payload_opened": False, "independent_truth_claim": False}
(HERE / "M03_case_alignment_audit_20261014.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
print(out)
