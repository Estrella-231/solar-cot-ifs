import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "refine-logs/cot_value_first_20261013"
PACK = ROOT / "experiments/monthly_R_station16_20261013_v1/results/ghi_head_pack_new_r_v1"
CAND = ROOT / "experiments/monthly_R_station16_20261013_v1/input/target_candidates_train_val.csv"
pred = np.load(LOG / "validation_predictions_monthly_R_seed42.npz")
clp = np.load(LOG / "clp_reference_monthly_R_seed42.npz")
rows = list(csv.DictReader((PACK / "rows.csv").open(encoding="utf-8")))
candidates = list(csv.DictReader(CAND.open(encoding="utf-8-sig")))
pack_idx = {(r["station"], r["target_time_utc"]): i for i, r in enumerate(rows) if r["split"] == "val"}
cand_idx = {(r["station"], r["target_time_utc"]): r for r in candidates if r["split"] == "val"}
geom = np.load(PACK / "geometry.npy", mmap_mode="r")
records = {s: [] for s in ("sili", "zhujia")}

for j, (s, t) in enumerate(zip(clp["station"].astype(str), clp["target_time_utc"].astype(str))):
    if (s, t) not in pack_idx or (s, t) not in cand_idx:
        continue
    k = pack_idx[(s, t)]
    p = np.expm1(pred["pred_log1p_cot"][j, 0].astype(np.float64))
    y = pred["reference_cot"][j, 0].astype(np.float64)
    c = clp["clp"][j, 0]
    v = pred["mask"][j, 0] & clp["common_mask"][j, 0]
    ps, ys, cs, vs = p[7:10, 7:10], y[7:10, 7:10], c[7:10, 7:10], v[7:10, 7:10]
    if vs.sum() < 8:
        continue
    clear = vs & (cs == 0)
    cloud = vs & np.isin(cs, [1, 2])
    if clear.any():
        records[s].append({"kind": "clear_false_alarm", "station": s, "time": t, "pack_index": k, "r_index": j,
            "score": float(ps[clear].mean()), "pred_center_mean": float(ps[vs].mean()), "ref_center_mean": float(ys[vs].mean()),
            "clear_fraction": float(clear.sum()/vs.sum()), "cosSOZ": float(np.nanmean(geom[k, 0]))})
    thick = cloud & (ys >= 30)
    if thick.any():
        pos = np.argwhere(thick)
        a, b = pos[np.argmax(ys[thick])]
        records[s].append({"kind": "thick_cloud_miss", "station": s, "time": t, "pack_index": k, "r_index": j,
            "score": float(ps[a,b]), "pred_at_ref_peak": float(ps[a,b]), "ref_peak": float(ys[a,b]),
            "pred_center_mean": float(ps[vs].mean()), "ref_center_mean": float(ys[vs].mean()),
            "cloud_fraction": float(cloud.sum()/vs.sum()), "peak_offset_rc": [int(a-1), int(b-1)],
            "cosSOZ": float(np.nanmean(geom[k, 0]))})

chosen = []
for s, items in records.items():
    fa = min((x for x in items if x["kind"] == "clear_false_alarm"), key=lambda x: (-x["score"], x["time"]))
    miss = min((x for x in items if x["kind"] == "thick_cloud_miss"), key=lambda x: (x["score"], x["time"]))
    chosen.extend([fa, miss])
    controls = []
    for x in items:
        if x["kind"] != "clear_false_alarm" or x["time"] == fa["time"]:
            continue
        err = abs(x["pred_center_mean"] - x["ref_center_mean"])
        if x["clear_fraction"] == 1.0 and x["score"] < 2 and x["pred_center_mean"] < 3 and err < 2:
            controls.append((abs(x["cosSOZ"]-fa["cosSOZ"]), err, x["time"], x))
    if controls:
        control = min(controls)[-1].copy()
        control["kind"] = "clear_correct_control"
        control["so_abs_diff_from_false_alarm"] = min(controls)[0]
        control["score"] = min(controls)[1]
        chosen.append(control)

for x in chosen:
    t = x["time"]
    x.update({"target_time_bjt": (dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
              "split": "val", "agri_frame_relpath": cand_idx[(x["station"], t)]["agri_frame_relpath"],
              "station_grid_rc": cand_idx[(x["station"], t)]["station_grid_rc"]})
out = Path(__file__).with_name("M03_case_manifest_20261014.json")
out.write_text(json.dumps({"selection_rule": "Per station: highest 3x3 CLP-clear predicted-COT mean; lowest prediction at the max-reference-COT>=30 pixel in the 3x3 cloud subset; control is a clear case with center mean absolute error<2, predicted mean<3, nearest cosSOZ to the false alarm. Exact (station,target_time_utc), validation only.", "cases": chosen, "test_payload_opened": False}, indent=2) + "\n", encoding="utf-8")
print(f"cases={len(chosen)} -> {out}")
for x in chosen:
    print(x["station"], x["kind"], x["time"], "score", round(x["score"], 3), "ref", round(x.get("ref_peak", x["ref_center_mean"]), 3), "cosSOZ", round(x["cosSOZ"], 5))
