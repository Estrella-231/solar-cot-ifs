from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="AL normreload experiment root")
    p.add_argument("--pack", type=Path, required=True, help="frozen train/validation pack")
    p.add_argument("--feature-bank", type=Path, required=True, help="frozen-R feature bank")
    p.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.scripts))
    from hubei_ghi_head_al import HubeiLocalGHIHeadAL

    torch.set_num_threads(1)
    device = torch.device("cpu")
    pack_report_path = a.pack / "pack_report.json"
    feature_report_path = a.feature_bank / "AL_FEATURE_REPORT.json"
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if pack_report.get("test_and_buffer_values_read") != 0 or feature_report.get("test_payloads_read") != 0:
        raise AssertionError("pack/feature report does not keep test and buffer closed")
    if pack_report.get("split_sha256") != feature_report.get("split_sha256"):
        raise AssertionError("pack and feature bank split mismatch")
    pack_sha = sha256(pack_report_path)
    feature_sha = sha256(feature_report_path)
    if feature_report.get("pack_report_sha256") != pack_sha:
        raise AssertionError("feature bank is not bound to supplied pack")
    feature_array = a.feature_bank / "r_dec1_station_3x3.npy"
    feature_rows_path = a.feature_bank / "rows.csv"
    if sha256(feature_array) != feature_report.get("feature_file_sha256"):
        raise AssertionError("R-feature array hash mismatch")
    if sha256(feature_rows_path) != feature_report.get("rows_sha256"):
        raise AssertionError("R-feature key file hash mismatch")

    rows = read_csv(a.pack / "rows.csv")
    frows = read_csv(feature_rows_path)
    if sha256(a.pack / "rows.csv") != pack_report.get("rows_sha256"):
        raise AssertionError("pack rows hash mismatch")
    if any(r["split"] not in ("train", "val") for r in rows) or any(r["split"] not in ("train", "val") for r in frows):
        raise AssertionError("non-train/validation row present")
    feature_index = {(r["time_bjt"], r["split"]): i for i, r in enumerate(frows)}
    if len(feature_index) != len(frows):
        raise AssertionError("duplicate feature key")

    agri = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geom = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    ghi = np.load(a.pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(a.pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    r_all = np.load(feature_array, mmap_mode="r", allow_pickle=False)
    nrows = len(rows)
    if agri.shape != (nrows, 13, 3, 3) or geom.shape != (nrows, 2, 3, 3) or r_all.shape != (len(frows), 3, 16, 3, 3):
        raise AssertionError("input array shape mismatch")

    result = {"state": "HUBEI_AL_COLDSTART_NORMRELOAD_AUDIT_PASS_TEST_CLOSED",
              "scope": "existing AL checkpoints; train-only normalization recomputed from raw train rows; CPU validation replay",
              "pack_report_sha256": pack_sha, "feature_report_sha256": feature_sha,
              "split_sha256": pack_report["split_sha256"], "test_payloads_read": 0,
              "buffer_payloads_read": 0, "stations": {}, "seeds": [42, 43, 44]}
    for seed in result["seeds"]:
        run = a.root / "runs_maskfix_v2" / f"seed{seed}"
        run_meta_path = run / "run_metadata.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta.get("state") != "HUBEI_GHI_HEAD_AL_SEED_COMPLETE" or run_meta.get("test_payloads_read") != 0:
            raise AssertionError(f"invalid AL run metadata for seed {seed}")
        if run_meta.get("pack_report_sha256") != pack_sha or run_meta.get("feature_report_sha256") != feature_sha:
            raise AssertionError(f"AL run provenance mismatch for seed {seed}")
        if run_meta.get("split_sha256") != pack_report["split_sha256"]:
            raise AssertionError(f"AL run split mismatch for seed {seed}")
        norm_path = run / "normalization_train_only.json"
        norms = json.loads(norm_path.read_text(encoding="utf-8"))
        seed_result = {"run_metadata_sha256": sha256(run_meta_path), "normalization_sha256": sha256(norm_path), "stations": {}}
        for station in STATIONS:
            station_idx = STATIONS.index(station)
            idx = np.asarray([i for i, r in enumerate(rows) if r["station"] == station], dtype=np.int64)
            train_local = np.asarray([j for j, i in enumerate(idx) if rows[i]["split"] == "train"], dtype=np.int64)
            val_local = np.asarray([j for j, i in enumerate(idx) if rows[i]["split"] == "val"], dtype=np.int64)
            if not len(train_local) or not len(val_local):
                raise AssertionError(f"missing train/val rows for {station}")
            raw_x = np.asarray(agri[idx], dtype=np.float32)
            raw_geom = np.asarray(geom[idx], dtype=np.float32)
            raw_r = np.empty((len(idx), 16, 3, 3), dtype=np.float32)
            for j, row_i in enumerate(idx):
                row = rows[row_i]
                key = (row["time_bjt"], row["split"])
                if key not in feature_index:
                    raise AssertionError(f"missing R feature key: {station}/{key}")
                raw_r[j] = r_all[feature_index[key], station_idx]
            if not np.isfinite(raw_x).all() or not np.isfinite(raw_r).all() or not np.isfinite(raw_geom).all():
                raise AssertionError(f"nonfinite train/val model input for {station}")

            train_x = raw_x[train_local].astype(np.float64)
            train_r = raw_r[train_local].astype(np.float64)
            expected = {
                "agri_mean": train_x.mean(axis=(0, 2, 3)),
                "agri_std": np.maximum(train_x.std(axis=(0, 2, 3)), 1e-6),
                "r_mean": train_r.mean(axis=(0, 2, 3)),
                "r_std": np.maximum(train_r.std(axis=(0, 2, 3)), 1e-6),
            }
            saved_norm = norms[station]
            max_norm_delta = 0.0
            for key, value in expected.items():
                delta = float(np.max(np.abs(np.asarray(saved_norm[key], dtype=np.float64) - value)))
                max_norm_delta = max(max_norm_delta, delta)
                if delta > 1e-12:
                    raise AssertionError(f"saved {key} differs from train-only recomputation: {station}/{seed}/{delta}")
            if int(saved_norm["fit_rows"]) != len(train_local):
                raise AssertionError(f"normalization train-row count mismatch for {station}/{seed}")

            x = (raw_x - np.asarray(saved_norm["agri_mean"], dtype=np.float32)[None, :, None, None]) / np.asarray(saved_norm["agri_std"], dtype=np.float32)[None, :, None, None]
            x = np.concatenate((x, raw_geom), axis=1)
            rr = (raw_r - np.asarray(saved_norm["r_mean"], dtype=np.float32)[None, :, None, None]) / np.asarray(saved_norm["r_std"], dtype=np.float32)[None, :, None, None]
            x_t = torch.from_numpy(np.ascontiguousarray(x[val_local])).to(device)
            r_t = torch.from_numpy(np.ascontiguousarray(rr[val_local])).to(device)
            c_t = torch.as_tensor(np.asarray(clear[idx[val_local]], dtype=np.float32), device=device)
            y = np.asarray(ghi[idx[val_local]], dtype=np.float64)
            val_rows = [rows[idx[j]] for j in val_local]
            checkpoint = run / station / "AL_best.pt"
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            if saved.get("station") != station or saved.get("seed") != seed or saved.get("arm") != "AL":
                raise AssertionError(f"AL checkpoint identity mismatch: {station}/{seed}")
            model = HubeiLocalGHIHeadAL().to(device).eval()
            model.load_state_dict(saved["model"], strict=True)
            with torch.inference_mode():
                pred = (model(x_t, r_t) * c_t).cpu().numpy().astype(np.float64)
            bundle_path = run / station / "AL_validation_predictions_with_keys.npz"
            with np.load(bundle_path, allow_pickle=False) as z:
                bstation = z["station"].astype(str)
                btime = z["target_time_bjt"].astype(str)
                bp = z["pred_ghi_wm2"].astype(np.float64)
                by = z["truth_ghi_wm2"].astype(np.float64)
            etime = np.asarray([r["time_bjt"] for r in val_rows], dtype=str)
            if not np.all(bstation == station) or not np.array_equal(btime, etime):
                raise AssertionError(f"checkpoint validation keys differ: {station}/{seed}")
            max_pred_diff = float(np.max(np.abs(pred - bp)))
            err = pred - y
            coldstart_rmse = float(np.sqrt(np.mean(err * err)))
            reported = json.loads((run / station / "AL_validation.json").read_text(encoding="utf-8"))
            rmse_delta = abs(coldstart_rmse - float(reported["rmse_wm2"]))
            if not np.array_equal(by, y) or max_pred_diff > 0.5 or rmse_delta > 0.03:
                raise AssertionError(
                    f"checkpoint cold-start mismatch: {station}/{seed}, "
                    f"maxdiff={max_pred_diff}, rmse={coldstart_rmse}, "
                    f"reported={reported['rmse_wm2']}, rmse_delta={rmse_delta}"
                )
            seed_result["stations"][station] = {
                "train_rows": int(len(train_local)), "validation_rows": int(len(val_local)),
                "normalization_max_abs_delta": max_norm_delta,
                "checkpoint_sha256": sha256(checkpoint), "bundle_sha256": sha256(bundle_path),
                "coldstart_max_prediction_delta_wm2": max_pred_diff,
                "coldstart_rmse_wm2": coldstart_rmse,
                "reported_rmse_wm2": float(reported["rmse_wm2"]),
                "rmse_abs_delta_wm2": rmse_delta,
                "test_payloads_read": 0,
            }
            del model, saved, x_t, r_t, c_t
        result["stations"][str(seed)] = seed_result
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
