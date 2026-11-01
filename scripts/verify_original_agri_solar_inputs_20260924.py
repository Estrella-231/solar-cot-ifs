"""Verify the packed validation AGRI channels and solar angles against originals."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "results/cot_staged_diagnostic_20260924/source_cases"
PACK = CASE_DIR / "pack_readonly"
SOURCE = CASE_DIR / "original_agri_20ch"
BOUNDS = {"sili": (155, 171, 140, 156), "zhujia": (65, 81, 142, 158)}
FEATURES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    temporal = np.load(CASE_DIR / "four_case_temporal_agri_model_inputs.npz")
    manifest = json.loads(str(temporal["manifest_json"].item()))
    rows = pd.read_csv(PACK / "rows.csv")
    pack_raw = np.load(PACK / "x_raw.npy", mmap_mode="r")
    norm = json.loads((PACK / "norm.json").read_text(encoding="utf-8"))
    row_pos = {int(row_id): i for i, row_id in enumerate(rows["index"].to_numpy())}
    crop_list = []
    frame_results = []
    for w in manifest:
        stamp = pd.Timestamp(w["timestamp_utc"]).strftime("%Y%m%d%H%M")
        path = SOURCE / f"{stamp}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        full = np.load(path, mmap_mode="r")
        if full.shape != (20, 256, 256):
            raise ValueError(f"unexpected AGRI shape {full.shape}: {path.name}")
        r0, r1, c0, c1 = BOUNDS[w["station_id"]]
        crop = np.asarray(full[:, r0:r1, c0:c1], dtype=np.float32)
        crop_list.append(crop)
        packed = np.asarray(pack_raw[row_pos[int(w["global_row_index"])]])
        packed_agri = crop[FEATURES]
        expected_cotsoz = np.cos(np.deg2rad(crop[18])).astype(np.float32)
        expected_cosraa = np.cos(np.deg2rad(crop[17] - crop[15])).astype(np.float32)
        expected_day = (crop[19] > 0.5).astype(np.float32)
        comparisons = {
            "13_raw_imagery_channels_max_abs_error": float(np.max(np.abs(packed[:13] - packed_agri))),
            "cosSOZ_max_abs_error": float(np.max(np.abs(packed[13] - expected_cotsoz))),
            "cosRAA_max_abs_error": float(np.max(np.abs(packed[14] - expected_cosraa))),
            "day_mask_mismatch_pixels": int(np.count_nonzero(packed[15] != expected_day)),
            "input_day_mask_unique": [float(x) for x in np.unique(packed[15])],
            "source_SOZ_degrees_range": [float(np.nanmin(crop[18])), float(np.nanmax(crop[18]))],
            "source_SOA_degrees_range": [float(np.nanmin(crop[17])), float(np.nanmax(crop[17]))],
            "source_SAA_degrees_range": [float(np.nanmin(crop[15])), float(np.nanmax(crop[15]))],
            "source_C13_BT_K_range": [float(np.nanmin(crop[12])), float(np.nanmax(crop[12]))],
            "original_file_sha256": sha(path),
            "original_file": path.name,
        }
        frame_results.append({**w, **comparisons})

    crops = np.stack(crop_list)
    np.savez_compressed(CASE_DIR / "four_case_temporal_original_agri_crops.npz",
                        original_agri=crops, manifest_json=json.dumps(manifest, ensure_ascii=False))
    errors13 = np.asarray([x["13_raw_imagery_channels_max_abs_error"] for x in frame_results])
    errors_z = np.asarray([x["cosSOZ_max_abs_error"] for x in frame_results])
    errors_a = np.asarray([x["cosRAA_max_abs_error"] for x in frame_results])
    day_mismatch = sum(x["day_mask_mismatch_pixels"] for x in frame_results)
    summary = {
        "source": "/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/data/2025/YYYYMM/YYYYMMDD/YYYYMMDDHHMM.npy",
        "scope": "12 selected real-AGRI validation frames, each matched by station and UTC time; only each station 16x16 patch was compared; test not accessed",
        "source_channel_contract": "20 channels: C01-C15, SAA(index 15), SAZ(index 16), SOA(index 17), SOZ(index 18), day_mask(index 19); channel 13 in the 0-based source list is C13",
        "packed_contract": "13 imagery channels from source indices 0..5 and 8..14, then cos(deg2rad(SOZ)), cos(deg2rad(SOA-SAA)), and day_mask=(source day_mask>0.5)",
        "training_normalization": {
            "cosSOZ_mean": norm["mean"][13], "cosSOZ_std": norm["std"][13],
            "cosRAA_mean": norm["mean"][14], "cosRAA_std": norm["std"][14],
            "day_mask_mean": norm["mean"][15], "day_mask_std": norm["std"][15],
        },
        "max_abs_error_over_12_frames": {
            "13_raw_imagery_channels": float(errors13.max()),
            "cosSOZ": float(errors_z.max()),
            "cosRAA": float(errors_a.max()),
            "day_mask_mismatch_pixels": int(day_mismatch),
        },
        "frames": frame_results,
    }
    (CASE_DIR / "original_agri_solar_input_audit_20260924.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["max_abs_error_over_12_frames"], indent=2))
    for f in frame_results:
        print(f["case_id"], f["timestamp_utc"], "SOZ", f["source_SOZ_degrees_range"],
              "cosSOZ err", f["cosSOZ_max_abs_error"], "cosRAA err", f["cosRAA_max_abs_error"],
              "day mismatches", f["day_mask_mismatch_pixels"])


if __name__ == "__main__":
    main()
