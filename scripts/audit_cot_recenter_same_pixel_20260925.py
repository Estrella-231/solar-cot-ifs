"""CPU diagnostic: recrop the same observed AGRI pixel into new 16x16 positions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_cot_repaired import COTUNet
from train_cot_real_agri_staged_hurdle_20260924 import CloudClassifier, ConditionalRegressors


ROOT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907")
RUN = ROOT / "runs/cot_real_agri_staged_hurdle_seed42_20260924"
SOURCE = Path("/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/data/2025/202507")
BOUNDS = {"sili": (155, 140), "zhujia": (65, 142)}
CHANNELS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
CASES = [
    (1, "zhujia", "202507060815", 15506, 0, 9),
    (2, "sili", "202507140515", 6655, 0, 9),
    (3, "zhujia", "202507060630", 15499, 8, 11),
    (4, "sili", "202507070515", 6327, 0, 6),
]
OUT = RUN / "same_pixel_recrop_validation_20260925.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def feature_window(full, row, col):
    crop = np.asarray(full[:, row:row + 16, col:col + 16], np.float32)
    assert crop.shape == (20, 16, 16)
    features = np.concatenate((crop[CHANNELS],
                               np.cos(np.deg2rad(crop[18]))[None],
                               np.cos(np.deg2rad(crop[17] - crop[15]))[None],
                               (crop[19] > .5).astype(np.float32)[None]), axis=0)
    return features.astype(np.float32)


def main():
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / "configs/r_frozen_repaired_seed42.json").read_text())
    ckpt = Path(cfg["checkpoint"])
    assert sha(ckpt) == cfg["checkpoint_sha256"]
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    norm = saved["metadata"]["norm"]
    means = np.asarray(norm["mean"], np.float32)[:, None, None]
    stds = np.asarray(norm["std"], np.float32)[:, None, None]
    backbone = COTUNet().eval()
    backbone.load_state_dict(saved["model"], strict=True)
    cls = CloudClassifier(.5).eval()
    cls.load_state_dict(torch.load(RUN / "stage1_classifier/best.pt", map_location="cpu",
                                   weights_only=False)["state_dict"], strict=True)
    regs = ConditionalRegressors(backbone.head).eval()
    regs.load_state_dict(torch.load(RUN / "stage2_conditional_regressors/best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"], strict=True)
    pack = ROOT / "data/cot_repaired_pack_20260907_v1"
    packed = np.load(pack / "x_raw.npy", mmap_mode="r")
    saved_prob = np.load(RUN / "stage1_classifier/best_validation_probabilities.npz")
    saved_final = np.load(RUN / "stage2_conditional_regressors/best_validation_predictions.npz")
    assert np.array_equal(saved_prob["row_indices"], saved_final["row_indices"])
    row_to_saved = {int(row): i for i, row in enumerate(saved_prob["row_indices"])}
    result = []
    with torch.inference_mode():
        for case_id, station, stamp, rowid, r, c in CASES:
            original_r, original_c = BOUNDS[station]
            full_path = SOURCE / stamp[:8] / (stamp + ".npy")
            full = np.load(full_path, mmap_mode="r")
            assert full.shape == (20, 256, 256)
            global_r, global_c = original_r + r, original_c + c
            baseline = feature_window(full, original_r, original_c)
            assert np.array_equal(baseline, packed[rowid]), (case_id, "original crop vs packed")
            placements = [r, 4, 8] if r == 0 else [r, 4, 0]
            variants = []
            for new_r in dict.fromkeys(placements):
                new_origin_r = global_r - new_r
                x_raw = feature_window(full, new_origin_r, original_c)
                assert np.array_equal(x_raw[:, new_r, c], baseline[:, r, c])
                x = (x_raw - means) / stds
                x[~np.isfinite(x)] = 0
                xt = torch.from_numpy(x[None])
                feat = backbone.forward_features(xt)
                probability = torch.sigmoid(cls(feat))[0, 0, new_r, c].item()
                low, cloud, _, _ = regs(feat)
                final = cloud if probability >= .5 else low
                frozen = torch.expm1(backbone(xt))[0, 0, new_r, c].item()
                variants.append({"pixel_local_row": new_r, "pixel_local_col": c,
                                 "crop_origin_row": new_origin_r, "crop_origin_col": original_c,
                                 "probability": probability,
                                 "low_expert_cot": low[0, 0, new_r, c].item(),
                                 "cloud_expert_cot": cloud[0, 0, new_r, c].item(),
                                 "final_cot": final[0, 0, new_r, c].item(),
                                 "frozen_R_cot": frozen,
                                 "same_pixel_input_max_abs_difference": float(np.max(np.abs(x_raw[:, new_r, c] - baseline[:, r, c]))),
                                 "input_nonfinite_count": int((~np.isfinite(x_raw)).sum())})
            saved_pos = row_to_saved[rowid]
            replay_p = variants[0]["probability"]
            replay_cot = variants[0]["final_cot"]
            result.append({"case_id": case_id, "station": station, "timestamp_utc": stamp,
                           "validation_row_id": rowid, "global_source_pixel": [global_r, global_c],
                           "original_local_pixel": [r, c], "reference_cot": float(saved_final["reference_cot"][saved_pos, 0, r, c]),
                           "saved_probability": float(saved_prob["cloud_probability"][saved_pos, 0, r, c]),
                           "saved_final_cot": float(saved_final["pred_cot"][saved_pos, 0, r, c]),
                           "original_replay_probability_abs_delta": abs(replay_p - float(saved_prob["cloud_probability"][saved_pos, 0, r, c])),
                           "original_replay_cot_abs_delta": abs(replay_cot - float(saved_final["pred_cot"][saved_pos, 0, r, c])),
                           "variants": variants})
    report = {"scope": "four selected validation real-AGRI pixels, frozen model CPU replay; no training or test",
              "interpretation": "Target pixel and all 16 input values held exactly constant; 16x16 surrounding scene changes with recrop. This diagnoses context/position sensitivity, not padding in isolation.",
              "checkpoint_sha256": sha(ckpt), "classifier_best_sha256": sha(RUN / "stage1_classifier/best.pt"),
              "regressor_best_sha256": sha(RUN / "stage2_conditional_regressors/best.pt"),
              "cases": result}
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{"case": x["case_id"], "saved_p": x["saved_probability"],
                       "replay_delta_p": x["original_replay_probability_abs_delta"],
                       "position_p": [round(v["probability"], 4) for v in x["variants"]],
                       "position_cot": [round(v["final_cot"], 2) for v in x["variants"]]}
                      for x in result], indent=2))


if __name__ == "__main__":
    main()
