#!/usr/bin/env python3
"""Extract frozen Hubei R dec1 features for the same-time AL control.

Only keys present in the already-frozen train/validation GHI pack are used.
The script reads AGRI/geometry NPZ payloads for those keys, but never reads
test/buffer rows, GHI values, or CPP targets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

AGRI_CH = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
EXPECTED_CHECKPOINT = "1e6c58b97ede89dbeb23789df867076f0413c14c4d23d158be0c2d0e897f7d83"
EXPECTED_SOURCE = "8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05"
EXPECTED_NORM = "11b9014bc07df2d72926ed230ed954b7eea95621869e6479c4aba7f6986bb86a"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def time_key(value: str) -> str:
    text = value.strip().replace("-", "").replace(":", "").replace(" ", "")
    if len(text) == 14:
        return text[:12]
    if len(text) == 12:
        return text
    raise ValueError(f"Bad BJT timestamp: {value!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("pack", "feature-rows", "feature-report", "r-source", "r-checkpoint", "r-norm", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--agri-root", type=Path,
                   help="Optional staged data16_label root; files resolve as ROOT/<month>/<basename>")
    p.add_argument("--batch-frames", type=int, default=64)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("AL feature extraction requires a scheduler-assigned CUDA GPU")

    pack_meta = json.loads((a.pack / "pack_report.json").read_text(encoding="utf-8"))
    feature_meta = json.loads(a.feature_report.read_text(encoding="utf-8"))
    if pack_meta.get("state") != "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("Expected frozen same-time Hubei train/validation pack")
    if pack_meta.get("test_and_buffer_values_read") != 0:
        raise ValueError("GHI pack test/buffer gate failed")
    if feature_meta.get("state") != "HUBEI_STATION_3X3_RAW_FEATURES_TRAINVAL_COMPLETE_TEST_CLOSED":
        raise ValueError("Feature bank is not an audited train/validation bank")
    if feature_meta.get("test_payloads_read") != 0:
        raise ValueError("Feature-bank test gate failed")
    if (feature_meta.get("split_sha256") != pack_meta.get("split_sha256") or
            feature_meta.get("feature_shape", [None])[0] != pack_meta.get("feature_rows")):
        raise ValueError("Feature bank and GHI pack are not bound to the same frozen feature manifest")
    for path, expected, label in ((a.r_checkpoint, EXPECTED_CHECKPOINT, "checkpoint"),
                                  (a.r_source, EXPECTED_SOURCE, "source"),
                                  (a.r_norm, EXPECTED_NORM, "normalization")):
        if sha(path) != expected:
            raise ValueError(f"Frozen Hubei R {label} hash mismatch")

    with (a.pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        ghi_rows = list(csv.DictReader(f))
    if any(r["split"] not in ("train", "val") for r in ghi_rows):
        raise ValueError("GHI pack contains non-train/validation rows")
    required_keys = {(r["station"], time_key(r["time_bjt"]), r["split"]) for r in ghi_rows}
    if len(required_keys) != len(ghi_rows):
        raise ValueError("Duplicate GHI pack station/time key")
    needed_times = {time_key(r["time_bjt"]): r["split"] for r in ghi_rows}

    with a.feature_rows.open(encoding="utf-8-sig", newline="") as f:
        all_frames = list(csv.DictReader(f))
    frame_by_time = {}
    for r in all_frames:
        if r["split"] not in ("train", "val"):
            raise ValueError("Feature manifest has a non-train/validation key")
        t = time_key(r["time_bjt"])
        if t in frame_by_time:
            raise ValueError(f"Duplicate feature time key {t}")
        frame_by_time[t] = r
    if set(needed_times) - set(frame_by_time):
        raise ValueError(f"Missing AGRI frames for {len(set(needed_times)-set(frame_by_time))} GHI times")
    for t, split in needed_times.items():
        if frame_by_time[t]["split"] != split:
            raise ValueError(f"Pack/feature split mismatch at {t}")

    norm = json.loads(a.r_norm.read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, 16, 1, 1)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(1, 16, 1, 1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid frozen R normalization")

    spec = importlib.util.spec_from_file_location("hubei_frozen_R_for_AL", a.r_source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.COTUNet().cuda().eval().requires_grad_(False)
    checkpoint = torch.load(a.r_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    times = sorted(needed_times)
    time_to_index = {t: i for i, t in enumerate(times)}
    a.output.mkdir(parents=True)
    features = np.lib.format.open_memmap(
        a.output / "r_dec1_station_3x3.npy", mode="w+", dtype=np.float32,
        shape=(len(times), 3, 16, 3, 3))
    with (a.output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("index", "time_bjt", "split"))
        for i, t in enumerate(times):
            w.writerow((i, t, needed_times[t]))

    batch_x = []
    batch_times = []
    done = 0

    def flush() -> None:
        nonlocal batch_x, batch_times, done
        if not batch_x:
            return
        x = torch.from_numpy(np.stack(batch_x)).cuda(non_blocking=True)
        with torch.inference_mode():
            # Match frozen R's inference precision used by the audited COT bank.
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model.forward_features(x).float()
            if hidden.shape != (len(batch_x), 16, 16, 16):
                raise ValueError(f"Unexpected Hubei R dec1 shape {tuple(hidden.shape)}")
            arr = hidden.cpu().numpy()
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite frozen-R dec1 features")
        for j, t in enumerate(batch_times):
            idx = time_to_index[t]
            for si, station in enumerate(STATIONS):
                r, c = CENTERS[station]
                features[idx, si] = arr[j, :, r-1:r+2, c-1:c+2]
        done += len(batch_times)
        features.flush()
        print(f"FEATURES {done}/{len(times)}", flush=True)
        batch_x, batch_times = [], []

    for t in times:
        row = frame_by_time[t]
        source_path = (a.agri_root / Path(row["path_agri"]).parent.name / Path(row["path_agri"]).name
                       if a.agri_root is not None else Path(row["path_agri"]))
        with np.load(source_path, allow_pickle=False) as z:
            if not {"patch", "cosSOZ", "cosRAA", "day_mask"}.issubset(z.files):
                raise ValueError(f"Missing R input fields at {t}")
            patch = np.asarray(z["patch"], dtype=np.float32)
            geo = [np.asarray(z[k], dtype=np.float32) for k in ("cosSOZ", "cosRAA", "day_mask")]
        if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
            raise ValueError(f"Unexpected AGRI patch shape at {t}: {patch.shape}")
        if any(g.shape != (16, 16) for g in geo):
            raise ValueError(f"Unexpected geometry shape at {t}")
        raw = np.concatenate((patch[list(AGRI_CH)], np.stack(geo)), axis=0)
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite frozen-R input at {t}")
        x = ((raw[None] - mean) / std).astype(np.float32)
        # Match the frozen-R forecast inference contract exactly: day_mask is
        # constant in the R train split, so its fitted std is 1e-6.  Applying
        # ordinary z-scoring to nighttime zeros would create -1e6, overflow
        # under the audited fp16 inference path, and produce nonfinite hidden
        # features.  The production inference path maps normalized night-mask
        # values to zero; keep the AL feature extractor identical.
        solar = raw[15:16] > 0.5
        x[:, 15:16] = np.where(solar[None], x[:, 15:16], 0.0)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite normalized frozen-R input at {t}")
        batch_x.append(x[0])
        batch_times.append(t)
        if len(batch_x) >= a.batch_frames:
            flush()
    flush()
    features.flush()

    # Validate only keys required by the fixed GHI pack, station by station.
    key_rows = {(r["station"], time_key(r["time_bjt"]), r["split"]) for r in ghi_rows}
    if not key_rows.issubset({(s, t, needed_times[t]) for s, t, _ in key_rows}):
        raise ValueError("AL sidecar key coverage failed")
    report = {
        "state": "HUBEI_FROZEN_R_DEC1_AL_FEATURE_BANK_TRAINVAL_COMPLETE_TEST_CLOSED",
        "feature": "frozen Hubei R dec1 activation before COT head; station-specific 3x3 crop",
        "channels": 16, "stations": list(STATIONS), "rows": len(times),
        "feature_shape": list(features.shape), "ghi_pack_rows": len(ghi_rows),
        "split_sha256": pack_meta["split_sha256"],
        "pack_report_sha256": sha(a.pack / "pack_report.json"),
        "feature_report_sha256": sha(a.feature_report),
        "frozen_R_checkpoint_sha256": sha(a.r_checkpoint),
        "frozen_R_source_sha256": sha(a.r_source), "frozen_R_norm_sha256": sha(a.r_norm),
        "agri_source_mode": ("staged-month-basename-remap" if a.agri_root is not None else "manifest-paths"),
        "test_payloads_read": 0, "ghi_labels_read": False, "cpp_payloads_read": 0,
        "feature_file_sha256": sha(a.output / "r_dec1_station_3x3.npy"),
        "rows_sha256": sha(a.output / "rows.csv"),
        "note": "Train/validation only. R is frozen and used as a feature extractor; this control tests whether local R hidden features explain the AC increment, not whether R is an independent truth source.",
    }
    (a.output / "AL_FEATURE_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
