#!/usr/bin/env python3
"""Build a train/validation COT bank using train-date OOF cloud gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    for key in ("candidate", "data-root", "monthly-norm", "r-norm", "raw-bank",
                "fold-manifest", "c1-output", "input-builder", "classifier-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--batch-frames", type=int, default=512)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"output exists: {a.output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one scheduler-assigned CUDA device is required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    if sha(a.candidate) != "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807":
        raise ValueError("candidate manifest hash differs from audited train/val table")
    raw_manifest = json.loads((a.raw_bank / "bank_manifest.json").read_text())
    if (raw_manifest.get("state") != "COMPLETE" or raw_manifest.get("test_payload_opened") is not False
            or raw_manifest.get("frames_with_source_errors") != 0
            or raw_manifest.get("R_checkpoint_sha256") != "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001"):
        raise ValueError("raw COT bank provenance/status gate failed")

    builder = load_module("monthly_bank_builder", a.input_builder)
    clf_mod = load_module("cloud_gate_model", a.classifier_source)
    with a.candidate.open(newline="", encoding="utf-8-sig") as f:
        candidate_rows = list(csv.DictReader(f))
    frames = builder.rows_to_frames(a.candidate)
    raw_keys = list(csv.DictReader((a.raw_bank / "frame_keys.csv").open(newline="", encoding="utf-8")))
    if len(frames) != len(raw_keys) or len(frames) != 55733:
        raise ValueError(f"frame count mismatch: candidate={len(frames)} raw={len(raw_keys)}")
    for i, (fr, key) in enumerate(zip(frames, raw_keys)):
        if (key["index"] != str(i) or key["split"] != fr["split"]
                or key["target_time_utc"] != fr["target_time_utc"]
                or key["agri_frame_relpath"] != fr["agri_frame_relpath"]):
            raise ValueError(f"raw bank frame key mismatch at {i}")

    fold_by_date = {}
    with a.fold_manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["target_date_bjt"] in fold_by_date and fold_by_date[row["target_date_bjt"]] != int(row["fold"]):
                raise ValueError("BJT date appears in multiple OOF folds")
            fold_by_date[row["target_date_bjt"]] = int(row["fold"])

    threshold_data = json.loads((a.c1_output / "threshold_frozen.json").read_text())
    threshold = float(threshold_data["threshold"])
    if threshold_data.get("state") != "SELECTED" or threshold_data.get("selection_split") != "train_oof":
        raise ValueError("threshold was not frozen from train OOF")

    models = {}
    for key in ("fold0", "fold1", "fold2", "full_train"):
        ckpt_path = a.c1_output / f"{key}_last.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("epoch") != 30 or ckpt.get("seed") != 42:
            raise ValueError(f"classifier checkpoint contract mismatch: {ckpt_path}")
        model = clf_mod.CloudGate().to(device).float().eval().requires_grad_(False)
        model.load_state_dict(ckpt["model"], strict=True)
        models[key] = model

    monthly = json.loads(a.monthly_norm.read_text())
    simvp_mean = np.asarray(monthly["mean"], np.float32)[:, None, None]
    simvp_std = np.asarray(monthly["std"], np.float32)[:, None, None]
    rnorm = json.loads(a.r_norm.read_text())
    rmean = np.asarray(rnorm["mean"], np.float32)[:, None, None]
    rstd = np.asarray(rnorm["std"], np.float32)[:, None, None]
    if simvp_mean.shape != (13, 1, 1) or rmean.shape != (16, 1, 1) or (simvp_std <= 0).any() or (rstd <= 0).any():
        raise ValueError("normalization shape/value mismatch")

    raw_cot = np.load(a.raw_bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    raw_support = np.load(a.raw_bank / "input_valid_fraction.npy", mmap_mode="r", allow_pickle=False)
    if raw_cot.shape != (len(frames), 2, 1, 16, 16) or raw_support.shape != (len(frames), 2):
        raise ValueError("raw bank array shape mismatch")

    a.output.mkdir(parents=True)
    shutil.copyfile(a.raw_bank / "frame_keys.csv", a.output / "frame_keys.csv")
    support = np.lib.format.open_memmap(a.output / "input_valid_fraction.npy", mode="w+",
                                        dtype=np.float32, shape=raw_support.shape)
    support[:] = raw_support[:]
    gated = np.lib.format.open_memmap(a.output / "cot_log1p.npy", mode="w+",
                                      dtype=np.float32, shape=raw_cot.shape)
    prob_out = np.lib.format.open_memmap(a.output / "p_cloud.npy", mode="w+",
                                         dtype=np.float16, shape=(len(frames), 2, 16, 16))
    assignment_path = a.output / "classifier_assignment.csv"
    assignments = []
    for i, fr in enumerate(frames):
        bjt_date = (datetime.strptime(fr["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat()
        if fr["split"] == "val":
            ck = "full_train"
        else:
            ck = f"fold{fold_by_date[bjt_date]}" if bjt_date in fold_by_date else "full_train_unlabeled_date"
        assignments.append((i, fr["split"], fr["target_time_utc"], bjt_date, ck))

    # Dates with no CPP-supported rows have no labels in the classifier fit; use
    # the full classifier there. All supervised training dates use their heldout fold.
    frame_groups = {k: [r for r in assignments if r[4] == k] for k in
                    ("fold0", "fold1", "fold2", "full_train", "full_train_unlabeled_date")}
    with assignment_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["frame_index", "split", "target_time_utc", "target_date_bjt", "classifier_source"])
        w.writerows(assignments)

    progress = a.output / "progress.json"
    done = 0
    for group, group_rows in frame_groups.items():
        if not group_rows:
            continue
        model_key = "full_train" if group == "full_train_unlabeled_date" else group
        model = models[model_key]
        for lo in range(0, len(group_rows), a.batch_frames):
            chunk = group_rows[lo:lo + a.batch_frames]
            xs = []
            for idx, *_ in chunk:
                fr = frames[idx]
                x, _ = builder.get_station_inputs(a.data_root, fr, simvp_mean, simvp_std, rmean, rstd)
                xs.append(x)
            x_np = np.stack(xs).reshape(-1, 16, 16, 16)
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                probs = torch.sigmoid(model(torch.from_numpy(x_np).to(device))).float().cpu().numpy()[:, 0]
            probs = probs.reshape(len(chunk), 2, 16, 16)
            if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
                raise ValueError("classifier emitted invalid probabilities")
            for j, (idx, *_rest) in enumerate(chunk):
                pcloud = probs[j]
                source = np.asarray(raw_cot[idx, :, 0], dtype=np.float32)
                if not np.isfinite(source).all() or (source < -1e-6).any():
                    raise ValueError(f"invalid raw log1p COT at frame {idx}")
                gated[idx, :, 0] = np.where(pcloud >= threshold, source, 0.0)
                prob_out[idx] = pcloud.astype(np.float16)
            done += len(chunk)
            if done % 2048 < len(chunk) or done == len(frames):
                gated.flush(); prob_out.flush()
                progress.write_text(json.dumps({"state": "BUILDING", "frames_complete": done,
                    "frames_total": len(frames), "test_payload_opened": False}, indent=2) + "\n")
                print(f"PROGRESS {done}/{len(frames)}", flush=True)

    gated.flush(); prob_out.flush(); support.flush()
    if done != len(frames) or not np.isfinite(np.asarray(gated)).all() or not np.isfinite(np.asarray(prob_out)).all():
        raise RuntimeError("incomplete or nonfinite gate bank")
    with assignment_path.open(newline="", encoding="utf-8") as f:
        assignment_hash = sha(assignment_path)
    manifest = {
        "state": "COMPLETE", "split_id": raw_manifest["split_id"],
        "candidate_sha256": sha(a.candidate), "raw_bank_manifest_sha256": sha(a.raw_bank / "bank_manifest.json"),
        "R_checkpoint_sha256": raw_manifest["R_checkpoint_sha256"],
        "R_source_sha256": raw_manifest["R_source_sha256"],
        "R_norm_sha256": raw_manifest["R_norm_sha256"],
        "classifier_source_sha256": sha(a.classifier_source),
        "classifier_checkpoint_sha256": {k: sha(a.c1_output / f"{k}_last.pt") for k in ("fold0", "fold1", "fold2", "full_train")},
        "oof_fold_manifest_sha256": sha(a.fold_manifest), "threshold_file_sha256": sha(a.c1_output / "threshold_frozen.json"),
        "threshold": threshold, "threshold_fit_source": "train-date three-fold OOF; fixed before GHI pack construction",
        "classifier_assignment_sha256": assignment_hash,
        "frames_total": len(frames), "train_frames": sum(fr["split"] == "train" for fr in frames),
        "val_frames": sum(fr["split"] == "val" for fr in frames),
        "frames_by_classifier": {k: len(v) for k, v in frame_groups.items()},
        "OOF_supervised_dates": len(fold_by_date), "unlabeled_train_dates_use_full_classifier": True,
        "layout": "[unique_trainval_target_frame, station(sili,zhujia), 1,16,16]",
        "COT_transform": "raw frozen-R log1p COT retained where p_cloud >= threshold; set to log1p(0)=0 otherwise",
        "probability_layout": "[unique_trainval_target_frame, station(sili,zhujia),16,16], float16",
        "input": "observed AGRI 13 channels plus cosSOZ/cosRAA/day_mask, same R normalization and station-centered 16x16 crop",
        "test_payload_opened": False,
        "array_sha256": {n: sha(a.output / n) for n in ("cot_log1p.npy", "p_cloud.npy", "input_valid_fraction.npy", "frame_keys.csv", "classifier_assignment.csv")},
    }
    (a.output / "bank_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress.write_text(json.dumps({"state": "COMPLETE", "frames_complete": done,
        "frames_total": len(frames), "test_payload_opened": False}, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("state", "threshold", "frames_total", "frames_by_classifier", "test_payload_opened")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
