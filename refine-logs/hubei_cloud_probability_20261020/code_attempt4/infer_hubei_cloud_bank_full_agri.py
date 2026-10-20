#!/usr/bin/env python3
"""Infer Hubei p_cloud over the exact full train/validation AGRI manifest.

Train keys with a date-held-out CPP-labelled classifier output use OOF values;
all other train keys and all validation keys use the full-train model. The
CPP/CLP sidecar is a retrieval reference, not an independent cloud truth.
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
from torch import nn


AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]


class CloudClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_features(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if not {"patch", "cosSOZ", "cosRAA", "day_mask"}.issubset(z.files):
            raise ValueError(f"missing AGRI/geometry channels: {path}")
        patch = np.asarray(z["patch"], dtype=np.float32)
        if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
            raise ValueError(f"unexpected AGRI patch shape at {path}: {patch.shape}")
        geom = [np.asarray(z[k], dtype=np.float32) for k in ("cosSOZ", "cosRAA", "day_mask")]
        if any(g.shape != (16, 16) for g in geom):
            raise ValueError(f"unexpected geometry shape at {path}")
        return np.concatenate((patch[AGRI_CH], np.stack(geom)), axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--classifier-run", required=True, type=Path)
    p.add_argument("--oof-bank", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full Hubei cloud bank inference requires allocated CUDA")
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = list(csv.DictReader(a.manifest.open(newline="", encoding="utf-8-sig")))
    if not rows or any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("manifest must contain train/val only")
    keys = [r["time_bjt"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate full AGRI time keys")
    model_path = a.classifier_run / "full_train.pt"
    report_path = a.classifier_run / "classifier_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_HUBEI_CLOUD_CLASSIFIER_OOF_VALIDATION_ONLY_TEST_CLOSED":
        raise ValueError("classifier run is not complete validation-only Hubei OOF")
    if report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError("classifier report violates train/validation boundary")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not {"model", "mean", "std"}.issubset(payload):
        raise ValueError("full-train classifier checkpoint lacks state/normalization")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (16,) or std.shape != (16,) or np.any(std <= 0):
        raise ValueError("invalid train-only classifier normalization")

    with np.load(a.oof_bank, allow_pickle=False) as z:
        oof_keys = z["time_bjt"].astype(str)
        oof_splits = z["split"].astype(str)
        oof_p = np.asarray(z["p_cloud"], dtype=np.float32)
    if oof_p.shape != (len(oof_keys), 16, 16) or len(set(oof_keys.tolist())) != len(oof_keys):
        raise ValueError("malformed date-OOF bank")
    oof_map = {k: i for i, k in enumerate(oof_keys.tolist())}
    split_map = {r["time_bjt"]: r["split"] for r in rows}
    for k, sp in zip(oof_keys.tolist(), oof_splits.tolist()):
        if k in split_map and split_map[k] != sp:
            raise ValueError(f"split mismatch for OOF key {k}")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    model = CloudClassifier().to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    p_full = np.empty((len(rows), 16, 16), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), a.batch):
            end = min(start + a.batch, len(rows))
            batch = np.stack([raw_features(r["path_agri"]) for r in rows[start:end]])
            xn = (batch - mean[None, :, None, None]) / std[None, :, None, None]
            xn[~np.isfinite(xn)] = 0.0
            prob = torch.sigmoid(model(torch.as_tensor(xn, dtype=torch.float32, device=device)))[:, 0]
            p_full[start:end] = prob.cpu().numpy().astype(np.float32)
            if end % 2048 == 0 or end == len(rows):
                print(f"INFERRED {end}/{len(rows)}", flush=True)

    output = p_full.copy()
    source = np.full(len(rows), "full_train", dtype="U16")
    n_oof = 0
    for i, row in enumerate(rows):
        j = oof_map.get(row["time_bjt"])
        if row["split"] == "train" and j is not None:
            if oof_splits[j] != "train":
                raise ValueError(f"train key maps to non-train OOF prediction: {row['time_bjt']}")
            output[i] = oof_p[j]
            source[i] = "date_oof"
            n_oof += 1
        elif row["split"] == "val" and j is not None and oof_splits[j] != "val":
            raise ValueError(f"validation key maps to train OOF row: {row['time_bjt']}")
    if not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise ValueError("invalid probability bank")
    a.output.mkdir(parents=True, exist_ok=False)
    np.save(a.output / "p_cloud_trainval.npy", output, allow_pickle=False)
    with (a.output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "time_bjt", "split", "path_agri", "prediction_source"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"index": i, **r, "prediction_source": source[i]})
    result = {
        "state": "COMPLETE_HUBEI_FULL_AGRI_P_CLOUD_TRAINVAL_TEST_CLOSED",
        "rows": len(rows), "train_rows": sum(r["split"] == "train" for r in rows),
        "val_rows": sum(r["split"] == "val" for r in rows), "train_date_oof_rows": n_oof,
        "full_train_fallback_rows": int(np.count_nonzero(source == "full_train")),
        "manifest_sha256": sha(a.manifest), "classifier_report_sha256": sha(report_path),
        "full_train_checkpoint_sha256": sha(model_path), "oof_bank_sha256": sha(a.oof_bank),
        "prediction_sha256": sha(a.output / "p_cloud_trainval.npy"),
        "rows_sha256": sha(a.output / "rows.csv"), "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "prediction_policy": "train keys with date OOF use OOF; all other train and all validation keys use full-train Hubei classifier",
        "reference_caveat": "CPP CLP is a retrieval reference, not independent cloud truth",
    }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
