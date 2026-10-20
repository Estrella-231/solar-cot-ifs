"""Infer frozen Hubei R for every frozen train/val AGRI NPZ path; no labels read."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-report", required=True)
    p.add_argument("--norm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    manifest, norm_path, run, model_source, output = map(Path, (a.manifest, a.norm, a.run, a.model_source, a.output))
    report = json.loads(Path(a.manifest_report).read_text(encoding="utf-8"))
    split_ok = report.get("split_sha256") == EXPECTED_SPLIT_SHA
    test_closed = (report.get("test_payloads_read") == 0 or
                   report.get("test_or_buffer_payloads_staged") == 0)
    if not split_ok or not test_closed:
        raise ValueError("Full train/validation path manifest has wrong split provenance")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; no CPU fallback")
    r_status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    r_meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    if r_status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or r_meta.get("test_used") is not False:
        raise ValueError("Frozen R is not validation-only/test-closed")
    if r_meta.get("model_source_sha256") != sha(model_source):
        raise ValueError("R model source SHA mismatch")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Manifest includes a non-train/validation row")
    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    expected_counts = report.get("rows_by_split") or report.get("paths_by_frozen_split_metadata_only")
    if isinstance(expected_counts, dict):
        expected_counts = {s: int(expected_counts.get(s, -1)) for s in ("train", "val")}
        if split_counts != expected_counts or not all(split_counts.values()):
            raise ValueError(f"Manifest/report count mismatch: rows={split_counts}, report={expected_counts}")
    else:
        staged_n = report.get("unique_trainval_agri_frames")
        if staged_n is None or int(staged_n) != len(rows) or not all(split_counts.values()):
            raise ValueError(f"Manifest/stage count mismatch: rows={len(rows)}, staged={staged_n}")
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    if norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("Normalization is not train-only")
    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, 16, 1, 1)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(1, 16, 1, 1)

    spec = importlib.util.spec_from_file_location("hubei_frozen_R_full_bank", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.COTUNet().cuda().eval()
    checkpoint = torch.load(run / "best.pt", map_location="cuda:0", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    output.mkdir(parents=True, exist_ok=False)
    pred = np.empty((len(rows), 1, 16, 16), dtype=np.float32)
    nonfinite_by_channel = np.zeros(16, dtype=np.int64)
    no_solar_pixels = 0
    total_pixels = 0
    for start in range(0, len(rows), a.batch):
        end = min(start + a.batch, len(rows))
        raw = np.empty((end-start, 16, 16, 16), dtype=np.float32)
        for j, row in enumerate(rows[start:end]):
            with np.load(row["path_agri"], allow_pickle=True) as z:
                required = {"patch", "cosSOZ", "cosRAA", "day_mask"}
                if not required.issubset(z.files):
                    raise ValueError(f"Missing R input fields at {row['time_bjt']}")
                patch = np.asarray(z["patch"], dtype=np.float32)
                if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
                    raise ValueError(f"Unexpected AGRI input shape at {row['time_bjt']}: {patch.shape}")
                raw[j] = np.concatenate((patch[AGRI_CH], np.asarray(z["cosSOZ"], np.float32)[None],
                                         np.asarray(z["cosRAA"], np.float32)[None],
                                         np.asarray(z["day_mask"], np.float32)[None]), axis=0)
        # The train-only day_mask is constant 1 on CPP-supervised samples,
        # so its fitted std is 1e-6. A literal night-time 0 would normalize
        # to -1e6 and overflow the R network. COT is not retrievable where
        # day_mask is off: feed the neutral normalized value there and store
        # a zero COT sentinel at those pixels. Keep every AGRI row/key.
        solar_valid = raw[:, 15:16] > 0.5
        no_solar_pixels += int((~solar_valid).sum())
        total_pixels += int(solar_valid.size)
        normalized = (raw - mean) / std
        normalized[:, 15:16] = np.where(solar_valid, normalized[:, 15:16], 0.0)
        nonfinite = ~np.isfinite(normalized)
        nonfinite_by_channel += nonfinite.sum(axis=(0, 2, 3))
        normalized[nonfinite] = 0.0
        x = torch.from_numpy(normalized).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            pred_log = model(x).float()
        if not torch.isfinite(pred_log).all():
            raise RuntimeError(
                f"Nonfinite log1p(COT) prediction at row range {start}:{end}; "
                f"input finite={bool(torch.isfinite(x).all())}, "
                f"input absmax={float(x.abs().max())}, first_time={rows[start]['time_bjt']}, "
                f"channel_absmax={x.abs().amax(dim=(0, 2, 3)).detach().cpu().tolist()}"
            )
        cot = torch.expm1(pred_log)
        solar_valid_t = torch.from_numpy(solar_valid).cuda()
        cot = cot.masked_fill(~solar_valid_t, 0.0)
        if not torch.isfinite(cot).all() or (cot < 0).any():
            finite_log = pred_log[torch.isfinite(pred_log)]
            raise RuntimeError(
                f"Invalid COT prediction at row range {start}:{end}; "
                f"log_finite={bool(torch.isfinite(pred_log).all())}, "
                f"log_min={float(finite_log.min()) if finite_log.numel() else 'NA'}, "
                f"log_max={float(finite_log.max()) if finite_log.numel() else 'NA'}, "
                f"cot_finite={bool(torch.isfinite(cot).all())}, cot_min={float(torch.nan_to_num(cot).min())}, "
                f"cot_max={float(torch.nan_to_num(cot).max())}, input_absmax={float(x.abs().max())}"
            )
        pred[start:end] = cot.cpu().numpy().astype(np.float32)
        if end % 4096 == 0 or end == len(rows):
            print(f"INFERRED {end}/{len(rows)}", flush=True)

    np.save(output / "cot_pred_trainval_full.npy", pred, allow_pickle=False)
    with (output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "time_bjt", "split", "path_agri"])
        w.writerows([[i, r["time_bjt"], r["split"], r["path_agri"]] for i, r in enumerate(rows)])
    out_report = {
        "state": "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED",
        "trainval_rows_by_split": split_counts,
        "prediction_shape": list(pred.shape), "finite_nonnegative": bool(np.isfinite(pred).all() and (pred >= 0).all()),
        "prediction_min_max": [float(pred.min()), float(pred.max())],
        "test_payloads_read": 0, "test_path_rows_in_manifest": 0,
        "cpp_supervision_not_required_for_inference": True,
        "normalization_source": "R fixed-v2 pack, fitted on CPP-supervised frozen train subset only",
        "normalized_nonfinite_input_values_by_channel": nonfinite_by_channel.tolist(),
        "day_mask_off_pixels": no_solar_pixels, "pixels_total": total_pixels,
        "cot_sentinel": "0 where AGRI day_mask <= 0.5; model output not claimed outside daytime retrieval support",
        "frozen_R_checkpoint_sha256": sha(run / "best.pt"),
        "model_source_sha256": sha(model_source), "manifest_sha256": sha(manifest),
        "norm_sha256": sha(norm_path), "prediction_sha256": sha(output / "cot_pred_trainval_full.npy"),
        "rows_sha256": sha(output / "rows.csv"), "batch": a.batch,
        "train_predictions_are_oof": False,
        "note": "R predictions cover every frozen train/validation AGRI NPZ path. CPP is retrieval supervision, not independent truth; train predictions are in-domain, not OOF.",
    }
    (output / "BANK_REPORT.json").write_text(json.dumps(out_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out_report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
