"""Apply frozen Hubei R to every closed train/validation AGRI row."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    pack, run, model_source, output = map(Path, (a.pack, a.run, a.model_source, a.output))
    if not torch.cuda.is_available():
        raise RuntimeError("COT bank inference requires allocated GPU; CPU fallback disabled")
    run_status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if run_status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or run_status.get("test_payloads_read") != 0:
        raise ValueError("Frozen R run is not validation-only complete")
    run_meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    if run_meta.get("model_source_sha256") != sha(model_source):
        raise ValueError("Model source does not match the trained checkpoint provenance")
    status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
    expected_pack = json.loads((pack / "SHA256.json").read_text(encoding="utf-8"))
    actual_pack = {name: sha(pack / name) for name in expected_pack}
    if status.get("state") != "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED" or actual_pack != expected_pack:
        raise ValueError("Train/validation pack state or hashes invalid")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite COT bank output: {output}")
    with (pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Non-train/val key in source rows")
    x_np = np.load(pack / "x_norm.npy", allow_pickle=False)
    if x_np.shape != (len(rows), 16, 16, 16) or not np.isfinite(x_np).all():
        raise ValueError("Invalid R input pack")
    spec = importlib.util.spec_from_file_location("hubei_frozen_cot_r", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.COTUNet().cuda().eval()
    checkpoint = torch.load(run / "best.pt", map_location="cuda:0", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    prediction = np.empty((len(rows), 1, 16, 16), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), a.batch):
            end = min(start + a.batch, len(rows))
            batch = torch.from_numpy(x_np[start:end]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                pred_log = model(batch).float()
            cot = torch.expm1(pred_log)
            if not torch.isfinite(cot).all() or (cot < 0).any():
                raise RuntimeError(f"Invalid predicted COT in rows {start}:{end}")
            prediction[start:end] = cot.cpu().numpy().astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "cot_pred_trainval.npy", prediction, allow_pickle=False)
    with (output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "split", "time_bjt", "path_agri"])
        writer.writerows([[r["index"], r["split"], r["time_bjt"], r["path_agri"]] for r in rows])
    by_split = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    report = {
        "state": "HUBEI_FROZEN_R_COT_BANK_CPP_SUPPORTED_SUBSET_ONLY_TEST_CLOSED",
        "test_payloads_read": 0,
        "train_bank_is_oof": False,
        "rows_by_split": by_split,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "prediction_min_max": [float(prediction.min()), float(prediction.max())],
        "cpp_reference": "cpp_16_fixed_v2; supervision only, not independent truth",
        "frozen_R_checkpoint_sha256": sha(run / "best.pt"),
        "model_source_sha256": sha(model_source),
        "pack_hashes": actual_pack,
        "batch": a.batch,
        "note": "This pack contains only AGRI rows paired with fixed-v2 CPP files; this is not the full AGRI train/validation bank. Train COT predictions are in-domain and not OOF.",
    }
    report["cot_prediction_sha256"] = sha(output / "cot_pred_trainval.npy")
    report["rows_sha256"] = sha(output / "rows.csv")
    (output / "BANK_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
