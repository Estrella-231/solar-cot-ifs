"""Read-only frozen COT retriever audit on fixed validation rows."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from train_cot_repaired import COTUNet


ROOT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907")
FEATURES = [f"C{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)]
FEATURES += ["cosSOZ", "cosRAA", "day_mask"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    config = json.loads((ROOT / "configs/r_frozen_repaired_seed42.json").read_text())
    checkpoint_path = Path(config["checkpoint"])
    assert sha(checkpoint_path) == config["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    assert sha(inspect.getfile(COTUNet)) == metadata["source_code_sha256"]
    assert metadata["test_used"] is False and metadata["train"] == 11978
    assert metadata["validation"] == 6536
    pack = ROOT / "data/cot_repaired_pack_20260907_v1"
    norm_path = pack / "norm.json"
    assert sha(norm_path) == metadata["pack_sha256"]["norm.json"]
    norm = json.loads(norm_path.read_text())
    assert norm == metadata["norm"] and norm["feature_names"] == FEATURES
    assert norm["cot_scale"] == 100 and norm["test_used"] is False
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()

    with np.load(ROOT / "runs/cot_repaired_seed42_v1/validation_predictions.npz") as saved:
        row_indices = saved["row_indices"]
        positions = np.array([0, len(row_indices) // 2, len(row_indices) - 1])
        rows = row_indices[positions]
        saved_log = saved["pred_log1p_cot"][positions]
        saved_reference = saved["reference_cot"][positions]
        saved_mask = saved["mask"][positions]
    raw = np.asarray(np.load(pack / "x_raw.npy", mmap_mode="r")[rows], dtype=np.float32)
    target = np.asarray(np.load(pack / "target.npy", mmap_mode="r")[rows], dtype=np.float32)
    mask = np.asarray(np.load(pack / "mask.npy", mmap_mode="r")[rows], dtype=bool)
    assert np.array_equal(target * np.float32(100), saved_reference)
    assert np.array_equal(mask, saved_mask)
    normalized = (raw - mean) / std
    imputed = int((~np.isfinite(normalized)).sum())
    normalized[~np.isfinite(normalized)] = 0
    model = COTUNet().float().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        inp = torch.from_numpy(np.ascontiguousarray(normalized))
        direct = model(inp)
        decomposed = torch.nn.functional.softplus(model.head(model.forward_features(inp)))
    assert direct.dtype == torch.float32 and torch.isfinite(direct).all()
    assert torch.equal(direct, decomposed)
    max_saved_difference = float(np.max(np.abs(direct.numpy() - saved_log)))
    assert max_saved_difference < 0.01, max_saved_difference  # Saved validation used GPU FP16.
    report = {
        "state": "PASS_FROZEN_R_CODE_AND_VALIDATION_RELOAD",
        "test_used": False,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "model_source_sha256": metadata["source_code_sha256"],
        "norm_sha256": sha(norm_path),
        "sample_pack_rows": rows.tolist(),
        "sample_nonfinite_values_imputed": imputed,
        "direct_vs_decomposed_max_log1p_difference": 0.0,
        "cpu_fp32_vs_saved_gpu_fp16_max_log1p_difference": max_saved_difference,
        "note": "Three fixed validation rows; does not certify CPP upstream production or forecast-domain accuracy.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
