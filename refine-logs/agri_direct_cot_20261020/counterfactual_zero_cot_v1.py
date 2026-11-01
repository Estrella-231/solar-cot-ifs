from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from single_frame_ghi_head import SingleFrameGHIHead


ROOT = Path(__file__).resolve().parent
TMP = Path(r"C:\Users\29728\AppData\Local\Temp\direct_cot_rowbank_recovery_20261021_0530")
PACK = TMP / "results" / "direct_cot_single_frame_ghi_pack_v1"
WORKERS = ROOT / "results" / "direct_cot_single_frame_ghi_heads_v1_local4060"
OUT = ROOT / "results" / "direct_cot_ghi_validation_score_v1_local4060" / "event_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required for same-head counterfactual inference")
device = torch.device("cuda")
rows_out = {}

for station in ("sili", "zhujia"):
    sd = PACK / station / "val"
    agri = torch.from_numpy(np.load(sd / "agri.npy", allow_pickle=False).astype(np.float32)).to(device)
    geom = torch.from_numpy(np.load(sd / "geometry.npy", allow_pickle=False).astype(np.float32)).to(device)
    cot = torch.from_numpy(np.load(sd / "cot" / "p_xc" / "val.npy", allow_pickle=False).astype(np.float32)).to(device)
    ghi = np.load(sd / "ghi.npy", allow_pickle=False).astype(np.float64)
    clear = np.load(sd / "clear.npy", allow_pickle=False).astype(np.float64)
    keys = (sd / "keys.jsonl").read_bytes()
    checkpoint = WORKERS / f"{station}_worker" / station / "p_xc" / "seed42" / "epoch_030.pt"
    model = SingleFrameGHIHead().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["station"] != station or state["arm"] != "p_xc" or state["seed"] != 42:
        raise ValueError(f"unexpected checkpoint identity: {checkpoint}")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    cot_pred, cot_zero = [], []
    with torch.inference_mode():
        for lo in range(0, len(ghi), 1024):
            sl = slice(lo, min(lo + 1024, len(ghi)))
            p1 = model(agri[sl], geom[sl], cot[sl]).double() * torch.from_numpy(clear[sl]).to(device)
            p0 = model(agri[sl], geom[sl], torch.zeros_like(cot[sl])).double() * torch.from_numpy(clear[sl]).to(device)
            cot_pred.append(p1.cpu().numpy())
            cot_zero.append(p0.cpu().numpy())
    pred_cot = np.concatenate(cot_pred)
    pred_zero = np.concatenate(cot_zero)
    saved = np.load(checkpoint.parent / "validation_predictions.npz", allow_pickle=False)
    saved_bytes = np.asarray(saved["keys_jsonl_bytes"], dtype=np.uint8).tobytes()
    if saved_bytes != keys:
        raise ValueError(f"validation key mismatch for {station}")
    saved_pred = np.asarray(saved["prediction_ghi_wm2"], dtype=np.float64)
    max_drift = float(np.max(np.abs(saved_pred - pred_cot)))
    if max_drift > 1e-3:
        raise ValueError(f"replayed model differs from saved prediction: {station} {max_drift}")
    np.savez_compressed(OUT / f"{station}_seed42_same_head_zero_cot.npz",
                        prediction_with_pxc_cot=pred_cot,
                        prediction_same_head_cot_zero=pred_zero,
                        truth_ghi_wm2=ghi,
                        keys_jsonl_bytes=np.frombuffer(keys, dtype=np.uint8))
    rows_out[station] = {
        "n_validation": len(ghi),
        "saved_prediction_replay_max_abs_drift_wm2": max_drift,
        "mean_delta_with_cot_minus_zero_cot_wm2": float(np.mean(pred_cot - pred_zero)),
        "rmse_with_pxc_cot": float(np.sqrt(np.mean((pred_cot - ghi) ** 2))),
        "rmse_same_head_zero_cot": float(np.sqrt(np.mean((pred_zero - ghi) ** 2))),
        "delta_rmse_with_cot_minus_zero_cot_wm2": float(np.sqrt(np.mean((pred_cot - ghi) ** 2)) - np.sqrt(np.mean((pred_zero - ghi) ** 2))),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "pack_manifest_sha256": hashlib.sha256((PACK / "pack_manifest.json").read_bytes()).hexdigest(),
        "keys_sha256": hashlib.sha256(keys).hexdigest(),
        "test_payload_opened": False,
        "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False,
        "cpp_used_as_input": False,
    }
    del agri, geom, cot, model
    torch.cuda.empty_cache()

manifest = {
    "state": "COMPLETE_DIRECT_COT_SAME_HEAD_ZERO_COT_COUNTERFACTUAL",
    "scope": "validation-only inference counterfactual; P-XC seed42 head frozen; identical AGRI and geometry; replace normalized predicted-COT channel with the exact neutral zero tensor",
    "not_a_retrained_model": True,
    "cpp_reference_used": False,
    "test_payload_opened": False,
    "buffer_payload_opened": False,
    "future_observed_agri_used_as_input": False,
    "cpp_used_as_input": False,
    "stations": rows_out,
}
(OUT / "same_head_zero_cot_counterfactual.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, allow_nan=False))
