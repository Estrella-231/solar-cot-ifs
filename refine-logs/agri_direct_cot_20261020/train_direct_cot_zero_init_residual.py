"""One frozen-no-COT-base conditional residual candidate for the direct-COT study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from single_frame_ghi_head import SingleFrameGHIHead


STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)
EPOCHS = 30
BATCH = 512
ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32 * 3 * 3 + 9, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, agri_features: torch.Tensor, cot: torch.Tensor):
        x = torch.cat((agri_features.flatten(1), cot.flatten(1)), dim=1)
        return self.net(x).squeeze(1)


def load_arrays(root: Path, station: str, split: str, device):
    d = root / station / split
    arrays = {
        "agri": np.load(d / "agri.npy", mmap_mode="r", allow_pickle=False),
        "geometry": np.load(d / "geometry.npy", mmap_mode="r", allow_pickle=False),
        "ghi": np.load(d / "ghi.npy", mmap_mode="r", allow_pickle=False),
        "clear": np.load(d / "clear.npy", mmap_mode="r", allow_pickle=False),
        "cot": np.load(d / "cot" / "p_xc" / f"{split}.npy", mmap_mode="r", allow_pickle=False),
    }
    keys = (d / "keys.jsonl").read_bytes()
    n = len(arrays["ghi"])
    if tuple(arrays["agri"].shape) != (n, 13, 3, 3) or tuple(arrays["geometry"].shape) != (n, 2, 3, 3):
        raise ValueError(f"invalid {station}/{split} feature shape")
    if tuple(arrays["cot"].shape) != (n, 1, 3, 3) or len(keys.splitlines()) != n:
        raise ValueError(f"invalid {station}/{split} COT/key shape")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError(f"nonfinite {station}/{split} data")
    if (arrays["clear"] <= 0).any() or (arrays["ghi"] < 0).any():
        raise ValueError(f"invalid {station}/{split} target/clear values")
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32).copy()).to(device) for k, v in arrays.items()}, keys


def frozen_features(base, data, batch=2048):
    feats, base_ghi = [], []
    base.eval()
    with torch.inference_mode():
        for lo in range(0, len(data["ghi"]), batch):
            sl = slice(lo, min(lo + batch, len(data["ghi"])))
            inp = torch.cat((data["agri"][sl], data["geometry"][sl]), dim=1)
            ft = base.agri_encoder(inp)
            kt = base(data["agri"][sl], data["geometry"][sl], torch.zeros_like(data["cot"][sl]))
            feats.append(ft)
            base_ghi.append((kt * data["clear"][sl]).detach())
    return torch.cat(feats, dim=0).detach(), torch.cat(base_ghi, dim=0).detach()


def train_one(station: str, seed: int, pack: Path, output: Path, device):
    run = output / station / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    base_ckpt = ROOT / "results" / "direct_cot_single_frame_ghi_heads_v1_local4060" / f"{station}_worker" / station / "none" / f"seed{seed}" / "epoch_030.pt"
    if not base_ckpt.is_file():
        raise FileNotFoundError(base_ckpt)
    state = torch.load(base_ckpt, map_location=device, weights_only=False)
    if state.get("station") != station or state.get("arm") != "none" or int(state.get("seed")) != seed or int(state.get("epochs")) != EPOCHS:
        raise ValueError(f"unexpected frozen base checkpoint identity: {base_ckpt}")
    base = SingleFrameGHIHead().to(device)
    base.load_state_dict(state["model"], strict=True)
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()

    train, train_keys = load_arrays(pack, station, "train", device)
    val, val_keys = load_arrays(pack, station, "val", device)
    if torch.count_nonzero(train["cot"]).item() == 0:
        raise ValueError("P-XC training COT unexpectedly all zero")
    feat_tr, base_tr = frozen_features(base, train)
    feat_va, base_va = frozen_features(base, val)
    # Replay check against the original no-COT arm's saved validation prediction.
    baseline_npz = np.load(base_ckpt.parent / "validation_predictions.npz", allow_pickle=False)
    if np.asarray(baseline_npz["keys_jsonl_bytes"], dtype=np.uint8).tobytes() != val_keys:
        raise ValueError("frozen baseline validation keys differ from current pack")
    replay = base_va.double().cpu().numpy()
    saved = np.asarray(baseline_npz["prediction_ghi_wm2"], dtype=np.float64)
    drift = float(np.max(np.abs(replay - saved)))
    if drift > 1e-3:
        raise ValueError(f"frozen baseline replay drift too large: {drift}")

    seed_all(seed + 100000)
    residual = Residual().to(device)
    if torch.count_nonzero(residual.net[-1].weight).item() or torch.count_nonzero(residual.net[-1].bias).item():
        raise RuntimeError("residual branch must initialize to exact zero")
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    n = len(train["ghi"])
    updates = 0
    curve = []
    started = time.monotonic()
    for epoch in range(EPOCHS):
        order = np.random.default_rng(seed + epoch).permutation(n)
        residual.train()
        loss_sum = 0.0
        for lo in range(0, n, BATCH):
            ids = torch.as_tensor(order[lo:lo + BATCH], device=device, dtype=torch.long)
            base_scaled = base_tr[ids] / 1000.0
            delta_scaled = residual(feat_tr[ids], train["cot"][ids])
            pred = torch.clamp_min(base_scaled + delta_scaled, 0.0)
            target = train["ghi"][ids] / 1000.0
            loss = (pred - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite residual loss: {station}/{seed}/{epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            updates += 1
        scheduler.step()
        rec = {"epoch": epoch + 1, "train_scaled_mse": loss_sum / n, "updates": updates}
        curve.append(rec)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps({"station": station, "seed": seed, **rec}), flush=True)

    residual.eval()
    pred_va = []
    with torch.inference_mode():
        for lo in range(0, len(val["ghi"]), 4096):
            sl = slice(lo, min(lo + 4096, len(val["ghi"])))
            delta = residual(feat_va[sl], val["cot"][sl])
            pred = torch.clamp_min(base_va[sl] / 1000.0 + delta, 0.0) * 1000.0
            pred_va.append(pred.double().cpu().numpy())
    pred_va = np.concatenate(pred_va)
    truth = val["ghi"].double().cpu().numpy()
    torch.save({"residual": residual.state_dict(), "station": station, "seed": seed,
                "epochs": EPOCHS, "batch_size": BATCH, "frozen_base_checkpoint_sha256": sha(base_ckpt),
                "pack_manifest_sha256": sha(pack / "pack_manifest.json"),
                "zero_initialized_last_layer": True,
                "residual_definition": "normalized_GHI = frozen_noCOT_base_GHI/1000 + residual(feature_base_agri_geometry, normalized_PXC_COT_3x3), clamp_min_zero",
                "test_payload_opened": False, "buffer_payload_opened": False,
                "future_observed_agri_used_as_input": False, "cpp_used_as_input": False},
               run / f"epoch_{EPOCHS:03d}.pt")
    np.savez_compressed(run / "validation_predictions.npz", prediction_ghi_wm2=pred_va,
                        truth_ghi_wm2=truth, keys_jsonl_bytes=np.frombuffer(val_keys, dtype=np.uint8),
                        frozen_baseline_ghi_wm2=base_va.double().cpu().numpy())
    (run / "training_curve.json").write_text(json.dumps({"history": curve}, indent=2) + "\n", encoding="utf-8")
    metrics = {"station": station, "seed": seed, "n_train": n, "n_val": len(truth),
               "epochs": EPOCHS, "batch_size": BATCH, "updates": updates,
               "frozen_baseline_replay_max_abs_drift_wm2": drift,
               "candidate_rmse_wm2": float(np.sqrt(np.mean((pred_va - truth) ** 2))),
               "baseline_rmse_wm2": float(np.sqrt(np.mean((base_va.double().cpu().numpy() - truth) ** 2))),
               "delta_rmse_candidate_minus_baseline_wm2": float(np.sqrt(np.mean((pred_va - truth) ** 2)) - np.sqrt(np.mean((base_va.double().cpu().numpy() - truth) ** 2))),
               "elapsed_seconds": time.monotonic() - started}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    del base, residual, train, val, feat_tr, feat_va, base_tr, base_va
    torch.cuda.empty_cache()
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, default=Path(r"C:\Users\29728\AppData\Local\Temp\direct_cot_rowbank_recovery_20261021_0530\results\direct_cot_single_frame_ghi_pack_v1"))
    p.add_argument("--output", type=Path, default=ROOT / "results" / "direct_cot_ghi_residual_v1_local4060")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = json.loads((a.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_PACK":
        raise ValueError("GHI pack not complete")
    for name in ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input"):
        if manifest.get(name) is not False:
            raise ValueError(f"pack violates forbidden-access guard: {name}")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    all_metrics = []
    for station in STATIONS:
        for seed in SEEDS:
            all_metrics.append(train_one(station, seed, a.pack, a.output, torch.device("cuda")))
    receipt = {"state": "COMPLETE_DIRECT_COT_ZERO_INIT_CONDITIONAL_RESIDUAL_HEADS",
               "stations": list(STATIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH,
               "architecture": "frozen no-COT single-frame head plus 297→64→32→1 zero-initialized residual branch",
               "pack_manifest_sha256": sha(a.pack / "pack_manifest.json"),
               "trainer_sha256": sha(Path(__file__)),
               "head_definition_sha256": sha(Path(__file__).with_name("single_frame_ghi_head.py")),
               "runs": all_metrics, "test_payload_opened": False, "buffer_payload_opened": False,
               "future_observed_agri_used_as_input": False, "cpp_used_as_input": False}
    (a.output / "run_manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": receipt["state"], "runs": len(all_metrics), "trainer_sha256": receipt["trainer_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
