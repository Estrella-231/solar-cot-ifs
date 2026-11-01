"""GPU B0 acceptance for station-aware, position-aligned AGRI+COT fusion."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from station_aware_spatial_head_v2 import STATION_ROW_COL, StationAwareSpatialHead

FUSION_DESCRIPTION = "concat(13 AGRI, 1 COT) then Conv2d at each aligned pixel"
AGGREGATION_DESCRIPTION = ("exact station feature + 5x5 station neighbourhood + "
                           "learned position-preserving 4x4 full-patch encoding; "
                           "no COT mean/percentile/statistical feature")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main(root, output, sidecar):
    root, output, sidecar = root.resolve(), output.resolve(), sidecar.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite a B0 acceptance directory")
    base = root / "data/head_pack_trainval_20260908_v1"
    base_audit = json.loads((base / "audit.json").read_text())
    map_audit = json.loads((sidecar / "audit.json").read_text())
    if base_audit["test_used"] or map_audit["test_used"]:
        raise RuntimeError("B0 must be test-free")
    if sha(base / "audit.json") != map_audit["base_pack_audit_sha256"]:
        raise RuntimeError("base/sidecar binding drift")
    if sha(sidecar / "cot_map.npy") != map_audit["arrays"]["cot_map.npy"]["sha256"]:
        raise RuntimeError("COT sidecar hash drift")
    arrays = {name: np.load(base / (name + ".npy"), mmap_mode="r")
              for name in ("x", "g", "station", "lead", "split")}
    cot = np.load(sidecar / "cot_map.npy", mmap_mode="r")
    if arrays["x"].shape != (637902, 13, 16, 16) or cot.shape != (637902, 1, 16, 16):
        raise RuntimeError("unexpected aligned patch shape")
    # Use one contiguous training span: a sparse sweep across the 9.8-GB AGRI
    # array turns this acceptance test into a shared-filesystem benchmark.
    train_rows = np.flatnonzero(arrays["split"] == 0)
    sample = train_rows[:2048]
    if (len(sample) != 2048 or not np.all(arrays["split"][sample] == 0)
            or not np.all(np.diff(sample) == 1)):
        raise RuntimeError("smoke sample is not a complete contiguous train span")
    device = "cuda"
    torch.manual_seed(42)
    model = StationAwareSpatialHead().to(device)
    values = [torch.from_numpy(np.asarray(arrays[k][sample])).to(device)
              for k in ("x",)]
    values += [torch.from_numpy(np.asarray(cot[sample])).to(device),
               torch.from_numpy(np.asarray(arrays["g"][sample])).to(device),
               torch.from_numpy(np.asarray(arrays["station"][sample])).to(device),
               torch.from_numpy(np.asarray(arrays["lead"][sample])).to(device)]
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(*values).float()
        loss = pred.square().mean()
    loss.backward()
    if hasattr(model, "encoder"):
        cot_grad = model.encoder[0].weight.grad[:, 13].abs().sum().item()
    elif hasattr(model, "cot_gate"):
        cot_grad = model.cot_gate.weight.grad.abs().sum().item()
    else:
        raise RuntimeError("model exposes no inspectable COT fusion gradient")
    if not torch.isfinite(loss) or not np.isfinite(cot_grad) or cot_grad <= 0:
        raise RuntimeError("joint COT/AGRI fusion did not backpropagate")
    opt.step()
    model.eval()
    with torch.no_grad():
        baseline = model(*values)
        changed = model(values[0], values[1] + 0.25, *values[2:])
        spatially_rearranged = model(values[0], values[1].flip(-1), *values[2:])
    response = (changed - baseline).abs()
    spatial_response = (spatially_rearranged - baseline).abs()
    if not torch.isfinite(response).all() or response.max().item() <= 0:
        raise RuntimeError("COT plane is not connected to station prediction")
    if not torch.isfinite(spatial_response).all() or spatial_response.max().item() <= 0:
        raise RuntimeError("COT spatial arrangement is not connected to station prediction")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    output.mkdir(parents=True)
    report = {
        "state": "PASS_B0_STATION_AWARE_SPATIAL_GPU_SMOKE",
        "test_used": False,
        "station_center_patch_row_col": list(STATION_ROW_COL),
        "fusion": FUSION_DESCRIPTION,
        "aggregation": AGGREGATION_DESCRIPTION,
        "batch": len(sample), "loss": float(loss.detach()),
        "cot_first_layer_gradient_l1": cot_grad,
        "cot_plus_0.25_prediction_abs_mean": float(response.mean()),
        "cot_plus_0.25_prediction_abs_max": float(response.max()),
        "cot_horizontal_reflection_prediction_abs_mean": float(spatial_response.mean()),
        "cot_horizontal_reflection_prediction_abs_max": float(spatial_response.max()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_GiB": peak, "total_GiB": total, "peak_fraction": peak / total,
        "sidecar_audit_sha256": sha(sidecar / "audit.json"),
        "code_sha256": sha(__file__), "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "seconds": time.time(),
    }
    if report["peak_fraction"] >= .80:
        raise RuntimeError("B0 exceeded retained-memory limit")
    atomic(output / "station_aware_spatial_smoke.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output, args.sidecar)
