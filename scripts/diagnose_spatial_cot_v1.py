"""No-training activation, alignment, and spatial-order diagnostic for AC_map."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spatial_head_smoke_v1 import SpatialHead


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def predict(model, data, ids, c_map):
    zeros = torch.zeros_like(data["c"][ids])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(data["x"][ids], c_map, zeros, data["g"][ids],
                     data["station"][ids], data["lead"][ids]).float()


def metrics(pred, data, ids):
    residual = pred.double() * data["clear"][ids] - data["ghi"][ids]
    vals = []
    for station in (0, 1):
        mask = data["station"][ids] == station
        vals.append(float(torch.sqrt(residual[mask].square().mean())))
    return {"station_ghi_rmse": vals, "station_equal_ghi_rmse": float(np.mean(vals))}


def main(root, output):
    root, output = root.resolve(), output.resolve()
    base = root / "data/head_pack_trainval_20260908_v1"
    sidecar = root / "data/head_spatial_cot_trainval_20260917_v2"
    run = root / "runs/spatial_b1_exploratory_seed42_20260917_v1"
    checkpoint = run / "AC_map_best.pt"
    if output.exists():
        raise RuntimeError("refuse to overwrite diagnostic directory")
    base_audit = json.loads((base / "audit.json").read_text())
    map_audit = json.loads((sidecar / "audit.json").read_text())
    if base_audit["test_used"] or map_audit["test_used"]:
        raise RuntimeError("test-bearing input")
    if sha(base / "audit.json") != map_audit["base_pack_audit_sha256"] or sha(checkpoint) != json.loads((run / "complete.json").read_text())["runs"]["AC_map"]["checkpoint_sha256"]:
        raise RuntimeError("input/checkpoint hash binding drift")
    data = {name: torch.from_numpy(np.load(base / (name + ".npy"), allow_pickle=False)).cuda()
            for name in ("x", "c", "g", "ghi", "clear", "station", "lead", "split")}
    data["cot_map"] = torch.from_numpy(np.load(sidecar / "cot_map.npy", allow_pickle=False)).cuda()
    val = torch.where(data["split"] == 1)[0]
    model = SpatialHead().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=True)
    others = torch.tensor([i for i in range(256) if i != 8 * 16 + 8], device="cuda")
    shuffled = []
    original, zeroed, permuted = [], [], []
    with torch.no_grad():
        for ids in val.split(2048):
            c = data["cot_map"][ids]
            flat = c[:, 0].reshape(len(ids), 256)
            sflat = flat.clone()
            sflat[:, others] = flat[:, torch.roll(others, shifts=37, dims=0)]
            smap = sflat.reshape(len(ids), 1, 16, 16)
            original.append(predict(model, data, ids, c).cpu())
            zeroed.append(predict(model, data, ids, torch.zeros_like(c)).cpu())
            permuted.append(predict(model, data, ids, smap).cpu())
    original, zeroed, permuted = [torch.cat(v).cuda() for v in (original, zeroed, permuted)]
    report = {"state": "COMPLETE_SPATIAL_COT_NO_TRAIN_DIAGNOSTIC", "test_used": False,
              "checkpoint_sha256": sha(checkpoint), "sidecar_audit_sha256": sha(sidecar / "audit.json"),
              "rows": len(val), "permutation": "center fixed; noncenter cyclic shift 37; train-independent",
              "original": metrics(original, data, val), "zeroed": metrics(zeroed, data, val),
              "permuted": metrics(permuted, data, val)}
    diff_zero, diff_perm = original - zeroed, original - permuted
    for name, diff in (("zero", diff_zero), ("permuted", diff_perm)):
        report["sensitivity_" + name] = {
            "prediction_rms": float(torch.sqrt(diff.double().square().mean())),
            "prediction_abs_p95": float(torch.quantile(diff.abs(), .95)),
            "fraction_abs_gt_1e-3": float((diff.abs() > 1e-3).float().mean()),
            "station_prediction_rms": [float(torch.sqrt(diff[data["station"][val] == s].double().square().mean())) for s in (0, 1)]
        }
    first = model.cnn[0]
    weight = first.weight.detach().float()
    agri_norm, cot_norm = float(weight[:, :13].norm()), float(weight[:, 13:].norm())
    sample = val[: min(4096, len(val))]
    x = data["x"][sample].float()
    c = data["cot_map"][sample].float()
    agri_out = F.conv2d(x, weight[:, :13], first.bias, padding=1)
    cot_out = F.conv2d(c, weight[:, 13:], None, padding=1)
    report["first_layer"] = {
        "agri_weight_frobenius": agri_norm, "cot_weight_frobenius": cot_norm,
        "cot_to_agri_weight_ratio": cot_norm / agri_norm,
        "sample_agri_activation_rms": float(torch.sqrt(agri_out.double().square().mean())),
        "sample_cot_activation_rms": float(torch.sqrt(cot_out.double().square().mean())),
        "sample_cot_to_agri_activation_ratio": float(torch.sqrt(cot_out.double().square().mean()) / torch.sqrt(agri_out.double().square().mean()))
    }
    norm = json.loads((sidecar / "norm.json").read_text())
    report["cot_input"] = {"normalized_mean": float(data["cot_map"][val].mean()),
                           "normalized_std": float(data["cot_map"][val].std()),
                           "train_norm_mean": norm["mean"], "train_norm_std": norm["std"],
                           "center_normalized_mean": float(data["cot_map"][val, 0, 8, 8].mean())}
    report["script_sha256"] = sha(__file__)
    report["completed_unix"] = time.time()
    output.mkdir(parents=True)
    atomic(output / "diagnostic.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
