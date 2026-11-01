"""B0 CPU/GPU acceptance for the spatial COT head; no pilot training or test use."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


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


class SpatialHead(nn.Module):
    """Matched 14-channel first-stage head. COT enters before convolution."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(14, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.lead = nn.Embedding(16, 8)
        self.station = nn.Embedding(2, 4)
        self.mlp = nn.Sequential(nn.Linear(52, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus())

    def forward(self, x, c_map, c_stat, geometry, station, lead):
        spatial = self.cnn(torch.cat((x, c_map), dim=1))
        zero = torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)
        return self.mlp(torch.cat((spatial, c_stat, zero, geometry, self.lead(lead),
                                   self.station(station)), dim=1)).squeeze(1)


def main(root, output, sidecar):
    root, output = root.resolve(), output.resolve()
    base = root / "data/head_pack_trainval_20260908_v1"
    sidecar = sidecar.resolve()
    audit = json.loads((sidecar / "audit.json").read_text())
    if audit["state"] != "COMPLETE_SPATIAL_COT_SIDECAR" or audit["test_used"]:
        raise RuntimeError("spatial sidecar is not test-free complete")
    if sha(sidecar / "cot_map.npy") != audit["arrays"]["cot_map.npy"]["sha256"]:
        raise RuntimeError("spatial COT array hash drift")
    base_audit = json.loads((base / "audit.json").read_text())
    if sha(base / "audit.json") != audit["base_pack_audit_sha256"] or base_audit["test_used"]:
        raise RuntimeError("base pack binding drift")
    output.mkdir(parents=True, exist_ok=True)
    x = np.load(base / "x.npy", mmap_mode="r")
    c = np.load(base / "c.npy", mmap_mode="r")
    g = np.load(base / "g.npy", mmap_mode="r")
    station = np.load(base / "station.npy", mmap_mode="r")
    lead = np.load(base / "lead.npy", mmap_mode="r")
    split = np.load(base / "split.npy", mmap_mode="r")
    cot = np.load(sidecar / "cot_map.npy", mmap_mode="r")
    if x.shape != (637902, 13, 16, 16) or cot.shape != (637902, 1, 16, 16):
        raise RuntimeError("spatial input shape")
    if not (np.isfinite(cot).all() and np.isin(split, [0, 1]).all() and (split == 1).sum() == 99849):
        raise RuntimeError("spatial input integrity")
    sample = np.linspace(0, len(x) - 1, 2048, dtype=np.int64)
    if not np.any(split[sample] == 0):
        raise RuntimeError("smoke needs train rows")
    device = "cuda"
    torch.manual_seed(42)
    model = SpatialHead().to(device)
    values = [torch.from_numpy(np.asarray(a[sample])).to(device) for a in (x, cot, c, g, station, lead)]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(*values).float()
        loss = pred.square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite spatial smoke loss")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("nonfinite spatial smoke gradients")
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    report = {"state": "PASS_B0_SPATIAL_GPU_SMOKE", "test_used": False, "batch": len(sample),
              "loss": float(loss.detach()), "peak_GiB": peak, "total_GiB": total,
              "peak_fraction": peak / total, "parameters": sum(p.numel() for p in model.parameters()),
              "sidecar_audit_sha256": sha(sidecar / "audit.json"), "code_sha256": sha(__file__),
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "seconds": time.time()}
    if report["peak_fraction"] >= .80:
        raise RuntimeError("smoke exceeded 80 percent GPU-memory reserve")
    atomic(output / "spatial_head_smoke.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output, args.sidecar)
