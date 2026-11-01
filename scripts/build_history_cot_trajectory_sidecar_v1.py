"""Build causal historic-COT trajectories from the eight observed AGRI frames.

This is deliberately separate from the frozen forecast cache.  For every
source sequence it reads only the eight frames ending at `init_time`, crops
each station's 16x16 patch, and applies the already-frozen COT retriever R.
It never reads future AGRI, reference CPP, GHI, or test samples.
"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

# The frozen S checkpoint's audited implementation lives outside this paper
# repository.  Make that source root explicit instead of relying on a shell
# PYTHONPATH inherited from the 2026-09-07 cache job.
SIMVP_SOURCE = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPDDP_20260907/scripts")
if str(SIMVP_SOURCE) not in sys.path:
    sys.path.insert(0, str(SIMVP_SOURCE))
from cache_frozen_forecast import ForecastInputs, loader
from train_cot_repaired import COTUNet


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def build_split(project, output, split, sequence_total, config, rmodel, rnorm, batch):
    source_root = Path("/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed")
    node_cache = "/tmp/slfu_hunan_simvp_frames_a08d895dbe30c7ef"
    simvp_norm = json.loads(Path(config["normalization"]).read_text())
    dataset = ForecastInputs(str(source_root), config["manifest"], simvp_norm, split, cache=node_cache)
    target = np.lib.format.open_memmap(output / f"{split}_history_cot_log1p.npy", mode="w+",
                                       dtype=np.float32, shape=(sequence_total, 2, 8, 1, 16, 16))
    written = np.zeros(sequence_total, dtype=np.bool_)
    mean = torch.as_tensor(simvp_norm["mean"], device="cuda", dtype=torch.float32)[None, None, :, None, None]
    std = torch.as_tensor(simvp_norm["std"], device="cuda", dtype=torch.float32)[None, None, :, None, None]
    rmean = torch.as_tensor(rnorm["mean"], device="cuda", dtype=torch.float32)[None, :, None, None]
    rstd = torch.as_tensor(rnorm["std"], device="cuda", dtype=torch.float32)[None, :, None, None]
    patches = list(config["station_patches"].values())
    for step, (ids, history, geometry) in enumerate(loader(dataset, batch, 4)):
        if (ids < 0).any() or (ids >= sequence_total).any() or written[ids.numpy()].any():
            raise RuntimeError("sequence index overlap/range drift")
        history, geometry = history.cuda(non_blocking=True), geometry.cuda(non_blocking=True)
        physical = history.float() * std + mean
        cropped_agri = torch.stack([physical[:, :, :, r0:r1, c0:c1] for r0, r1, c0, c1 in patches], 1)
        cropped_geometry = torch.stack([geometry[:, :8, :, r0:r1, c0:c1] for r0, r1, c0, c1 in patches], 1)
        rinput = torch.cat((cropped_agri, cropped_geometry), 3).flatten(0, 2)
        if rinput.shape[1:] != (16, 16, 16):
            raise RuntimeError("R input shape drift")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cot = torch.nn.functional.softplus(rmodel.head(rmodel.forward_features((rinput - rmean) / rstd)))
        cot = cot.float().reshape(len(ids), 2, 8, 1, 16, 16)
        if not torch.isfinite(cot).all():
            raise RuntimeError("non-finite historic COT")
        target[ids.numpy()] = cot.cpu().numpy()
        written[ids.numpy()] = True
        if step % 20 == 0:
            target.flush()
            atomic(output / "progress.json", {"state": "BUILDING_HISTORY_COT", "split": split,
                   "written_sequences": int(written.sum()), "sequence_total": sequence_total,
                   "test_used": False})
    target.flush()
    base = project / "data/head_pack_trainval_20260908_v1"
    sequence = np.load(base / "sequence.npy", mmap_mode="r")
    rowsplit = np.load(base / "split.npy", mmap_mode="r")
    required = np.unique(sequence[rowsplit == (0 if split == "train" else 1)])
    if not written[required].all():
        raise RuntimeError("head row references a missing historic trajectory")
    return {"sequence_slots": sequence_total, "source_sequences_written": int(written.sum()),
            "head_sequences_required": int(len(required)),
            "sha256": sha(output / f"{split}_history_cot_log1p.npy")}


def main(project, output, batch):
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite history trajectory sidecar")
    sc = json.loads((project / "configs/s_frozen_hunan_seed42.json").read_text())
    rc = json.loads((project / "configs/r_frozen_repaired_seed42.json").read_text())
    for key in ("checkpoint", "manifest", "normalization"):
        if sha(sc[key]) != sc[key + "_sha256"]:
            raise RuntimeError("SimVP frozen contract drift: " + key)
    if sha(rc["checkpoint"]) != rc["checkpoint_sha256"]:
        raise RuntimeError("COT frozen checkpoint drift")
    checkpoint = torch.load(rc["checkpoint"], map_location="cpu")
    source_file = Path(inspect.getfile(COTUNet))
    if checkpoint["metadata"]["source_code_sha256"] != sha(source_file):
        raise RuntimeError("COT source code contract drift")
    rmodel = COTUNet().cuda().eval().requires_grad_(False)
    rmodel.load_state_dict(checkpoint["model"], strict=True)
    if any(parameter.requires_grad for parameter in rmodel.parameters()):
        raise RuntimeError("R must remain frozen")
    output.mkdir(parents=True)
    report = {"state": "RUNNING_HISTORY_COT_TRAJECTORY", "test_used": False,
              "causality": "each map is retrieved from one of the eight observed AGRI history frames only",
              "R_checkpoint_sha256": rc["checkpoint_sha256"], "R_source_sha256": sha(source_file),
              "splits": {}}
    for split, total in (("train", 27675), ("val", 4258)):
        report["splits"][split] = build_split(project, output, split, total, sc, rmodel,
                                                 checkpoint["metadata"]["norm"], batch)
        atomic(output / "progress.json", report)
    report["state"] = "COMPLETE_HISTORY_COT_TRAJECTORY_SIDECAR"
    report["layout"] = "[sequence, station, historic_frame, 1, 16, 16] log1p(COT)"
    atomic(output / "audit.json", report)
    print(report["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    main(args.project, args.output, args.batch)
