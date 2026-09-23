"""Build FP32 historical COT sidecars for a matched old-R/new-R head retrain.

Both retrievers see exactly the same real observed AGRI train/validation
sequences through the audited Hunan frozen-S loader. No test data, future
AGRI, CPP, or GHI labels are read by the retrieval step.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from build_history_cot_trajectory_sidecar_v2 import build_split
from train_cot_repaired import COTUNet


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(root, candidate, candidate_sha256, output, batch):
    root, candidate, output = root.resolve(), candidate.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite paired train/val history COT")
    if not torch.cuda.is_available() or not (1 <= batch <= 256):
        raise RuntimeError("GPU unavailable or invalid batch")

    s_config = json.loads((root / "configs/s_frozen_hunan_seed42.json").read_text())
    r_config = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    for key in ("checkpoint", "manifest", "normalization"):
        if sha(s_config[key]) != s_config[key + "_sha256"]:
            raise RuntimeError("frozen SimVP source contract drift: " + key)
    if sha(r_config["checkpoint"]) != r_config["checkpoint_sha256"]:
        raise RuntimeError("original R checkpoint hash drift")
    if sha(candidate) != candidate_sha256:
        raise RuntimeError("continued R checkpoint hash drift")

    original = torch.load(r_config["checkpoint"], map_location="cpu", weights_only=False)
    continued = torch.load(candidate, map_location="cpu", weights_only=False)
    contract = continued["contract"]
    if continued["arm"] != "log_control" or contract["initial_checkpoint_sha256"] != r_config["checkpoint_sha256"]:
        raise RuntimeError("candidate is not the authorized continuation of original R")
    if contract["input_domain"] != "observed real AGRI for both train and validation; no forecast AGRI":
        raise RuntimeError("candidate R input domain drift")
    norm = json.loads((root / "data/cot_repaired_pack_20260907_v1/norm.json").read_text())
    if original["metadata"]["norm"] != norm:
        raise RuntimeError("original R train-only normalization drift")

    output.mkdir(parents=True)
    report = {
        "state": "RUNNING_REAL_AGRI_HISTORY_COT_TRAINVAL_PAIR",
        "test_used": False,
        "train_sequences": 27675,
        "validation_sequences": 4258,
        "sequence_source": "same frozen-Hunan-S real observed AGRI history loader; 8 frames ending at init",
        "forward_precision": "FP32, autocast disabled",
        "normalization": "same frozen original-R train-only AGRI/geometry normalization for both retrievers",
        "original_R_sha256": r_config["checkpoint_sha256"],
        "continued_R_sha256": candidate_sha256,
        "builder_sha256": sha(__file__),
        "batch": batch,
        "arms": {},
    }
    for name, checkpoint in (("original_R", original), ("continued_R", continued)):
        model = COTUNet().cuda().float().eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model"], strict=True)
        directory = output / name
        directory.mkdir()
        arm = {"checkpoint_sha256": r_config["checkpoint_sha256"] if name == "original_R" else candidate_sha256,
               "splits": {}}
        for split, total in (("train", 27675), ("val", 4258)):
            arm["splits"][split] = build_split(root, directory, split, total, s_config, model, norm, batch)
            report["arms"][name] = arm
            (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        arm["state"] = "COMPLETE_HISTORY_COT_TRAJECTORY_SIDECAR"
        (directory / "audit.json").write_text(json.dumps({
            "state": arm["state"], "test_used": False, "arm": name,
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "builder_sha256": report["builder_sha256"],
            "forward_precision": report["forward_precision"],
            "splits": arm["splits"],
            "layout": "[sequence, station, historic_frame, 1, 16, 16] log1p(COT)",
        }, indent=2) + "\n")
        del model
        torch.cuda.empty_cache()

    old, new = report["arms"]["original_R"]["splits"], report["arms"]["continued_R"]["splits"]
    for split in ("train", "val"):
        if old[split]["head_sequences_required"] != new[split]["head_sequences_required"]:
            raise RuntimeError("old/new R required sequence counts differ")
        if old[split]["sha256"] == new[split]["sha256"]:
            raise RuntimeError("old/new R sidecars unexpectedly identical")
    report["state"] = "COMPLETE_REAL_AGRI_HISTORY_COT_TRAINVAL_PAIR"
    (output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    main(args.root, args.candidate, args.candidate_sha256, args.output, args.batch)
