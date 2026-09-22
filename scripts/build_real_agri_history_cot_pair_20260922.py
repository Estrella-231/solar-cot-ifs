"""Build paired validation historic COT from the same observed AGRI frames.

Original and continued R use identical FP32 inference, source sequences,
normalization, channel contract, and output shape. Test is never read.
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
        raise RuntimeError("refuse to overwrite paired history COT")
    if not torch.cuda.is_available() or not (1 <= batch <= 256):
        raise RuntimeError("GPU unavailable or invalid batch")
    source = root / "configs/s_frozen_hunan_seed42.json"
    retriever = root / "configs/r_frozen_repaired_seed42.json"
    s_config, r_config = json.loads(source.read_text()), json.loads(retriever.read_text())
    for key in ("checkpoint", "manifest", "normalization"):
        if sha(s_config[key]) != s_config[key + "_sha256"]:
            raise RuntimeError("frozen SimVP source contract drift: " + key)
    if sha(r_config["checkpoint"]) != r_config["checkpoint_sha256"]:
        raise RuntimeError("original R checkpoint hash drift")
    if sha(candidate) != candidate_sha256:
        raise RuntimeError("candidate R checkpoint hash drift")
    original = torch.load(r_config["checkpoint"], map_location="cpu", weights_only=False)
    continued = torch.load(candidate, map_location="cpu", weights_only=False)
    if continued["arm"] != "log_control" or continued["contract"]["initial_checkpoint_sha256"] != r_config["checkpoint_sha256"]:
        raise RuntimeError("candidate source/arm is not real-AGRI continuation")
    if continued["contract"]["input_domain"] != "observed real AGRI for both train and validation; no forecast AGRI":
        raise RuntimeError("candidate trained on a different input domain")
    norm = json.loads((root / "data/cot_repaired_pack_20260907_v1/norm.json").read_text())
    if original["metadata"]["norm"] != norm:
        raise RuntimeError("original R normalization drift")
    output.mkdir(parents=True)
    report = {"state": "RUNNING_PAIRED_REAL_AGRI_HISTORY_COT", "test_used": False,
              "split": "val", "source_sequences": 4258, "forward_precision": "FP32 without autocast",
              "original_R_sha256": r_config["checkpoint_sha256"],
              "continued_R_sha256": candidate_sha256,
              "source_script_sha256": sha(__file__), "batch": batch, "arms": {}}
    for name, checkpoint in (("original_fp32", original), ("continued_fp32", continued)):
        model = COTUNet().cuda().float().eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model"], strict=True)
        directory = output / name
        directory.mkdir()
        metrics = build_split(root, directory, "val", 4258, s_config, model, norm, batch)
        report["arms"][name] = metrics
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        del model
        torch.cuda.empty_cache()
    if report["arms"]["original_fp32"]["head_sequences_required"] != report["arms"]["continued_fp32"]["head_sequences_required"]:
        raise RuntimeError("paired history sequence mismatch")
    report["state"] = "COMPLETE_PAIRED_REAL_AGRI_HISTORY_COT"
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
