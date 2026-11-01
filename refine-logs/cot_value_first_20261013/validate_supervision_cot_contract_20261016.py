#!/usr/bin/env python3
"""Source-backed validation of SimVP 8-to-16 provenance for R07 checkpoints.

This does not modify checkpoint metadata or inspect any frame payloads. It pins
the exact training/inference/data/model sources and validates the complete
manifest's metadata-level 24-frame chronology before issuing a receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import torch


EXPECTED = {
    "manifest": "f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2",
    "normalization": "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d",
    "triad_data": "8915b988d9db1b52d01a0c6cab9a7ee16bbde70835ebf6f6bf11732a9ee11502",
    "triad_models": "9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb",
    "simvp": "2e0d280f7962a77f0c9b7e4cc817fed21d142c1436a12eb858e3234485313d16",
    "hunan_data": "f4b4aab55101bde575f474d5daa4a184466c16399b401c2ac67bf666c5f28b91",
    "monthly_trainer": "9a34309b6af4d7debd2f7343c9266e9225df098e96fe699ab31b07c5d9188992",
    "supervision_trainer": "e2e406843edd583be6b796abd277b03d30740dbeb46c4da82c2fef38a75c7ff8",
}
EXPECTED_CKPTS = {
    "mse": "881f249aebf501220f451ca018d193d38a88746289c7a3ac1d16dce86f54665f",
    "temporal": "fea2d406459a3a91c6ce6d11146be63affaaadc16203733d075304aa40338c34",
    "spatial": "24ae31c5fe882588a4afc30495523f003f420f3f1228e1d9f9be9e97ad40c691",
}
CONTRACT = "history_8_then_target_16_v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(path: Path) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}
    val_keys = set()
    total = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in counts:
                raise RuntimeError(f"unexpected split={split}")
            paths = row["data_relpaths"].split("|")
            if len(paths) != 24:
                raise RuntimeError(f"{row['seq_id']}: expected 24 frame paths")
            times = [datetime.strptime(Path(p).stem, "%Y%m%d%H%M") for p in paths]
            if any(b - a != timedelta(minutes=15) for a, b in zip(times, times[1:])):
                raise RuntimeError(f"{row['seq_id']}: paths not 15-minute chronological")
            if times[0] + timedelta(hours=8) != datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S"):
                raise RuntimeError(f"{row['seq_id']}: BJT_start does not match first path")
            counts[split] += 1
            total += 1
            if split == "val":
                key = row["seq_id"]
                if key in val_keys:
                    raise RuntimeError(f"duplicate validation sequence id={key}")
                val_keys.add(key)
    if counts != {"train": 35320, "val": 3140, "test": 3202}:
        raise RuntimeError(f"unexpected manifest counts={counts}")
    return {"rows": total, "split_counts": counts, "val_sequences_unique": len(val_keys),
            "frame_contract": "24 chronological 15-minute paths; indices 0..7 history, 8..23 targets"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--simvp-source", type=Path, required=True)
    p.add_argument("--supervision-source", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="directory containing mse/temporal/spatial/step_4000.pt")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    paths = {
        "manifest": a.manifest,
        "normalization": a.normalization,
        "triad_data": a.simvp_source / "triad_data.py",
        "triad_models": a.simvp_source / "triad_models.py",
        "simvp": a.simvp_source / "simvp.py",
        "hunan_data": a.simvp_source / "hunan_data.py",
        "monthly_trainer": a.simvp_source / "train_monthly_simvp_20261002.py",
        "supervision_trainer": a.supervision_source,
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise RuntimeError(f"{name} source hash mismatch: {hashes[name]} != {expected}")

    manifest_summary = validate_manifest(a.manifest)
    norm = json.loads(a.normalization.read_text(encoding="utf-8"))
    if norm.get("manifest_sha256") != EXPECTED["manifest"] or norm.get("validation_test_used") is not False:
        raise RuntimeError("normalization is not train-only for the pinned manifest")

    ckpt_meta = {}
    for arm, expected in EXPECTED_CKPTS.items():
        ckpt_path = a.checkpoint_dir / arm / "step_4000.pt"
        if sha(ckpt_path) != expected:
            raise RuntimeError(f"{arm} checkpoint hash mismatch")
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = data.get("metadata", {})
        required = {
            "arm": {"mse": "A", "temporal": "B", "spatial": "C"}[arm],
            "manifest_sha256": EXPECTED["manifest"],
            "normalization_sha256": EXPECTED["normalization"],
            "model_code_sha256": EXPECTED["triad_models"],
            "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
            "train_windows": 35320,
            "val_windows": 3140,
            "test_payloads_read": 0,
            "steps": 4000,
            "source_code_sha256": EXPECTED["supervision_trainer"],
        }
        for key, value in required.items():
            if meta.get(key) != value:
                raise RuntimeError(f"{arm} checkpoint metadata mismatch at {key}: {meta.get(key)!r}")
        if meta.get("geometry_contract") not in (None, CONTRACT):
            raise RuntimeError(f"{arm} checkpoint declares conflicting geometry contract")
        ckpt_meta[arm] = {"sha256": expected, "metadata_geometry_field": meta.get("geometry_contract"),
                          "train_windows": meta["train_windows"], "val_windows": meta["val_windows"],
                          "steps": meta["steps"], "test_payloads_read": meta["test_payloads_read"]}

    receipt = {
        "state": "PASS_SOURCE_VERIFIED_GEOMETRY_CONTRACT",
        "geometry_contract": CONTRACT,
        "geometry_contract_source": "pinned training dataset code + pinned inference dataset code + full manifest chronology audit",
        "manifest": manifest_summary,
        "hashes": hashes,
        "checkpoints": ckpt_meta,
        "split": "validation-only inference; test metadata inspected only; no test frame payload read",
        "test_payloads_read": 0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
