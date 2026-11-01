#!/usr/bin/env python3
"""Create audited PBS-path copies of frozen per-fold SimVP OOF inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import runpy
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[2, 3, 4])
    args = ap.parse_args()
    stage = args.stage_root.resolve()
    control = stage / "control"
    source_stage = "/tmp/zjnu_hubei_simvp_stage_20261016"
    prepared = []
    for fold in args.folds:
        if fold not in range(5):
            raise SystemExit(f"INVALID_FOLD {fold}")
        fold_dir = control / f"fold_{fold}"
        source_index = fold_dir / "simvp_trainval_index.csv"
        source_norm = fold_dir / "simvp_train_only_norm.npz"
        source_config = control / "configs" / f"hbagri_simvp_oof_fold{fold}_seed42_20261016.py"
        for p in (source_index, source_norm, source_config):
            if not p.is_file():
                raise SystemExit(f"MISSING_FROZEN_FOLD_INPUT {p}")
        cfg = runpy.run_path(str(source_config))
        if cfg.get("data_root") != source_stage:
            raise SystemExit(f"UNEXPECTED_SOURCE_DATA_ROOT fold={fold}: {cfg.get('data_root')}")
        expected_suffix = f"/fold_{fold}/simvp_trainval_index.csv"
        if not str(cfg.get("index_csv", "")).endswith(expected_suffix):
            raise SystemExit(f"UNEXPECTED_SOURCE_INDEX fold={fold}: {cfg.get('index_csv')}")
        for key, value in {
            "dataname": "hbagri", "model_type": "gSTA", "pre_seq_length": 8,
            "aft_seq_length": 16, "total_length": 24, "batch_size": 4,
            "epoch": 50, "seed": 42,
        }.items():
            if cfg.get(key) != value:
                raise SystemExit(f"FROZEN_CONFIG_MISMATCH fold={fold} {key}={cfg.get(key)!r}")

        rebased_index = fold_dir / "simvp_trainval_index_pbs.csv"
        n_rows = 0
        with source_index.open(newline="", encoding="utf-8") as src, rebased_index.open(
            "w", newline="", encoding="utf-8"
        ) as dst:
            reader = csv.DictReader(src)
            if reader.fieldnames != ["split", "data_paths", "aux_paths"]:
                raise SystemExit(f"BAD_INDEX_HEADER fold={fold} {reader.fieldnames}")
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["split"] not in {"train", "val"}:
                    raise SystemExit(f"TEST_OR_BUFFER_ROW_IN_FOLD_INDEX fold={fold}")
                for col in ("data_paths", "aux_paths"):
                    paths = row[col].split("|")
                    if len(paths) != 24:
                        raise SystemExit(f"BAD_SEQUENCE_LENGTH fold={fold} row={n_rows}")
                    new_paths = []
                    for old in paths:
                        prefix = source_stage.rstrip("/") + "/"
                        if not old.startswith(prefix):
                            raise SystemExit(f"UNEXPECTED_DATA_PATH fold={fold}: {old}")
                        new = str(stage).rstrip("/") + "/" + old[len(prefix):]
                        if not Path(new).is_file():
                            raise SystemExit(f"MISSING_REBASED_TARGET fold={fold}: {new}")
                        new_paths.append(new)
                    row[col] = "|".join(new_paths)
                writer.writerow(row)
                n_rows += 1

        source_text = source_config.read_text(encoding="utf-8")
        new_config = fold_dir / f"hbagri_simvp_oof_fold{fold}_seed42_20261016_pbs.py"
        new_text, root_n = re.subn(
            r"^data_root\s*=\s*['\"]" + re.escape(source_stage) + r"['\"]\s*$",
            f"data_root = {str(stage)!r}", source_text, flags=re.MULTILINE,
        )
        new_text, index_n = re.subn(
            r"^index_csv\s*=\s*['\"][^'\"]+/fold_" + str(fold) + r"/simvp_trainval_index\.csv['\"]\s*$",
            f"index_csv = {str(rebased_index)!r}", new_text, flags=re.MULTILINE,
        )
        if root_n != 1 or index_n != 1:
            raise SystemExit(f"CONFIG_REBASE_FAILED fold={fold} root={root_n} index={index_n}")
        new_config.write_text(new_text, encoding="utf-8")
        new_cfg = runpy.run_path(str(new_config))
        if new_cfg.get("data_root") != str(stage) or new_cfg.get("index_csv") != str(rebased_index):
            raise SystemExit(f"DERIVED_CONFIG_RELOAD_FAILED fold={fold}")
        prepared.append({
            "fold": fold,
            "source_config_sha256": sha256(source_config),
            "pbs_config": str(new_config),
            "pbs_config_sha256": sha256(new_config),
            "source_index_sha256": sha256(source_index),
            "pbs_index": str(rebased_index),
            "pbs_index_sha256": sha256(rebased_index),
            "trainval_sequences": n_rows,
            "norm": str(source_norm),
            "norm_sha256": sha256(source_norm),
        })
    receipt = {
        "status": "PBS_FOLD_INPUTS_REBASED_TEST_CLOSED",
        "stage_root": str(stage),
        "source_root": source_stage,
        "folds": prepared,
    }
    out = control / "pbs_fold_input_receipt.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
