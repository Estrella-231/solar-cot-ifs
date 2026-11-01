"""Combine train/val AFNO COT banks and audit split alignment without copying arrays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.train = args.train.resolve()
    args.val = args.val.resolve()
    args.baseline = args.baseline.resolve()
    args.output = args.output.resolve()
    metadata = {}
    for split, source in (("train", args.train), ("val", args.val)):
        contract = json.loads((source / "complete.json").read_text())
        expected_state = ("COMPLETE_AFNO_FORECAST_COT_TRAINVAL" if split == "train"
                          else "COMPLETE_AFNO_FORECAST_COT_VALIDATION")
        if contract.get("state") != expected_state or contract.get("split") != split or contract.get("test_used") is not False:
            raise RuntimeError("invalid AFNO %s bank contract" % split)
        afno_rows = rows(source / "index.csv")
        baseline_rows = rows(args.baseline / split / "index.csv")
        if [row["seq_id"] for row in afno_rows] != [row["seq_id"] for row in baseline_rows]:
            raise RuntimeError("AFNO and frozen SimVP %s sequence order mismatch" % split)
        source_array = source / "forecast_cot_log1p.npy"
        metadata[split] = {"samples": contract["samples"], "sha256": sha(source_array),
                           "index_sha256": sha(source / "index.csv"),
                           "source_complete_sha256": sha(source / "complete.json")}
    if args.output.exists():
        raise RuntimeError("refuse to overwrite AFNO COT bank")
    args.output.mkdir(parents=True)
    for split, source in (("train", args.train), ("val", args.val)):
        (args.output / (split + "_forecast_cot_log1p.npy")).symlink_to(source / "forecast_cot_log1p.npy")
    audit = {
        "state": "COMPLETE_FORECAST_COT_TRAJECTORY_SIDECAR",
        "test_used": False,
        "region": "Hunan",
        "source": "AFNO+Transformer future AGRI -> frozen Hunan repaired R; train and val only",
        "split_order_checked_against": "frozen SimVP forecast sidecar index.csv",
        "afno_train_checkpoint_used_on_train_inputs": True,
        "note": "Training-set AFNO forecasts are in-sample with respect to the AFNO backbone; interpret as exploratory.",
        "splits": metadata,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
