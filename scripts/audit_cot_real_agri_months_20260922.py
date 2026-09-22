"""Read-only month decomposition of frozen and continued real-AGRI COT validation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(root, run):
    pack = root / "data/cot_repaired_pack_20260907_v1"
    with (pack / "rows.csv").open(newline="") as stream:
        val = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    ids = np.array([int(row["index"]) for row in val])
    months = np.array([row["timestamp_bjt"][:7] for row in val])
    report = {"state": "COMPLETE_REAL_AGRI_COT_MONTH_DECOMPOSITION", "test_used": False,
              "note": "The full validation period already selected epoch; these month slices are descriptive, not independent tests.",
              "months": {}}
    paths = {"frozen_R": root / "runs/cot_repaired_seed42_v1/validation_predictions.npz",
             "continued_R": run / "log_control/validation_predictions.npz"}
    saved = {}
    for arm, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["row_indices"], ids):
                raise RuntimeError("validation row mismatch")
            saved[arm] = {key: data[key] for key in ("pred_log1p_cot", "reference_cot", "mask")}
    if not np.array_equal(saved["frozen_R"]["reference_cot"], saved["continued_R"]["reference_cot"]):
        raise RuntimeError("different CPP reference")
    if not np.array_equal(saved["frozen_R"]["mask"], saved["continued_R"]["mask"]):
        raise RuntimeError("different CPP mask")
    for month in sorted(set(months)):
        chosen = months == month
        report["months"][month] = {"images": int(chosen.sum()), "arms": {}}
        for arm, fields in saved.items():
            reference = fields["reference_cot"][chosen].astype(np.float64)
            mask = fields["mask"][chosen]
            predicted = np.expm1(fields["pred_log1p_cot"][chosen].astype(np.float64))
            error = predicted - reference
            strata = {"all": mask, "high_ge30": mask & (reference >= 30),
                      "clear_lt0p1": mask & (reference < .1)}
            report["months"][month]["arms"][arm] = {
                label: {"pixels": int(valid.sum()), "rmse": float(np.sqrt(np.mean(error[valid] ** 2)))}
                for label, valid in strata.items() if valid.any()}
    output = run / "month_decomposition.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite month decomposition")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.run.resolve())
