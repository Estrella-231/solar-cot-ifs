"""Independent CPU checkpoint/forecast witness for the paired A/AC pilots.

The FP32-vs-BF16 tolerance and row selection were fixed before inspecting
prediction differences: abs_error <= 0.02 * max(1, abs(saved_prediction)).
No test payload or GPU is accessed. Failure is recorded and never relaxes the
tolerance. All eight candidates must be complete before this script is run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


EXPECTED_HEAD_SHA256 = "a610b06222f667d57d3646eeb986a2783f1d239b4578fc36093ff217d7274148"
RELATIVE_TOLERANCE = 0.02
LEAD_INDICES = (0, 5, 10, 15)
COHORT_RUNS = {
    "original": "head_A_AC_seed42_20260908_v1",
    "qc1": "head_A_AC_qc1_seed42_20260908_v1",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)


def verify(project, report):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    head_path = project / "scripts/train_head_pilot.py"
    require(sha(head_path) == EXPECTED_HEAD_SHA256, "original Head code changed")
    spec = importlib.util.spec_from_file_location("independent_paired_head", head_path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)

    candidates = []
    for cohort, dirname in COHORT_RUNS.items():
        output = project / "runs" / dirname
        require((output / "complete.json").exists(), f"cohort incomplete: {cohort}")
        contract = read_json(output / "execution_contract.json")
        for group in ("A", "AC"):
            for lr in (0.0003, 0.001):
                run = output / f"{group}_lr{lr:g}_seed42"
                complete = read_json(run / "complete.json")
                require(complete["state"] == "COMPLETE_VALIDATION_PILOT", str(run))
                require(complete["test_used"] is False, "unexpected test use")
                require(np.isfinite(complete["station_equal_ghi_rmse"]), "nonfinite selector")
                require(sha(run / "best.pt") == complete["checkpoint_sha256"], "checkpoint hash mismatch")
                require(sha(run / "validation_predictions.npz") == complete["predictions_sha256"], "prediction hash mismatch")
                candidates.append(dict(cohort=cohort, group=group, lr=lr, run=run,
                                       complete=complete, contract=contract))

    pack = project / "data/head_pack_trainval_20260908_v1"
    audit = read_json(pack / "audit.json")
    audit_sha = sha(pack / "audit.json")
    require(audit["state"] == "COMPLETE" and audit["test_used"] is False, "pack incomplete/test payload")
    keys = ("x", "c", "g", "ghi", "clear", "station", "lead", "split")
    arrays = {}
    source_hashes = {}
    for key in keys:
        item = audit["arrays"][key]
        path = pack / item["name"]
        digest = sha(path)
        require(digest == item["sha256"], f"source array hash mismatch: {key}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(list(array.shape) == item["shape"], f"source shape mismatch: {key}")
        require(str(array.dtype) == item["dtype"], f"source dtype mismatch: {key}")
        arrays[key] = array
        source_hashes[key] = digest
        print("SOURCE_VERIFIED", key, flush=True)
    val = np.flatnonzero(arrays["split"] == 1)
    require(len(val) == 99849, "validation count changed")
    require(np.isin(arrays["split"], [0, 1]).all(), "unexpected split payload")
    selected_positions = []
    for station in (0, 1):
        for lead in LEAD_INDICES:
            positions = np.flatnonzero((arrays["station"][val] == station) &
                                       (arrays["lead"][val] == lead))
            require(len(positions) > 0, "missing station/lead stratum")
            selected_positions.append(int(positions[0]))
    ids = val[np.asarray(selected_positions)]
    require(len(np.unique(ids)) == 8, "duplicate witness rows")
    report.update(pack_audit_sha256=audit_sha, source_array_sha256=source_hashes,
                  validation_count=len(val), witness_pack_rows=ids.tolist(),
                  witness_station=arrays["station"][ids].tolist(),
                  witness_lead_minutes=((arrays["lead"][ids] + 1) * 15).tolist())

    view_path = project / "data/head_qc_view_20260908_v1/view.json"
    view = read_json(view_path)
    require(view["state"] == "COMPLETE_QC_SENSITIVITY_VIEW", "QC view incomplete")
    require(view["pack_audit_sha256"] == audit_sha, "QC pack binding mismatch")
    selected = []
    for cohort in COHORT_RUNS:
        for group in ("A", "AC"):
            arms = [c for c in candidates if c["cohort"] == cohort and c["group"] == group]
            selected.append(min(arms, key=lambda c: (c["complete"]["station_equal_ghi_rmse"], c["lr"])))

    report["candidate_checks"] = []
    for candidate in candidates:
        contract = candidate["contract"]
        require(contract["pack_audit_sha256"] == audit_sha, "candidate pack binding mismatch")
        require(contract["batch"] == 2048, "batch changed")
        with np.load(candidate["run"] / "validation_predictions.npz", allow_pickle=False) as saved:
            require(np.array_equal(saved["pack_row"], val), "validation pack_row mismatch")
            for saved_key, pack_key in (("observed_ghi", "ghi"), ("clear_sky_ghi", "clear"),
                                        ("station", "station"), ("lead", "lead")):
                require(np.array_equal(saved[saved_key], arrays[pack_key][val]),
                        f"validation truth/key mismatch: {saved_key}")
            require(np.isfinite(saved["pred_kt"]).all(), "nonfinite saved predictions")
        report["candidate_checks"].append(dict(cohort=candidate["cohort"], group=candidate["group"],
             lr=candidate["lr"], same_validation_rows_and_truth=True,
             checkpoint_sha256=candidate["complete"]["checkpoint_sha256"],
             predictions_sha256=candidate["complete"]["predictions_sha256"]))

    report["selected_checkpoint_witnesses"] = []
    for candidate in selected:
        cohort, group = candidate["cohort"], candidate["group"]
        ck = torch.load(candidate["run"] / "best.pt", map_location="cpu")
        metadata = ck["metadata"]
        require(metadata == candidate["contract"], "checkpoint execution contract mismatch")
        require(ck["group"] == group and ck["lr"] == candidate["lr"] and ck["batch"] == 2048,
                "checkpoint arm mismatch")
        require(ck["seed"] == 42 and metadata["test_used"] is False, "seed/test contract mismatch")
        require(ck["next_epoch"] - 1 == candidate["complete"]["best_epoch_0based"], "best epoch mismatch")
        head_binding = metadata["imported_head_sha256"] if cohort == "qc1" else metadata["code_sha256"]
        require(head_binding == EXPECTED_HEAD_SHA256, "checkpoint Head code mismatch")
        tensors = {key: torch.from_numpy(np.array(arrays[key][ids], copy=True))
                   for key in ("x", "c", "g", "station", "lead")}
        if cohort == "qc1":
            require(metadata["view_sha256"] == sha(view_path), "checkpoint QC view mismatch")
            require(metadata["input_transform"] == view["transforms_in_base_pack_normalized_coordinates"],
                    "checkpoint affine transform mismatch")
            for key, transform in metadata["input_transform"].items():
                mean = torch.tensor(transform["mean"], dtype=torch.float32)
                std = torch.tensor(transform["std"], dtype=torch.float32)
                if key == "x":
                    tensors[key].sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                elif key == "g":
                    tensors[key][:, :2].sub_(mean).div_(std)
                else:
                    tensors[key].sub_(mean).div_(std)
        model = source.Head(group == "AC").cpu().float().eval()
        model.load_state_dict(ck["model"], strict=True)
        with torch.no_grad():
            pred = model(*(tensors[key] for key in ("x", "c", "g", "station", "lead"))).numpy()
        with np.load(candidate["run"] / "validation_predictions.npz", allow_pickle=False) as saved:
            reference = np.array(saved["pred_kt"][selected_positions], dtype=np.float64)
        absolute_error = np.abs(pred.astype(np.float64) - reference)
        tolerance = RELATIVE_TOLERANCE * np.maximum(1.0, np.abs(reference))
        passed = bool(np.isfinite(pred).all() and (absolute_error <= tolerance).all())
        witness = dict(cohort=cohort, group=group, selected_lr=candidate["lr"],
                       selection_score=candidate["complete"]["station_equal_ghi_rmse"],
                       best_epoch_0based=candidate["complete"]["best_epoch_0based"],
                       checkpoint_sha256=candidate["complete"]["checkpoint_sha256"],
                       cpu_fp32_pred_kt=pred.tolist(), saved_gpu_bf16_pred_kt=reference.tolist(),
                       absolute_errors=absolute_error.tolist(), per_row_tolerances=tolerance.tolist(),
                       max_absolute_error=float(absolute_error.max()),
                       max_error_over_tolerance=float((absolute_error / tolerance).max()), passed=passed)
        report["selected_checkpoint_witnesses"].append(witness)
        print("CHECKPOINT_WITNESS", json.dumps(witness), flush=True)
        require(passed, f"fixed CPU/BF16 tolerance failed: {cohort}/{group}")
    report["state"] = "PASS_4_SELECTED_CPU_CHECKPOINT_WITNESSES"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to overwrite an existing witness report")
    report = dict(state="IN_PROGRESS", test_used=False, gpu_used=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(__file__), expected_head_sha256=EXPECTED_HEAD_SHA256,
                  row_selection="first validation row for each station 0/1 and lead index 0/5/10/15",
                  candidate_selection="lowest saved station-equal validation GHI RMSE per cohort/group; tie uses lower LR",
                  fixed_tolerance="abs(CPU_FP32 - saved_GPU_BF16) <= 0.02 * max(1, abs(saved_GPU_BF16))",
                  tolerance_timing="frozen before inspecting any witness prediction differences")
    try:
        verify(args.project, report)
    except Exception as error:
        report.update(state="FAIL", error_type=type(error).__name__, error=str(error),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save_report(args.output, report)
        raise
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.output, report)
    print(report["state"], flush=True)


if __name__ == "__main__":
    main()
