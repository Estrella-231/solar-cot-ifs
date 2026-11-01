"""Validation-only post-hoc COT-bank swap through the same frozen GHI head.

The head was trained on SimVP-derived COT. This script changes only the future
COT bank to Motion+Source->R; it is an input-distribution diagnostic, not a
matched-head result or an independent test claim.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def metrics(error):
    error = np.asarray(error, dtype=np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise RuntimeError("metrics require nonempty finite errors")
    return {"n": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error))}


def date_bootstrap(truth, simvp, motion, days, repetitions=2000, seed=20260926):
    unique = np.unique(days)
    sse0, sse1, count = [], [], []
    for day in unique:
        take = days == day
        sse0.append(np.square(simvp[take] - truth[take]).sum(dtype=np.float64))
        sse1.append(np.square(motion[take] - truth[take]).sum(dtype=np.float64))
        count.append(int(take.sum()))
    sse0, sse1, count = map(lambda x: np.asarray(x, dtype=np.float64), (sse0, sse1, count))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    denom = np.maximum(count[draws].sum(1), 1)
    d = np.sqrt(sse1[draws].sum(1) / denom) - np.sqrt(sse0[draws].sum(1) / denom)
    return {"unit": "BJT initialization day", "days": int(len(unique)),
            "repetitions": repetitions, "seed": seed,
            "delta_motion_minus_simvp_RMSE_95pct": np.quantile(d, [0.025, 0.975]).tolist(),
            "probability_motion_rmse_lower": float(np.mean(d < 0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solar-root", type=Path, required=True)
    p.add_argument("--head-run", type=Path, required=True)
    p.add_argument("--simvp-bank", type=Path, required=True)
    p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--image-sidecar", type=Path, required=True)
    p.add_argument("--cpp-forecast-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=512)
    a = p.parse_args()
    for path in (a.solar_root, a.head_run, a.simvp_bank, a.history_dir,
                 a.image_sidecar, a.cpp_forecast_root):
        if not path.exists():
            raise RuntimeError(f"missing input: {path}")
    if a.output.exists():
        raise RuntimeError(f"refuse to overwrite: {a.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for faithful replay")

    head_script_dir = a.solar_root / "experiments/cot_dynamics_20260924/scripts"
    sys.path.insert(0, str(head_script_dir))
    from train_geometry24_cot_ghi_v5_20260924 import evaluate, load_bank, load_geometry24_forecast
    from causal_cot_trajectory_solarresnet_v5 import CausalCOTTrajectorySolarResNetV5

    head = json.loads((a.head_run / "complete.json").read_text())
    if head.get("state") != "COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY" or head.get("test_used") is not False:
        raise RuntimeError("frozen GHI head incomplete or test-contaminated")
    head_contract = json.loads((a.head_run / "execution_contract.json").read_text())
    if head_contract.get("future_cot_kind") != "simvp" or head_contract.get("backbone") != "geometry24 SimVP best checkpoint":
        raise RuntimeError("unexpected GHI head training source")

    simvp_contract = json.loads((a.simvp_bank / "complete.json").read_text())
    cpp_contract = json.loads((a.cpp_forecast_root / "contract.json").read_text())
    if (sha(a.simvp_bank / "complete.json") != head_contract["forecast_val_complete_sha256"] or
            simvp_contract.get("state") != "COMPLETE_GEOMETRY24_FORECAST_COT" or
            simvp_contract.get("split") != "val" or simvp_contract.get("test_used") is not False):
        raise RuntimeError("SimVP COT bank differs from head-training validation bank")
    if (cpp_contract.get("split") != "val" or cpp_contract.get("test_used") is not False or
            cpp_contract.get("manifest_sha256") != simvp_contract.get("manifest_sha256")):
        raise RuntimeError("four-backbone COT bank is not on the same validation manifest")
    motion_path = a.cpp_forecast_root / "motion_source" / "forecast_cot_log1p.npy"
    expected = cpp_contract["forecast_cot_sha256"]["motion_source"]
    if sha(motion_path) != expected:
        raise RuntimeError("Motion+Source COT bank hash mismatch")

    base = a.solar_root / "data/head_pack_trainval_20260908_v1"
    input_audit = json.loads((a.image_sidecar / "audit.json").read_text())
    if input_audit.get("test_used") or input_audit.get("state") != "COMPLETE_GEOMETRY24_SOLARRESNET_V5_INPUT_SIDECAR":
        raise RuntimeError("invalid or test-contaminated GHI input sidecar")
    hist_audit = json.loads((a.history_dir / "audit.json").read_text())
    if hist_audit.get("test_used") or hist_audit.get("arm") != "original_R":
        raise RuntimeError("invalid historical COT sidecar")
    if simvp_contract.get("R_checkpoint_sha256") != hist_audit.get("checkpoint_sha256"):
        raise RuntimeError("historical and forecast COT use different frozen R weights")
    if (sha(base / "audit.json") != head_contract["base_pack_audit_sha256"] or
            sha(a.image_sidecar / "audit.json") != head_contract["input_sidecar_audit_sha256"] or
            sha(a.history_dir / "audit.json") != head_contract["history_sidecar_audit_sha256"]):
        raise RuntimeError("frozen head training input provenance mismatch")
    simvp = load_geometry24_forecast(a.simvp_bank, "val")
    history = load_bank(a.history_dir, "history", "val")
    motion_np = np.load(motion_path, mmap_mode="r", allow_pickle=False)
    if simvp.shape != (4258, 2, 16, 1, 16, 16) or history.shape != (4258, 2, 8, 1, 16, 16):
        raise RuntimeError("unexpected GHI trajectory bank shape")
    if motion_np.shape != (4258, 2, 16, 16, 16) or not np.isfinite(motion_np).all():
        raise RuntimeError("Motion+Source COT bank shape/value mismatch")
    motion = torch.from_numpy(np.expand_dims(np.asarray(motion_np), axis=3).copy()).cuda()
    if not bool(torch.isfinite(motion).all()):
        raise RuntimeError("non-finite Motion+Source GHI COT bank")

    keys = ("ghi", "clear", "station", "lead", "sequence", "split", "weight")
    data = {k: torch.from_numpy(np.load(base / f"{k}.npy", allow_pickle=False)).cuda() for k in keys}
    image = np.load(a.image_sidecar / "image.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(a.image_sidecar / "metadata.npy", mmap_mode="r", allow_pickle=False)
    if len(data["split"]) != 637902 or int((data["split"] == 1).sum()) != 99849:
        raise RuntimeError("GHI validation pack row contract drift")
    if image.shape != (637902, 16, 16, 16) or metadata.shape != (637902, 5):
        raise RuntimeError("GHI frozen image/metadata shape drift")
    ids_all = torch.where(data["split"] == 1)[0]
    seq = data["sequence"][ids_all]
    if int(seq.max()) >= 4258 or int(seq.min()) < 0:
        raise RuntimeError("head rows point outside validation sequence bank")
    seq_cpu = data["sequence"].cpu().numpy()
    station_cpu = data["station"].cpu().numpy()
    lead_cpu = data["lead"].cpu().numpy()
    days_by_row = np.empty(len(data["split"]), dtype="U10")
    with (base / "rows.csv").open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if (seq_cpu[i] != int(row["cache_index"]) or
                    station_cpu[i] != int(row["station_index"]) or
                    lead_cpu[i] != int(row["lead_index"])):
                raise RuntimeError("GHI row table does not match pack sequence/station/lead tensors")
            if row["split"] in ("val", "validation"):
                days_by_row[i] = row["init_time_bjt"][:10]
    cuda = torch.device("cuda")
    norm = {k: torch.tensor(head_contract["cot_normalization"][k], device=cuda)
            for k in ("mean", "std")}
    if float(norm["std"]) <= 0:
        raise RuntimeError("invalid frozen train-only COT normalization")

    a.output.mkdir(parents=True)
    result = {"state": "RUNNING", "test_used": False, "scope": "validation-only fixed-head input swap",
              "arms": {}, "paired_date_bootstrap": {}, "prediction_files": {},
              "input_mutation": "future forecast COT bank only"}
    rows_for_save = {}
    for station_id, station in ((0, "Sili"), (1, "Zhujia")):
        ids = torch.where((data["split"] == 1) & (data["station"] == station_id))[0]
        day = days_by_row[ids.cpu().numpy()]
        results = {}
        model = CausalCOTTrajectorySolarResNetV5().cuda().eval().requires_grad_(False)
        ckpt_path = a.head_run / f"{station}_AC_ST_best.pt"
        expected_sha = head["runs"][station]["AC_ST"]["checkpoint_sha256"]
        if sha(ckpt_path) != expected_sha:
            raise RuntimeError(f"head checkpoint hash mismatch: {station}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state.get("station") != station or state.get("group") != "AC_ST":
            raise RuntimeError("wrong station/head checkpoint")
        model.load_state_dict(state["model"], strict=True)
        del state
        banks = {"val": {"history": history, "forecast": simvp}}
        for label, bank in (("SimVP", simvp), ("Motion+Source", motion)):
            banks["val"]["forecast"] = bank
            _, pred_kt = evaluate(model, data, banks, image, metadata, ids, "AC_ST",
                                  a.batch, norm, keep=True)
            truth_kt = data["ghi"][ids].cpu().numpy() / data["clear"][ids].cpu().numpy()
            clear = data["clear"][ids].cpu().numpy()
            truth_ghi = data["ghi"][ids].cpu().numpy()
            pred_ghi = pred_kt.astype(np.float64) * clear
            results[label] = {"kt": pred_kt.astype(np.float64), "ghi": pred_ghi,
                              "truth_kt": truth_kt.astype(np.float64), "truth_ghi": truth_ghi.astype(np.float64),
                              "clear": clear.astype(np.float64)}
        truth_kt = results["SimVP"]["truth_kt"]
        truth_ghi = results["SimVP"]["truth_ghi"]
        lead = data["lead"][ids].cpu().numpy().astype(int)
        pack_ids = ids.cpu().numpy().astype(np.int64)
        result["arms"][station] = {}
        for label in ("SimVP", "Motion+Source"):
            pred = results[label]
            per_lead = {}
            for li in range(16):
                sel = lead == li
                per_lead[str(15 * (li + 1))] = {"kt": metrics(pred["kt"][sel] - truth_kt[sel]),
                                                 "ghi_Wm2": metrics(pred["ghi"][sel] - truth_ghi[sel])}
            short, far = lead <= 7, lead >= 11
            result["arms"][station][label] = {
                "n": int(len(ids)), "short_15_to_120min": {
                    "kt": metrics((pred["kt"] - truth_kt)[short]), "ghi_Wm2": metrics((pred["ghi"] - truth_ghi)[short])},
                "far_180_to_240min": {
                    "kt": metrics((pred["kt"] - truth_kt)[far]), "ghi_Wm2": metrics((pred["ghi"] - truth_ghi)[far])},
                "per_lead": per_lead,
            }
        motion_delta = results["Motion+Source"]["ghi"] - truth_ghi
        simvp_delta = results["SimVP"]["ghi"] - truth_ghi
        result["paired_date_bootstrap"][station] = date_bootstrap(
            truth_ghi[lead >= 11], results["SimVP"]["ghi"][lead >= 11],
            results["Motion+Source"]["ghi"][lead >= 11], day[lead >= 11])
        # Compare change between adjacent forecast leads as a cloud-passage proxy.
        by_key = {}
        for j, pack_row in enumerate(pack_ids):
            seq_id, lead_id = int(seq_cpu[pack_row]), int(lead[j])
            by_key[(seq_id, lead_id)] = j
        deltas = {label: [] for label in ("SimVP", "Motion+Source")}
        far_deltas = {label: [] for label in ("SimVP", "Motion+Source")}
        truth_deltas = []
        far_truth_deltas = []
        event = []
        far_event = []
        for (seq_id, li), j in by_key.items():
            if li == 0 or (seq_id, li - 1) not in by_key:
                continue
            j0 = by_key[(seq_id, li - 1)]
            dt = truth_kt[j] - truth_kt[j0]
            truth_deltas.append(dt)
            event.append(abs(dt) >= 0.1)
            for label in deltas:
                change = results[label]["kt"][j] - results[label]["kt"][j0]
                deltas[label].append(change)
                if li >= 11:
                    far_deltas[label].append(change)
            if li >= 11:
                far_truth_deltas.append(dt)
                far_event.append(abs(dt) >= 0.1)
        td = np.asarray(truth_deltas, dtype=np.float64)
        ev = np.asarray(event, dtype=bool)
        result["arms"][station]["adjacent_lead_kt_change"] = {
            label: {"all": metrics(np.asarray(values) - td),
                   "observed_abs_delta_kt_ge_0.1": metrics((np.asarray(values) - td)[ev])}
            for label, values in deltas.items()}
        result["arms"][station]["adjacent_lead_kt_change"]["event_count"] = int(ev.sum())
        ftd = np.asarray(far_truth_deltas, dtype=np.float64)
        fev = np.asarray(far_event, dtype=bool)
        result["arms"][station]["far_adjacent_lead_kt_change_180_to_240min"] = {
            label: {"all": metrics(np.asarray(values) - ftd),
                   "observed_abs_delta_kt_ge_0.1": metrics((np.asarray(values) - ftd)[fev])}
            for label, values in far_deltas.items()}
        result["arms"][station]["far_adjacent_lead_kt_change_180_to_240min"]["event_count"] = int(fev.sum())
        # Existing saved SimVP predictions are an exact replay witness.
        ref_path = a.head_run / f"{station}_AC_ST_validation_predictions.npz"
        with np.load(ref_path, allow_pickle=False) as ref:
            if not np.array_equal(ref["pack_row"], pack_ids):
                raise RuntimeError("fixed-head baseline pack rows changed")
            replay = float(np.max(np.abs(ref["pred_kt"].astype(np.float64) - results["SimVP"]["kt"])))
        result["arms"][station]["SimVP"]["replay_max_abs_kt"] = replay
        if replay > 0.005:
            raise RuntimeError("SimVP frozen-head replay mismatch")
        prediction_path = a.output / f"{station}_paired_predictions.npz"
        np.savez_compressed(prediction_path, pack_row=pack_ids,
                            init_day=day, lead_minutes=15 * (lead + 1), true_kt=truth_kt,
                            simvp_kt=results["SimVP"]["kt"], motion_source_kt=results["Motion+Source"]["kt"],
                            true_ghi=truth_ghi, simvp_ghi=results["SimVP"]["ghi"],
                            motion_source_ghi=results["Motion+Source"]["ghi"])
        result["prediction_files"][station] = {"path": prediction_path.name, "sha256": sha(prediction_path)}
        print(station, json.dumps(result["arms"][station], allow_nan=False), flush=True)
        atomic_json(a.output / "progress.json", result)
        del model
        torch.cuda.empty_cache()

    result["state"] = "COMPLETE_FIXED_GHI_HEAD_MOTION_SOURCE_COT_SWAP"
    result["head_contract_sha256"] = sha(a.head_run / "execution_contract.json")
    result["head_complete_sha256"] = sha(a.head_run / "complete.json")
    result["motion_cot_sha256"] = sha(motion_path)
    result["source_contracts"] = {
        "simvp_cot_complete_sha256": sha(a.simvp_bank / "complete.json"),
        "four_backbone_cpp_export_contract_sha256": sha(a.cpp_forecast_root / "contract.json"),
        "history_cot_audit_sha256": sha(a.history_dir / "audit.json"),
        "solar_input_audit_sha256": sha(a.image_sidecar / "audit.json"),
        "head_pack_audit_sha256": sha(base / "audit.json"),
        "manifest_sha256": cpp_contract["manifest_sha256"],
    }
    result["cpp_reference_note"] = "GHI labels are direct observations; any CPP-based COT conclusions remain retrieval-reference diagnostics."
    result["limitation"] = "AC_ST head was trained on SimVP forecast COT; Motion+Source score is a post-hoc OOD input-swap diagnostic, not a matched-head model comparison."
    atomic_json(a.output / "complete.json", result)
    print(result["state"], flush=True)


if __name__ == "__main__":
    main()
