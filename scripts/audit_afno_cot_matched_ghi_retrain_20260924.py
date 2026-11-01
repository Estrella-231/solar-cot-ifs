"""Independent paired-row audit for matched SimVP-COT vs AFNO-COT head retraining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def day_lookup(rows_path):
    with Path(rows_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([row["init_time_bjt"][:10] for row in rows], dtype="U10"), rows


def load_pred(path, contract_item):
    if sha(path) != contract_item["predictions_sha256"]:
        raise RuntimeError("saved validation prediction hash mismatch: " + str(path))
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in
                ("pack_row", "pred_kt", "observed_ghi", "clear_sky_ghi", "station", "lead")}


def metrics(pack, rows, days):
    truth = pack["observed_ghi"].astype(np.float64)
    pred = pack["pred_kt"].astype(np.float64) * pack["clear_sky_ghi"].astype(np.float64)
    error = pred - truth
    result = {"n": int(len(error)), "rmse_wm2": float(np.sqrt(np.mean(error * error))),
              "bias_wm2": float(np.mean(error)), "init_days": int(len(np.unique(days[pack["pack_row"]])))}
    return error, result


def day_bootstrap(simvp, afno, day_simvp, day_afno, *, repetitions=5000, seed=20260924):
    stations = (0, 1)
    unique = np.unique(np.concatenate((day_simvp, day_afno)))
    sse = {"simvp": np.zeros((2, len(unique))), "afno": np.zeros((2, len(unique)))}
    count = {"simvp": np.zeros((2, len(unique))), "afno": np.zeros((2, len(unique)))}
    for method, pack, day_values in (("simvp", simvp, day_simvp), ("afno", afno, day_afno)):
        truth = pack["observed_ghi"].astype(np.float64)
        prediction = pack["pred_kt"].astype(np.float64) * pack["clear_sky_ghi"].astype(np.float64)
        errors = prediction - truth
        station_ids = pack["station"].astype(int)
        for station in stations:
            for index, day in enumerate(unique):
                selected = (station_ids == station) & (day_values == day)
                if selected.any():
                    sse[method][station, index] = np.square(errors[selected]).sum(dtype=np.float64)
                    count[method][station, index] = int(selected.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    scores = {}
    for method in ("simvp", "afno"):
        station_rmse = []
        for station in stations:
            summed_sse = sse[method][station][draws].sum(axis=1)
            summed_count = count[method][station][draws].sum(axis=1)
            station_rmse.append(np.sqrt(summed_sse / summed_count))
        scores[method] = np.mean(station_rmse, axis=0)
    delta = scores["afno"] - scores["simvp"]
    return {"unit": "paired BJT initialization day, jointly resampling all leads and both stations",
            "days": int(len(unique)), "repetitions": repetitions, "seed": seed,
            "equal_station_delta_afno_minus_simvp_wm2": float(np.mean(delta)),
            "ci95_wm2": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_afno_rmse_lower": float(np.mean(delta < 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite paired audit")
    run = root / "runs/afno_cot_matched_ghi_retrain_seed42_20260924"
    simvp_root, afno_root = run / "simvp_cot", run / "afno_cot"
    simvp_contract = json.loads((simvp_root / "complete.json").read_text())
    afno_contract = json.loads((afno_root / "complete.json").read_text())
    for name, contract in (("SimVP", simvp_contract), ("AFNO", afno_contract)):
        if contract.get("state") != "COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY" or contract.get("test_used") is not False:
            raise RuntimeError(name + " head run incomplete or test contaminated")
        if contract.get("seed") != 42 or contract.get("groups") != ["AC_F", "AC_ST"]:
            raise RuntimeError(name + " training contract mismatch")
    for key in ("mean", "std", "count"):
        if simvp_contract["cot_normalization"][key] != afno_contract["cot_normalization"][key]:
            raise RuntimeError("COT normalization differs: " + key)
    if simvp_contract["cot_normalization"] != afno_contract["cot_normalization"]:
        raise RuntimeError("COT normalization contracts differ")
    for key in ("base_pack_audit_sha256", "input_sidecar_audit_sha256", "history_sidecar_audit_sha256"):
        if simvp_contract[key] != afno_contract[key]:
            raise RuntimeError("shared input provenance differs: " + key)
    rows_path = root / "data/head_pack_trainval_20260908_v1/rows.csv"
    days_by_row, rows = day_lookup(rows_path)
    if len(rows) != 637902:
        raise RuntimeError("base pack row count mismatch")
    station_names = {0: "Sili", 1: "Zhujia"}
    output_payload = {"state": "PASS_PAIRED_ROW_AUDIT_EXPLORATORY_VALIDATION",
                      "test_used": False, "seed": 42,
                      "training_contract": {"groups": ["AC_F", "AC_ST"], "epochs_max": 10,
                                           "batch": 1536, "same_train_only_cot_normalization": True},
                      "AFNO_train_forecasts_in_sample": True,
                      "validation_selection_warning": "Best epoch selected on this validation split; this is exploratory, not independent test evidence.",
                      "groups": {}, "paired_bootstrap": {}}
    for group in ("AC_F", "AC_ST"):
        paired = {}
        group_station_results = {}
        for station_id, station_name in station_names.items():
            filename = station_name + "_" + group + "_validation_predictions.npz"
            sim = load_pred(simvp_root / filename, simvp_contract["runs"][station_name][group])
            afn = load_pred(afno_root / filename, afno_contract["runs"][station_name][group])
            for key in ("pack_row", "observed_ghi", "clear_sky_ghi", "station", "lead"):
                if not np.array_equal(sim[key], afn[key]):
                    raise RuntimeError("paired prediction mismatch: %s %s %s" % (group, station_name, key))
            pack_rows = sim["pack_row"].astype(np.int64)
            if not np.all(np.asarray([rows[i]["split"] == "validation" for i in pack_rows])):
                raise RuntimeError("prediction contains non-validation rows")
            if not np.all(np.asarray([rows[i]["station"].lower() == station_name.lower() for i in pack_rows])):
                raise RuntimeError("station mapping mismatch")
            row_days = days_by_row[pack_rows]
            _, sim_metric = metrics(sim, rows, days_by_row)
            _, afn_metric = metrics(afn, rows, days_by_row)
            group_station_results[station_name] = {
                "simvp": sim_metric, "afno": afn_metric,
                "delta_afno_minus_simvp_wm2": afn_metric["rmse_wm2"] - sim_metric["rmse_wm2"],
                "pair_rows_match": True,
            }
            paired[station_id] = (sim, afn, row_days)
        # Both station heads share the same sorted validation rows by init day and lead.
        sim_joint = {key: np.concatenate([paired[s][0][key] for s in (0, 1)]) for key in
                     ("pack_row", "pred_kt", "observed_ghi", "clear_sky_ghi", "station", "lead")}
        afn_joint = {key: np.concatenate([paired[s][1][key] for s in (0, 1)]) for key in
                     ("pack_row", "pred_kt", "observed_ghi", "clear_sky_ghi", "station", "lead")}
        day_joint = np.concatenate([paired[s][2] for s in (0, 1)])
        bootstrap = day_bootstrap(sim_joint, afn_joint, day_joint, day_joint)
        output_payload["groups"][group] = group_station_results
        output_payload["paired_bootstrap"][group] = bootstrap
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
