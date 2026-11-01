"""Verify Hubei old/corrected R lineage and rescore saved validation outputs."""
import csv
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from collect_remote_lineage import sha, time_audit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "refine-logs/hubei_cot_independent_20261014"
NEW = ROOT / "refine-logs/hubei_cloud_probability_20261020"
OLD_R = "1e6c58b97ede89dbeb23789df867076f0413c14c4d23d158be0c2d0e897f7d83"
NEW_R = "8ad4dbd1328325b778308ca4f8a6398dc476043ad41230ed094b9ae4855877c9"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def score_family(name, seed_roots, arms, lineage):
    records = []
    fingerprints = {}
    for seed, base in seed_roots.items():
        for station in ("Niushou", "Wangzhai", "Mizhuang"):
            pair = None
            for arm in arms:
                candidates = [p for p in base.rglob(f"{arm}_validation_predictions*.npz") if p.parent.name == station]
                assert len(candidates) == 1, (name, seed, station, arm, candidates)
                path = candidates[0]
                with np.load(path, allow_pickle=False) as data:
                    truth = data["truth_ghi_wm2"].astype(np.float64)
                    pred = data["pred_ghi_wm2"].astype(np.float64)
                    target = data["target_time_bjt"].astype(str)
                    station_values = data["station"].astype(str)
                    assert np.all(station_values == station)
                    assert np.all(np.isfinite(truth)) and np.all(np.isfinite(pred))
                    cols = [station_values, target]
                    if "init_time_bjt" in data:
                        cols += [data["init_time_bjt"].astype(str), data["lead_minutes"].astype(str)]
                    keys = np.column_stack(cols)
                    assert len(np.unique(keys, axis=0)) == len(keys)
                    if "split" in data:
                        assert np.all(data["split"] == "val")
                if pair is None:
                    pair = (keys, truth)
                else:
                    assert np.array_equal(keys, pair[0]) and np.array_equal(truth, pair[1]), (name, seed, station, arm)
                ref_key = (name, station)
                if ref_key in fingerprints:
                    assert np.array_equal(keys, fingerprints[ref_key][0])
                    assert np.array_equal(truth, fingerprints[ref_key][1])
                else:
                    fingerprints[ref_key] = (keys, truth)
                err = pred - truth
                val = load(path.parent / f"{arm}_validation.json")
                expected_pred_sha = val.get("prediction_bundle_sha256", val.get("prediction_sha256"))
                assert expected_pred_sha == sha(path), (name, path)
                declared_rmse = val.get("val_metrics_fp64", val).get("rmse_wm2")
                rmse = float(np.sqrt(np.mean(err * err)))
                assert abs(rmse - declared_rmse) < 1e-7, (path, rmse, declared_rmse)
                checkpoint = path.parent / f"{arm}_best.pt"
                checkpoint_rechecked = checkpoint.exists()
                if checkpoint_rechecked:
                    assert sha(checkpoint) == val["checkpoint_sha256"]
                records.append({"family": name, "seed": seed, "station": station, "arm": arm,
                                "n": len(truth), "rmse_wm2": rmse, "mae_wm2": float(np.mean(abs(err))),
                                "bias_wm2": float(np.mean(err)), "r_lineage": lineage,
                                "prediction_sha256": sha(path), "checkpoint_sha256": val["checkpoint_sha256"],
                                "checkpoint_file_rehashed": checkpoint_rechecked,
                                "prediction_path": str(path.relative_to(ROOT))})
    summary = {arm: float(np.mean([r["rmse_wm2"] for r in records if r["arm"] == arm])) for arm in arms}
    stations = {station: {arm: float(np.mean([r["rmse_wm2"] for r in records if r["station"] == station and r["arm"] == arm]))
                          for arm in arms} for station in ("Niushou", "Wangzhai", "Mizhuang")}
    support = {station: len(fingerprints[(name, station)][1]) for station in stations}
    return records, {"equal_station_seed_rmse": summary, "station_rmse": stations, "validation_rows_by_station": support,
                     "n_unique_validation_keys": sum(support.values()), "n_saved_prediction_files": len(records),
                     "checkpoint_files_rehashed": sum(r["checkpoint_file_rehashed"] for r in records)}


def main():
    z = load(HERE / "evidence/zjnu_snapshot.json")
    f = load(HERE / "evidence/fd201_snapshot.json")
    checks = {}
    old_ledger = time_audit(OLD / "r_path_audit_20261015_v1/r_paths_trainval.csv")
    assert old_ledger["source_offset_from_correct_utc_hours"] == {"8.0": 3479}
    assert old_ledger["unparsed_rows"] == 0
    checks["old_full_R_manifest_3479_rows_all_offset_plus8h"] = True
    # Metadata-only check of actual CPP BJT source dates, not new payload access.
    def csv_rows(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    assignment_path = OLD / "frozen_split_20261015_v1/day_assignment.csv"
    supervision_path = OLD / "r_path_audit_20261015_v1/r_supervision_trainval.csv"
    assignments = {r["date_BJT"]: r["split"] for r in csv_rows(assignment_path)}
    eligibility = {r["time_bjt"]: r for r in csv_rows(supervision_path)}
    source_splits, buffer_counts = Counter(), Counter()
    for row in csv_rows(OLD / "r_path_audit_20261015_v1/r_paths_trainval.csv"):
        cpp_utc = datetime.strptime(re.search(r"FY4B_AGRI_(\d{14})", row["path_cpp_fixed_v2"]).group(1), "%Y%m%d%H%M%S")
        physical_bjt = cpp_utc + timedelta(hours=8)
        source_split = assignments.get(physical_bjt.strftime("%Y-%m-%d"), "outside_calendar")
        source_splits[f"input_{row['split']}__CPP_source_{source_split}"] += 1
        if source_split == "buffer":
            prefix = row["split"]
            entry = eligibility[row["time_bjt"]]
            buffer_counts[f"{prefix}_buffer_source_rows"] += 1
            buffer_counts[f"{prefix}_buffer_source_supervision_eligible_rows"] += int(entry["r_supervision_eligible"])
            buffer_counts[f"{prefix}_buffer_source_valid_CPP_pixels"] += int(entry["valid_cpp_pixels"])
    old_source_date_audit = {"source_date_split_counts": dict(source_splits), "buffer_source_counts": dict(buffer_counts),
                             "day_assignment_sha256": sha(assignment_path), "supervision_ledger_sha256": sha(supervision_path),
                             "meaning": "Historical CPP source-date contact derived from old ledgers; no new buffer/test payloads opened."}
    assert z["file_hashes"]["old_r_checkpoint"]["sha256"] == OLD_R
    old_rebuilt_bank = z["receipts"]["old_rebuilt_bank_24705"]
    assert old_rebuilt_bank["json"]["frozen_R_checkpoint_sha256"] == OLD_R
    old_pack_path = OLD / "audit_receipts/cot_pack_source_rebuild_20261017_v2/pack_report.json"
    old_pack = load(old_pack_path)
    assert old_pack["r_checkpoint_sha256"] == OLD_R
    assert old_pack["r_bank_report_sha256"] == old_rebuilt_bank["sha256"]
    checks["old_20261017_source_rebuild_still_uses_misaligned_R"] = True
    al = load(OLD / "audit_receipts/al_normreload_raw_20261017/AL_FEATURE_REPORT.json")
    assert al["frozen_R_checkpoint_sha256"] == OLD_R and al["pack_report_sha256"] == sha(old_pack_path)
    checks["old_AL_features_from_misaligned_R"] = True
    forecast_path = OLD / "audit_receipts/forecast_ghi_label_binding_20261017/pack_report.json"
    forecast = load(forecast_path)
    assert forecast["frozen_R_checkpoint_sha256"] == OLD_R
    folds = {}
    for fold in range(5):
        tag = str(fold)
        fd = f["time_ledgers"][f"fold{fold}"]
        source = forecast["oof_fold_source_lineage"][tag]["fold_R_source_hashes"]
        assert fd["unparsed_rows"] == 0 and fd["source_offset_from_correct_utc_hours"] == {"8.0": fd["rows"]}
        assert fd["sha256"] == source["fold_r_manifest_sha256"]
        assert f["file_hashes"][f"fold{fold}/checkpoint"]["sha256"] == source["checkpoint_sha256"]
        assert f["receipts"][f"fold{fold}/run_metadata"]["sha256"] == source["run_metadata_sha256"]
        assert f["receipts"][f"fold{fold}/pack_status"]["sha256"] == source["pack_status_sha256"]
        assert source["source_r_path_manifest_sha256"] == old_ledger["sha256"]
        folds[tag] = {"rows": fd["rows"], "offset_hours": 8, "checkpoint_sha256": source["checkpoint_sha256"]}
    checks["old_forecast_full_R_and_all_five_fold_Rs_misaligned"] = True

    corrected = z["time_ledgers"]["corrected_cpp"]
    assert corrected["source_offset_from_correct_utc_hours"] == {"0.0": 3479} and corrected["unparsed_rows"] == 0
    new_r = z["receipts"]["new_r_metadata"]["json"]
    new_r_pack = z["receipts"]["new_r_pack"]["json"]
    new_bank = z["receipts"]["new_cot_bank"]["json"]
    new_pack = z["receipts"]["new_ghi_pack"]["json"]
    assert new_r["model"] == "hubei_R_station16_correctutc_v1"
    assert new_r["pack_status_sha256"] == z["receipts"]["new_r_pack"]["sha256"]
    assert new_r_pack["cpp_rows_sha256"] == corrected["sha256"]
    assert new_r_pack["cpp_sidecar_sha256"] == load(NEW / "cpp_attempt2_local/independent_audit.json")["sidecar_sha256"]
    for label, filename in (("new_r_target", "target.npy"), ("new_r_mask", "mask.npy"), ("new_r_x_norm", "x_norm.npy")):
        assert z["file_hashes"][label]["sha256"] == new_r["source_pack_hashes"][filename]
    assert z["file_hashes"]["new_r_checkpoint"]["sha256"] == NEW_R == new_bank["frozen_R_checkpoint_sha256"]
    assert new_bank["prediction_sha256"] == z["file_hashes"]["new_cot_bank_array"]["sha256"] == new_pack["cot_bank_prediction_sha256"]
    assert new_pack["cot_bank_report_sha256"] == z["receipts"]["new_cot_bank"]["sha256"]
    assert new_pack["old_time_misaligned_cot_array_read"] is False
    local_new_pack = NEW / "results_remote_attempt2/ghi_fourarm_pack_attempt5/pack_report.json"
    assert sha(local_new_pack) == z["receipts"]["new_ghi_pack"]["sha256"]
    for filename, expected in new_pack["hashes"].items():
        assert z["file_hashes"][f"new_ghi_pack/{filename}"]["sha256"] == expected
    for seed in (42, 43, 44):
        remote_meta = z["receipts"][f"new_ghi_seed{seed}"]
        assert remote_meta["json"]["pack_report_sha256"] == sha(local_new_pack)
        assert sha(NEW / f"results_remote_attempt2/ghi_fourarm_seed{seed}_attempt2/run_metadata.json") == remote_meta["sha256"]
        old_meta = load(OLD / f"audit_receipts/same_time_normreload_runs_20261017/same_time_seed{seed}_normreload_v1/run_metadata.json")
        assert old_meta["pack_sha256"] == sha(old_pack_path)
    checks["corrected_CPP_to_new_R_to_new_COT_bank_to_all_three_GHI_seeds_hash_chain_pass"] = True
    checks["corrected_R_target_and_mask_differ_from_old_R"] = all(
        new_r["source_pack_hashes"][k] != z["receipts"]["old_r_metadata"]["json"]["source_pack_hashes"][k]
        for k in ("target.npy", "mask.npy"))
    assert checks["corrected_R_target_and_mask_differ_from_old_R"]

    families = [
        ("old_same_time_normreload", {s: OLD / f"audit_receipts/same_time_normreload_runs_20261017/same_time_seed{s}_normreload_v1" for s in (42,43,44)}, ("A","AC","AP"), "old_misaligned_R"),
        ("old_forecast_normreload", {s: OLD / f"audit_receipts/forecast_ghi_normreload_seed{s}_20261017" for s in (42,43,44)}, ("FA","FAC"), "old_full_and_five_fold_Rs"),
        ("corrected_same_time_fourarm", {s: NEW / f"results_remote_attempt2/ghi_fourarm_seed{s}_attempt2" for s in (42,43,44)}, ("A","AC","AM","ACM"), "corrected_UTC_R"),
    ]
    metrics = []
    summaries = {}
    for name, roots, arms, lineage in families:
        rows, summary = score_family(name, roots, arms, lineage)
        metrics.extend(rows)
        summaries[name] = summary
    checks["all_81_saved_prediction_files_recomputed_and_paired"] = len(metrics) == 81
    boot = [v for v in load(NEW / "results_fourarm_attempt2/paired_two_day_block_bootstrap.json") if v["stratum"] == "all_validation"]
    for row in boot:
        summary = summaries["corrected_same_time_fourarm"]["equal_station_seed_rmse"]
        assert abs((summary[row["candidate"]]-summary[row["ref"]])-row["delta_rmse_point"]) < 1e-7

    out = {"state": "PASS_HUBEI_R_CPP_TIME_LINEAGE_AND_SAVED_VALIDATION_SCORE_AUDIT",
           "checks": checks, "old_R_sha256": OLD_R, "corrected_R_sha256": NEW_R,
           "old_manifest": old_ledger, "old_forecast_folds": folds, "corrected_CPP_ledger": corrected,
           "historical_old_CPP_actual_BJT_source_date_audit": old_source_date_audit,
           "summaries": summaries, "existing_corrected_same_time_date_block_intervals": boot,
           "score_prediction_files": len(metrics), "test_buffer_payloads_opened": 0, "training_jobs_submitted": 0,
           "scope_limit": "Named local and live ZJNU/FD-201 train/validation assets; no raw CPP re-extraction or new model forward pass.",
           "important_limitations": [
               "Old GHI numbers remain reproducible historical outputs; misaligned-R comparisons cannot establish valid COT physics or general COT forecast value.",
               "The corrected four-arm experiment differs in sample support and A-head capacity from the old three-arm experiment; their gain difference is not an isolated UTC-fix effect.",
               "No corrected-UTC full/fivefold-R future-GHI rerun was identified in the inspected experiment families.",
               "Several corrected R/bank descriptive reference strings still say fixed_v2; actual correct-UTC target/mask and weight hashes were checked instead of trusting these strings.",
               "The historical 22,061-frame old bank is retained locally; it was not found at its old ZJNU mirror path. The rebuilt 24,705-frame old bank was checked live.",
               "Saved validation checkpoints were chosen using validation; these are not independent-test conclusions.",
               "Historical old wrong-time CPP includes physical buffer-date sources despite input-key-based test/buffer=0 flags; these flags cannot certify correct physical CPP date isolation."
           ],
           "evidence_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [HERE/"evidence/zjnu_snapshot.json",HERE/"evidence/fd201_snapshot.json",old_pack_path,forecast_path,local_new_pack]}}
    (HERE / "audit.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (HERE / "rescored_station_seed_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    inventory = []
    for family, status, mode in (("old_same_time_normreload", "AFFECTED_OLD_MISALIGNED_R", "same_time"),
                                 ("old_forecast_normreload", "AFFECTED_OLD_FULL_AND_FIVE_FOLD_R", "future_15_to_240min"),
                                 ("corrected_same_time_fourarm", "CORRECTED_UTC_NEW_TRAINING_VERIFIED", "same_time")):
        s = summaries[family]
        base, candidate = ("FA", "FAC") if family == "old_forecast_normreload" else ("A", "AC")
        inventory.append({"family": family, "status": status, "input_mode": mode,
                          "n_unique_validation_keys": s["n_unique_validation_keys"],
                          "baseline_rmse_wm2": s["equal_station_seed_rmse"][base],
                          "cot_rmse_wm2": s["equal_station_seed_rmse"][candidate],
                          "r_checkpoint": NEW_R if family.startswith("corrected") else OLD_R,
                          "interpretation": "validation only; corrected and old experiments differ in support and control design"})
    inventory += [
        {"family":"old_initial_pack_and_figure", "status":"LEGACY_COT_LINEAGE_NOT_CLOSED", "input_mode":"same_time", "n_unique_validation_keys":2456,
         "baseline_rmse_wm2":"74.36 (historical rounded)", "cot_rmse_wm2":"71.43 (historical rounded)", "r_checkpoint":"not bound in initial pack", "interpretation":"not a corrected-UTC rerun"},
        {"family":"old_AL_and_subgroups", "status":"AFFECTED_OLD_MISALIGNED_R", "input_mode":"same_time", "n_unique_validation_keys":2456,
         "baseline_rmse_wm2":"74.333 (historical audit)", "cot_rmse_wm2":"AL 71.090 (not COT scalar)", "r_checkpoint":OLD_R, "interpretation":"hidden retrieval features and derived COT interpretations affected"},
        {"family":"corrected_future_ghi", "status":"NOT_IDENTIFIED_AS_COMPLETED_IN_INSPECTED_FAMILIES", "input_mode":"future_15_to_240min", "n_unique_validation_keys":"",
         "baseline_rmse_wm2":"", "cot_rmse_wm2":"", "r_checkpoint":"requires corrected full/fold R lineage", "interpretation":"no new job submitted; existing stop request remains in force"},
    ]
    with (HERE / "result_lineage_inventory.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
    pattern=re.compile(r"74\.333|70\.993|74\.33\b|70\.99\b|74\.361|71\.428|77\.150|78\.155|1e6c58b97ede")
    refs=[]
    for folder in [ROOT/"paper",ROOT/"docs",ROOT/"scripts",OLD]:
        for p in folder.rglob("*"):
            if p.suffix not in (".md",".tex",".py") or "code_snapshot" in p.parts:continue
            if p.name.startswith(("PROGRESS","EXPERIMENT_TRACKER","CPP_TIME")):continue
            for num,line in enumerate(p.read_text(encoding="utf-8-sig",errors="replace").splitlines(),1):
                if pattern.search(line):refs.append({"path":str(p.relative_to(ROOT)),"line":num,"excerpt":line[:380],"status":"reference_needs_provenance_notice_or_revision; not a new independent result"})
    (HERE/"old_result_reference_inventory.json").write_text(json.dumps(refs,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(out["state"])
    for name,summary in summaries.items(): print(name,summary["equal_station_seed_rmse"],summary["validation_rows_by_station"])
    print("checks",len(checks),"prediction_files",len(metrics),"document_references",len(refs))


if __name__ == "__main__":
    main()
