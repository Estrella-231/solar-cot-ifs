from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "audit_receipts"
BASE = RECEIPTS / "same_time_normreload_runs_20261017"
AL = RECEIPTS / "al_normreload_raw_20261017" / "runs_maskfix_v2"
COLDSTART_PATH = RECEIPTS / "coldstart_normreload_20261017_v3" / "same_time_coldstart_audit.json"
AL_FEATURE_REPORT = RECEIPTS / "al_normreload_raw_20261017" / "AL_FEATURE_REPORT.json"
AL_FEATURE_KEY_AUDIT = RECEIPTS / "al_normreload_raw_20261017" / "feature_bank_key_audit_20261017.json"
PACK_REPORT = RECEIPTS / "al_normreload_raw_20261017" / "pack_report.json"
OUT = RECEIPTS / "al_normreload_crossrun_audit_20261017.json"
SEEDS = (42, 43, 44)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP", "AL")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    err = p.astype(np.float64) - y.astype(np.float64)
    return {
        "n": int(err.size),
        "rmse_wm2": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
        "mae_wm2": float(np.mean(np.abs(err), dtype=np.float64)),
        "bias_wm2": float(np.mean(err, dtype=np.float64)),
        "r2": float(1.0 - np.sum(err * err) / np.sum((y - np.mean(y)) ** 2)),
    }


def load_bundle(path: Path, station: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {"station", "target_time_bjt", "pred_ghi_wm2", "truth_ghi_wm2", "clear_sky_ghi_wm2"}
        if not required.issubset(z.files):
            raise AssertionError(f"missing keys in {path}: {sorted(required - set(z.files))}")
        out = {k: z[k].copy() for k in required}
    out["station"] = out["station"].astype(str)
    out["target_time_bjt"] = out["target_time_bjt"].astype(str)
    if not np.all(out["station"] == station):
        raise AssertionError(f"station mismatch in {path}")
    n = len(out["target_time_bjt"])
    if any(len(v) != n for v in out.values()):
        raise AssertionError(f"array lengths differ in {path}")
    if len(set(out["target_time_bjt"].tolist())) != n:
        raise AssertionError(f"duplicate target keys in {path}")
    if not np.isfinite(out["pred_ghi_wm2"]).all() or not np.isfinite(out["truth_ghi_wm2"]).all():
        raise AssertionError(f"nonfinite predictions or truth in {path}")
    return out


def main() -> None:
    cold = json.loads(COLDSTART_PATH.read_text(encoding="utf-8"))
    if cold["state"] != "HUBEI_CHECKPOINT_COLDSTART_NORMRELOAD_AUDIT_PASS_TEST_CLOSED":
        raise AssertionError("baseline cold-start audit is not PASS")
    if cold["test_payloads_read"] != 0 or cold["buffer_payloads_read"] != 0:
        raise AssertionError("baseline audit reports test/buffer reads")
    expected_pack = cold["pack_report_sha256"]
    al_feature = json.loads(AL_FEATURE_REPORT.read_text(encoding="utf-8"))
    feature_keys = json.loads(AL_FEATURE_KEY_AUDIT.read_text(encoding="utf-8"))
    pack_report = json.loads(PACK_REPORT.read_text(encoding="utf-8"))
    if sha256(PACK_REPORT) != expected_pack or sha256(PACK_REPORT.parent / "rows.csv") != cold["pack_rows_sha256"]:
        raise AssertionError("downloaded pack report or keys do not match cold-start receipt")
    if al_feature.get("pack_report_sha256") != expected_pack or al_feature.get("test_payloads_read") != 0:
        raise AssertionError("AL feature report does not bind the expected train/validation pack")
    if al_feature.get("ghi_labels_read") is not False or al_feature.get("cpp_payloads_read") != 0:
        raise AssertionError("AL feature extraction crossed the declared data boundary")
    if feature_keys.get("pack_report_sha256") != expected_pack or feature_keys.get("key_sets_equal") is not True:
        raise AssertionError("AL feature key audit does not match the current pack")
    if feature_keys.get("test_payloads_read") != 0 or feature_keys.get("ghi_labels_read_by_feature_extractor") is not False:
        raise AssertionError("AL feature key audit reports forbidden reads")
    baseline_meta_path = BASE / "same_time_seed42_normreload_v1" / "run_metadata.json"
    expected_split = json.loads(baseline_meta_path.read_text(encoding="utf-8"))["split_sha256"]

    cells: dict[str, dict[str, dict[str, float | int]]] = {a: {} for a in ARMS}
    date_keys: dict[str, np.ndarray] = {}
    paired_deltas: dict[str, dict[str, list[float]]] = {k: {} for k in ("AC-A", "AC-AP", "AL-A", "AL-AP", "AC-AL")}
    checked = []
    for seed in SEEDS:
        baseline_meta_path = BASE / f"same_time_seed{seed}_normreload_v1" / "run_metadata.json"
        baseline_meta = json.loads(baseline_meta_path.read_text(encoding="utf-8"))
        if sha256(baseline_meta_path) != cold["run_provenance"][str(seed)]["run_metadata_sha256"]:
            raise AssertionError(f"baseline run metadata not bound by cold-start receipt: seed{seed}")
        if baseline_meta.get("pack_sha256") != expected_pack or baseline_meta.get("split_sha256") != expected_split:
            raise AssertionError(f"baseline run provenance mismatch: seed{seed}")
        run_meta_path = AL / f"seed{seed}" / "run_metadata.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta.get("state") != "HUBEI_GHI_HEAD_AL_SEED_COMPLETE" or run_meta.get("test_payloads_read") != 0:
            raise AssertionError(f"AL seed {seed} incomplete or test gate violated")
        if run_meta.get("pack_report_sha256") != expected_pack:
            raise AssertionError(f"AL seed {seed} pack SHA mismatch")
        if run_meta.get("split_sha256") != expected_split:
            raise AssertionError(f"AL seed {seed} split SHA mismatch")
        if run_meta.get("feature_report_sha256") != sha256(AL_FEATURE_REPORT):
            raise AssertionError(f"AL seed {seed} R feature bank SHA mismatch")
        for station in STATIONS:
            per_arm = {}
            for arm in ARMS:
                if arm == "AL":
                    d = AL / f"seed{seed}" / station
                    bundle_path = d / "AL_validation_predictions_with_keys.npz"
                    pred_path = d / "AL_val_prediction_ghi_wm2.npy"
                    meta = json.loads((d / "AL_validation.json").read_text(encoding="utf-8"))
                    checkpoint = d / "AL_best.pt"
                    if sha256(bundle_path) != meta.get("bundle_sha256"):
                        raise AssertionError(f"AL bundle SHA mismatch: {seed}/{station}")
                    if sha256(pred_path) != meta.get("prediction_sha256"):
                        raise AssertionError(f"AL vector SHA mismatch: {seed}/{station}")
                    if sha256(checkpoint) != meta.get("checkpoint_sha256"):
                        raise AssertionError(f"AL checkpoint SHA mismatch: {seed}/{station}")
                else:
                    d = BASE / f"same_time_seed{seed}_normreload_v1" / station
                    bundle_path = d / f"{arm}_validation_predictions_with_keys.npz"
                    pred_path = d / f"{arm}_val_prediction_ghi_wm2.npy"
                    meta = json.loads((d / f"{arm}_validation.json").read_text(encoding="utf-8"))
                    receipt = [x for x in cold["results"] if x["seed"] == seed and x["station"] == station and x["arm"] == arm]
                    if len(receipt) != 1 or sha256(bundle_path) != receipt[0]["prediction_bundle_sha256"]:
                        raise AssertionError(f"baseline bundle not bound by cold-start receipt: {seed}/{station}/{arm}")
                    if sha256(pred_path) != meta.get("prediction_sha256"):
                        raise AssertionError(f"baseline vector SHA mismatch: {seed}/{station}/{arm}")
                b = load_bundle(bundle_path, station)
                pvec = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.float64).reshape(-1)
                if not np.array_equal(pvec, b["pred_ghi_wm2"].astype(np.float64).reshape(-1)):
                    raise AssertionError(f"vector differs from keyed bundle: {seed}/{station}/{arm}")
                keys = b["target_time_bjt"].tolist()
                if arm == "A":
                    reference = b
                    date_keys[station] = np.array([x[:8] for x in keys])
                else:
                    if keys != reference["target_time_bjt"].tolist():
                        raise AssertionError(f"target keys differ across arms: {seed}/{station}/{arm}")
                    atol = 1e-4 if arm == "AL" else 0.0
                    if not np.allclose(b["truth_ghi_wm2"], reference["truth_ghi_wm2"], rtol=0, atol=atol):
                        raise AssertionError(f"truth differs across arms: {seed}/{station}/{arm}")
                    atol_clear = 5e-5 if arm == "AL" else 0.0
                    if not np.allclose(b["clear_sky_ghi_wm2"], reference["clear_sky_ghi_wm2"], rtol=0, atol=atol_clear):
                        raise AssertionError(f"clear-sky differs across arms: {seed}/{station}/{arm}")
                y = reference["truth_ghi_wm2"].astype(np.float64)
                p = b["pred_ghi_wm2"].astype(np.float64)
                m = metric(y, p)
                if int(meta["val_n"]) != len(y) or int(meta["seed"]) != seed or meta["station"] != station:
                    raise AssertionError(f"AL/baseline metric metadata mismatch: {seed}/{station}/{arm}")
                if abs(m["rmse_wm2"] - float(meta["rmse_wm2"])) > 0.03:
                    raise AssertionError(f"RMSE does not reproduce: {seed}/{station}/{arm}")
                cells[arm][f"{seed}/{station}"] = m
                per_arm[arm] = (y, p)
                checked.append({"seed": seed, "station": station, "arm": arm, "n": len(y), "bundle_sha256": sha256(bundle_path)})
            for left, right in (("AC", "A"), ("AC", "AP"), ("AL", "A"), ("AL", "AP"), ("AC", "AL")):
                yl, pl = per_arm[left]
                yr, pr = per_arm[right]
                if not np.array_equal(yl, yr):
                    raise AssertionError("paired truth mismatch")
                paired_deltas[f"{left}-{right}"][f"{seed}/{station}"] = ((pl - yl) ** 2 - (pr - yr) ** 2).tolist()

    means = {arm: {k: float(np.mean([v[k] for v in cells[arm].values()])) for k in ("rmse_wm2", "mae_wm2", "bias_wm2", "r2")} for arm in ARMS}
    deltas = {f"{a}_minus_{b}": means[a]["rmse_wm2"] - means[b]["rmse_wm2"] for a,b in (("AC","A"),("AC","AP"),("AL","A"),("AL","AP"),("AC","AL"))}
    result = {
        "state": "HUBEI_AL_NORMRELOAD_CROSSRUN_KEYED_RECHECK_PASS_TEST_CLOSED",
        "scope": "fixed validation only; same-time station GHI estimate, not forecast skill or independent test",
        "provenance": {"pack_report_sha256": expected_pack, "split_sha256": expected_split, "feature_report_sha256": run_meta["feature_report_sha256"], "baseline_coldstart_audit_sha256": sha256(COLDSTART_PATH)},
        "n_keyed_bundles_checked": len(checked),
        "n_validation_rows_by_station": {st: len(date_keys[st]) for st in STATIONS},
        "test_payloads_read": 0,
        "metrics_station_equal_seed_equal": means,
        "rmse_deltas_wm2": deltas,
        "claim_boundary": ["All results use validation data involved in model selection.", "AL adds frozen R dec1 features and is a feature-capacity control, not physical COT.", "No future forecast skill claim follows from this same-time experiment."],
        "checked_bundles": checked,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
