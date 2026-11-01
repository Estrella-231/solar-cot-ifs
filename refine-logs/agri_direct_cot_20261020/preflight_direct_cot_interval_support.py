from __future__ import annotations
import argparse, csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS = ("sili", "zhujia")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--inference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    bm = json.loads((a.bank / "bank_manifest.json").read_text(encoding="utf-8"))
    im = json.loads((a.inference / "inference_manifest.json").read_text(encoding="utf-8"))
    if bm.get("state") != "COMPLETE_DIRECT_COT_FORECAST_ROW_BANK" or bm.get("train_partition") != "H":
        raise RuntimeError("H-only forecast rowbank required")
    if im.get("state") != "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE":
        raise RuntimeError("causal H/V forecast receipt required")
    forbidden = ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input")
    if any(k in bm and bm[k] is not False for k in forbidden) or any(k in im and im[k] is not False for k in forbidden):
        raise RuntimeError("forbidden-source flag is not false")
    if im.get("future_observed_agri_read") is not False or im.get("cpp_read") is not False or im.get("ghi_labels_read") is not False:
        raise RuntimeError("forecast inference receipt contains a forbidden read")
    hdir = a.inference / "H"
    for rel, expected in im["artifact_sha256"].items():
        if rel.startswith("H/"):
            path = hdir / rel[2:]
            if sha(path) != expected:
                raise RuntimeError(f"inference H SHA mismatch: {rel}")
    idx = list(csv.DictReader((hdir / "index.csv").open(encoding="utf-8", newline="")))
    if len(idx) != int(im["sequence_counts"]["H"]):
        raise RuntimeError("H sequence index count mismatch")
    seq_to_i = {r["seq_id"]: int(r["sequence_index_within_split"]) for r in idx}
    last = np.load(hdir / "cot_last.npy", mmap_mode="r", allow_pickle=False)
    pxc = np.load(hdir / "cot_p_xc.npy", mmap_mode="r", allow_pickle=False)
    if last.shape != (len(idx), 2, 16, 1, 16, 16) or pxc.shape != last.shape:
        raise RuntimeError(f"unexpected COT forecast shapes {last.shape} / {pxc.shape}")
    if not np.array_equal(last[:, :, 0], last[:, :, 15]):
        raise RuntimeError("cot_last is not a repeated init-time observation across leads")
    counts = {"sili": 0, "zhujia": 0}
    first_interval = {"sili": 0, "zhujia": 0}
    center_mismatches = 0
    max_center_abs = 0.0
    missing = []
    for station, station_i in zip(STATIONS, (0, 1)):
        d = a.bank / station / "train"
        rows = [json.loads(line) for line in (d / "keys.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != int(bm["counts"][f"{station}/train"]):
            raise RuntimeError(f"row count mismatch: {station}")
        for name in ("keys.jsonl", "agri_raw.npy", "geometry_raw.npy", "ghi.npy", "clear.npy"):
            if sha(d / name) != bm["artifact_sha256"][f"{station}/train/{name}"]:
                raise RuntimeError(f"rowbank SHA mismatch: {station}/{name}")
        bank3_path = d / "cot_raw" / "p_xc.npy"
        bank3 = np.load(bank3_path, mmap_mode="r", allow_pickle=False)
        if sha(bank3_path) != bm["artifact_sha256"][f"{station}/train/cot_raw/p_xc.npy"]:
            raise RuntimeError(f"rowbank COT SHA mismatch: {station}/train/cot_raw/p_xc.npy")
        if bank3.shape != (len(rows), 1, 3, 3):
            raise RuntimeError(f"unexpected station COT rowbank shape: {bank3.shape}")
        for ri, row in enumerate(rows):
            init = datetime.fromisoformat(row["init_time_utc"])
            target = datetime.fromisoformat(row["target_time_utc"])
            lead = int(row["lead_minutes"])
            if target - init != timedelta(minutes=lead):
                raise RuntimeError(f"target/lead mismatch: {station}/{ri}")
            if lead not in range(15, 241, 15):
                raise RuntimeError(f"unexpected lead: {station}/{ri}")
            seq = "hunan_" + (init - timedelta(minutes=105)).strftime("%Y%m%d%H%M")
            if seq not in seq_to_i:
                missing.append(seq)
                continue
            si, li = seq_to_i[seq], lead // 15 - 1
            current = np.asarray(pxc[si, station_i, li, 0], dtype=np.float32)
            previous = np.asarray(last[si, station_i, li, 0] if li == 0 else pxc[si, station_i, li - 1, 0], dtype=np.float32)
            if not np.isfinite(current).all() or not np.isfinite(previous).all() or (current < 0).any() or (previous < 0).any():
                raise RuntimeError(f"nonfinite/negative causal COT endpoint: {station}/{ri}")
            err = float(np.max(np.abs(current[7:10, 7:10] - bank3[ri, 0])))
            max_center_abs = max(max_center_abs, err)
            if not np.allclose(current[7:10, 7:10], bank3[ri, 0], rtol=2e-5, atol=2e-5):
                center_mismatches += 1
            counts[station] += 1
            first_interval[station] += int(li == 0)
    if missing or center_mismatches:
        raise RuntimeError(f"unmatched sequence or COT center mismatch: missing={len(missing)}, mismatch={center_mismatches}")
    receipt = {
        "state": "PASS_DIRECT_COT_INTERVAL_SUPPORT_PREFLIGHT",
        "design": "per-row trapezoid endpoint COT mean; init-time history COT at +15, previous forecast COT thereafter",
        "h_sequence_count": len(idx),
        "row_counts": counts,
        "first_interval_rows_using_init_history_endpoint": first_interval,
        "max_center_3x3_abs_diff_vs_rowbank": max_center_abs,
        "verified_hashes": {
            "bank_manifest_sha256": sha(a.bank / "bank_manifest.json"),
            "inference_manifest_sha256": sha(a.inference / "inference_manifest.json"),
            "H_cot_last_sha256": sha(hdir / "cot_last.npy"),
            "H_cot_p_xc_sha256": sha(hdir / "cot_p_xc.npy"),
        },
        "test_payload_opened": False,
        "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False,
        "cpp_used_as_input": False,
        "ghi_labels_used_as_input": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()
