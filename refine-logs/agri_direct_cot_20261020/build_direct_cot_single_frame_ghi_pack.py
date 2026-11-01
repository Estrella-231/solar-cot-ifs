"""Normalize an aligned H/V forecast bank for the fixed single-frame GHI heads.

The producer of the bank must have generated predicted AGRI and all COT arms
from causal history-only inputs. This script reads only the declared H and V
directories; it has no paths or code for test/buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("none", "simvp_r", "p_c", "p_xc")
STATIONS = ("sili", "zhujia")
SPLITS = ("train", "val")
SOZ_COS_80 = float(np.cos(np.deg2rad(80.0)))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_array(path: Path, shape_tail: tuple[int, ...], n: int) -> np.ndarray:
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.dtype != np.float32 or x.shape != (n, *shape_tail):
        raise ValueError(f"{path}: expected float32 {(n, *shape_tail)}, got {x.dtype} {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite values in {path}")
    return x


def moments(x: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    # Chunked FP64 accumulation so the normalization is independent of row order.
    channels = x.shape[1]
    total = np.zeros(channels, np.float64)
    second = np.zeros(channels, np.float64)
    count = np.zeros(channels, np.int64)
    for lo in range(0, len(x), 4096):
        b = np.asarray(x[lo:lo + 4096], np.float64)
        if mask is None:
            m = np.ones((len(b), 1, *b.shape[2:]), bool)
        else:
            m = mask[lo:lo + len(b)]
            if m.ndim == b.ndim - 1:
                m = m[:, None]
        for c in range(channels):
            vals = b[:, c][m[:, 0] if m.shape[1] == 1 else m[:, c]]
            total[c] += vals.sum(dtype=np.float64)
            second[c] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
            count[c] += vals.size
    if (count == 0).any():
        raise ValueError("empty normalization support")
    mean = total / count
    var = np.maximum(second / count - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def read_keys(path: Path, n: int) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != n:
        raise ValueError(f"{path}: expected {n} JSONL keys, got {len(lines)}")
    rows = [json.loads(line) for line in lines]
    required = {"station", "init_time_utc", "target_time_utc", "lead_minutes"}
    if any(not required.issubset(r) for r in rows):
        raise ValueError(f"{path}: key fields missing")
    if len({(r["station"], r["init_time_utc"], r["target_time_utc"], int(r["lead_minutes"])) for r in rows}) != n:
        raise ValueError(f"{path}: duplicate sample keys")
    for r in rows:
        t0 = datetime.fromisoformat(r["init_time_utc"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(r["target_time_utc"].replace("Z", "+00:00"))
        if (t1 - t0).total_seconds() != 60 * int(r["lead_minutes"]):
            raise ValueError(f"{path}: lead does not equal target-init interval")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest_path = a.bank / "bank_manifest.json"
    bank = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bank.get("state") != "COMPLETE_DIRECT_COT_FORECAST_ROW_BANK":
        raise ValueError("aligned causal H/V forecast row bank is not complete")
    for field in ("test_payload_opened", "buffer_payload_opened",
                  "future_observed_agri_used_as_input", "cpp_used_as_input"):
        if bank.get(field) is not False:
            raise ValueError(f"bank must certify {field}=false")
    if bank.get("arms") != list(ARMS) or bank.get("stations") != list(STATIONS):
        raise ValueError("station/arm contract differs")
    support_sha = bank.get("split_support_audit_sha256")
    if not support_sha or bank.get("normalization_fit_partition") != "H":
        raise ValueError("bank must bind U/H/V support audit and H-only normalization scope")
    if bank.get("agri_source") != "fold1-SimVP-U-only-forecast":
        raise ValueError("predicted AGRI must come from the frozen fold1 U-only SimVP")
    declared_source_hashes = bank.get("artifact_sha256")
    if not isinstance(declared_source_hashes, dict) or not declared_source_hashes:
        raise ValueError("source bank must bind every H/V numeric array and key file by SHA-256")

    a.output.mkdir(parents=True)
    count_summary, hashes, norm_report, key_hashes = {}, {}, {}, {}
    source_hashes = {}
    loaded_by_station = {}
    for station in STATIONS:
        sdir = a.bank / station
        loaded = {}
        for split in SPLITS:
            d = sdir / split
            ghi = np.load(d / "ghi.npy", mmap_mode="r", allow_pickle=False)
            clear = np.load(d / "clear.npy", mmap_mode="r", allow_pickle=False)
            if ghi.ndim != 1 or clear.shape != ghi.shape or ghi.dtype != np.float32 or clear.dtype != np.float32:
                raise ValueError(f"{station}/{split}: invalid label arrays")
            n = len(ghi)
            if n == 0 or not np.isfinite(ghi).all() or not np.isfinite(clear).all() or (clear <= 0).any() or (ghi < 0).any():
                raise ValueError(f"{station}/{split}: invalid labels")
            keys_path = d / "keys.jsonl"
            keys = read_keys(keys_path, n)
            source_hashes[f"{station}/{split}/keys.jsonl"] = sha(keys_path)
            if any(r["station"].lower() != station for r in keys):
                raise ValueError(f"{station}/{split}: key station mismatch")
            if any(int(r["lead_minutes"]) not in range(15, 241, 15) for r in keys):
                raise ValueError(f"{station}/{split}: lead outside +15..+240 minutes")
            agri = load_array(d / "agri_raw.npy", (13, 3, 3), n)
            geo = load_array(d / "geometry_raw.npy", (2, 3, 3), n)
            cots = {arm: load_array(d / "cot_raw" / f"{arm}.npy", (1, 3, 3), n) for arm in ARMS}
            for rel in ("agri_raw.npy", "geometry_raw.npy", "ghi.npy", "clear.npy",
                        *(f"cot_raw/{arm}.npy" for arm in ARMS)):
                source_hashes[f"{station}/{split}/{rel}"] = sha(d / rel)
            if np.any(np.asarray(cots["none"]) != 0):
                raise ValueError(f"{station}/{split}: no-COT arm must be exact zero")
            for arm in ARMS[1:]:
                if (np.asarray(cots[arm]) < 0).any():
                    raise ValueError(f"negative predicted COT in {station}/{split}/{arm}")
            loaded[split] = {"n": n, "ghi": ghi, "clear": clear, "agri": agri,
                             "geo": geo, "cots": cots, "keys": keys, "keys_path": keys_path}
            key_hashes[f"{station}/{split}"] = sha(keys_path)
        # Split overlap is checked on full sample identity, not just lead or date.
        hkeys = {(r["station"], r["init_time_utc"], r["target_time_utc"], int(r["lead_minutes"]))
                 for r in loaded["train"]["keys"]}
        vkeys = {(r["station"], r["init_time_utc"], r["target_time_utc"], int(r["lead_minutes"]))
                 for r in loaded["val"]["keys"]}
        if hkeys & vkeys:
            raise ValueError(f"{station}: H/V sample keys overlap")
        loaded_by_station[station] = loaded

    if source_hashes != declared_source_hashes:
        missing = sorted(set(source_hashes) ^ set(declared_source_hashes))
        bad = sorted(k for k in set(source_hashes) & set(declared_source_hashes)
                     if source_hashes[k] != declared_source_hashes[k])
        raise ValueError(f"source bank artifact SHA mismatch; missing={missing[:5]}, changed={bad[:5]}")

    for station in STATIONS:
        loaded = loaded_by_station[station]
        tr = loaded["train"]
        ag_mean, ag_std = moments(tr["agri"])
        geo_mean, geo_std = moments(tr["geo"])
        norm_report[station] = {"agri_mean": ag_mean.tolist(), "agri_std": ag_std.tolist(),
                                "geometry_mean": geo_mean.tolist(), "geometry_std": geo_std.tolist(),
                                "cot_by_arm": {}}
        for split in SPLITS:
            d = a.output / station / split
            d.mkdir(parents=True)
            src = loaded[split]
            agri_n = (np.asarray(src["agri"], np.float64) - ag_mean[None, :, None, None]) / ag_std[None, :, None, None]
            geo_n = (np.asarray(src["geo"], np.float64) - geo_mean[None, :, None, None]) / geo_std[None, :, None, None]
            np.save(d / "agri.npy", agri_n.astype(np.float32), allow_pickle=False)
            np.save(d / "geometry.npy", geo_n.astype(np.float32), allow_pickle=False)
            np.save(d / "ghi.npy", np.asarray(src["ghi"], np.float32), allow_pickle=False)
            np.save(d / "clear.npy", np.asarray(src["clear"], np.float32), allow_pickle=False)
            (d / "keys.jsonl").write_bytes(src["keys_path"].read_bytes())
            sun_support = np.asarray(src["geo"][:, 0:1] > SOZ_COS_80)
            for arm in ARMS:
                out_cot = d / "cot" / arm
                out_cot.mkdir(parents=True)
                if arm == "none":
                    cot_norm = np.zeros_like(np.asarray(src["cots"][arm]), dtype=np.float32)
                    params = {"mean_log1p": 0.0, "std_log1p": 1.0, "fit_pixels": 0}
                else:
                    train_cot = loaded["train"]["cots"][arm]
                    train_sun = np.asarray(loaded["train"]["geo"][:, 0:1] > SOZ_COS_80)
                    train_log = np.log1p(np.maximum(np.asarray(train_cot, np.float32), 0.0))
                    vals = train_log[train_sun]
                    if vals.size == 0 or not np.isfinite(vals).all():
                        raise ValueError(f"empty/nonfinite H COT normalization for {station}/{arm}")
                    mean = float(vals.mean(dtype=np.float64))
                    std = max(float(vals.std(dtype=np.float64)), 1e-4)
                    raw_log = np.log1p(np.maximum(np.asarray(src["cots"][arm], np.float32), 0.0))
                    cot_norm = np.where(sun_support, (raw_log - mean) / std, 0.0).astype(np.float32)
                    params = {"mean_log1p": mean, "std_log1p": std, "fit_pixels": int(vals.size),
                              "fit_partition": "H", "outside_soz80_input_value": 0.0}
                np.save(out_cot / f"{split}.npy", cot_norm, allow_pickle=False)
                norm_report[station]["cot_by_arm"][arm] = params
                hashes[f"{station}/{split}/cot/{arm}"] = sha(out_cot / f"{split}.npy")
            hashes[f"{station}/{split}/agri"] = sha(d / "agri.npy")
            hashes[f"{station}/{split}/geometry"] = sha(d / "geometry.npy")
            hashes[f"{station}/{split}/ghi"] = sha(d / "ghi.npy")
            hashes[f"{station}/{split}/clear"] = sha(d / "clear.npy")
            hashes[f"{station}/{split}/keys"] = sha(d / "keys.jsonl")
            count_summary[f"{station}/{split}"] = src["n"]

    manifest = {
        "state": "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_PACK",
        "arms": list(ARMS), "stations": list(STATIONS), "counts": count_summary,
        "train_partition": "H", "validation_partition": "V",
        "normalization_fit_partition": "H",
        "normalization_and_common_key_contract": "PASS_TRAIN_ONLY_NORMALIZATION_AND_COMMON_KEYS",
        "split_support_audit_sha256": support_sha,
        "source_bank_manifest_sha256": sha(manifest_path),
        "source_bank_artifact_sha256": source_hashes,
        "keys_sha256": key_hashes, "array_sha256": hashes,
        "normalization": norm_report,
        "cot_transform": "log1p(raw COT), z-score fit on H pixels with cosSOZ>cos80; out-of-support future-COT input set to zero after normalization",
        "agri_geometry_transform": "per-channel H-only z-score",
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False, "cpp_used_as_input": False,
        "labels": "15-minute target-interval GHI and clear-sky mean; model predicts kt and scoring recovers GHI",
    }
    atomic_json(a.output / "pack_manifest.json", manifest)
    print(json.dumps({"state": manifest["state"], "counts": count_summary,
                      "test_payload_opened": False, "buffer_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
