#!/usr/bin/env python3
"""Join only train/validation CPP-COT targets to the 16 future frames of each S window."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bjt_from_aux(path: str) -> str:
    stamp = Path(path).name[:12]
    datetime.strptime(stamp, "%Y%m%d%H%M")
    return stamp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--r-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite attempt13 target sidecar: {out}")
    out.mkdir(parents=True)

    status = json.loads((a.r_pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED":
        raise ValueError(f"Unexpected R pack state: {status.get('state')}")
    expected = json.loads((a.r_pack / "SHA256.json").read_text(encoding="utf-8"))
    actual = {name: sha(a.r_pack / name) for name in expected}
    if actual != expected:
        raise ValueError("Corrected attempt12 R pack hash mismatch")

    rrows = rows(a.r_pack / "rows.csv")
    target = np.load(a.r_pack / "target.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(a.r_pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    if target.shape != (len(rrows), 1, 16, 16) or valid.shape != target.shape or valid.dtype != np.bool_:
        raise ValueError(f"Unexpected CPP COT pack arrays: target={target.shape}/{target.dtype}, mask={valid.shape}/{valid.dtype}")
    # The repaired R pack stores physical_COT / cot_scale (100); the frozen
    # R network predicts log1p(physical COT), so restore physical units here.
    if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1.000001):
        raise ValueError("R-pack CPP target must be finite normalized COT in [0,1]")

    by_time: dict[str, tuple[int, str]] = {}
    for i, row in enumerate(rrows):
        key = "".join(ch for ch in row["time_bjt"] if ch.isdigit())
        if key in by_time:
            raise ValueError(f"Duplicate R target_time_bjt={key}")
        by_time[key] = (i, row["split"])

    index_rows = rows(a.prepared / "indexes" / "full.csv")
    manifest = {
        "state": "TRAIN_VAL_CPP_TARGETS_JOINED_TO_SIMVP_FUTURE_FRAMES",
        "reference": "CPP retrieval COT; auxiliary supervision only, not independent truth",
        "target_representation_in_source": "R-pack stores physical COT / 100; sidecar multiplies by cot_scale=100 to recover physical COT in [0,100], then training loss applies log1p",
        "source_cot_scale": 100.0,
        "join_key": "exact BJT target timestamp parsed from aligned future aux frame filename",
        "R_pack_sha256": sha(a.r_pack / "pack_status.json"),
        "R_rows_sha256": sha(a.r_pack / "rows.csv"),
        "R_target_sha256": sha(a.r_pack / "target.npy"),
        "R_mask_sha256": sha(a.r_pack / "mask.npy"),
        "splits": {},
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
    }
    for split in ("train", "val"):
        samples = [r for r in index_rows if r["split"] == split]
        if not samples:
            raise ValueError(f"No {split} sequences in attempt12 verified full index")
        shape = (len(samples), 16, 1, 16, 16)
        tgt_path = out / f"{split}_target_physical.npy"
        mask_path = out / f"{split}_mask.npy"
        tgt = np.lib.format.open_memmap(tgt_path, mode="w+", dtype=np.float32, shape=shape)
        msk = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=shape)
        tgt[:] = 0
        msk[:] = False
        key_rows = []
        matched_frames = 0
        supervised_frames = 0
        cross_split = 0
        lead_pixels = np.zeros(16, dtype=np.int64)
        lead_frames = np.zeros(16, dtype=np.int64)
        for si, sample in enumerate(samples):
            aux_paths = sample["aux_paths"].split("|")
            if len(aux_paths) != 24:
                raise ValueError(f"Expected 24 aligned auxiliary timestamps, got {len(aux_paths)}")
            future_keys = [bjt_from_aux(x) for x in aux_paths[8:24]]
            if len(future_keys) != 16:
                raise ValueError("Future timestamp alignment failure")
            for li, key in enumerate(future_keys):
                item = by_time.get(key)
                if item is None:
                    continue
                ri, rsplit = item
                if rsplit != split:
                    cross_split += 1
                    continue
                matched_frames += 1
                tgt[si, li] = np.asarray(target[ri], dtype=np.float32) * np.float32(100.0)
                msk[si, li] = valid[ri]
                pixels = int(valid[ri].sum())
                lead_pixels[li] += pixels
                lead_frames[li] += int(pixels > 0)
                supervised_frames += int(pixels > 0)
            key_rows.append({
                "sample_index": si,
                "split": split,
                "init_aux_bjt": bjt_from_aux(aux_paths[7]),
                "target_times_bjt": "|".join(future_keys),
                "matched_cpp_frames": sum(k in by_time and by_time[k][1] == split for k in future_keys),
                "valid_cpp_frames": sum(k in by_time and by_time[k][1] == split and valid[by_time[k][0]].any() for k in future_keys),
            })
        tgt.flush(); msk.flush()
        with (out / f"{split}_keys.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
            writer.writeheader(); writer.writerows(key_rows)
        if cross_split:
            raise ValueError(f"{split} sequences join to {cross_split} CPP targets belonging to the opposite split")
        if supervised_frames == 0 or not msk.any():
            raise ValueError(f"No supervised CPP COT support for {split}")
        manifest["splits"][split] = {
            "sequences": len(samples),
            "matched_frames_with_cpp_row": matched_frames,
            "frames_with_valid_cpp_pixels": supervised_frames,
            "valid_cpp_pixels_by_lead": lead_pixels.tolist(),
            "valid_cpp_frames_by_lead": lead_frames.tolist(),
            "target_sha256": sha(tgt_path),
            "mask_sha256": sha(mask_path),
            "keys_sha256": sha(out / f"{split}_keys.csv"),
        }
    if set(r["split"] for r in index_rows if r["split"] not in ("train", "val")):
        raise ValueError("Unexpected non-train/validation rows present in training index")
    manifest["state"] = "PASS_TRAIN_VAL_CPP_TARGET_TIME_ALIGNMENT"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
