"""Build resumable exploratory 256px CPP shards for Hunan train/val frames.

Runs where raw CPP lives. Missing products remain explicit; no test payload or
future cloud field is used as a deployable input. Producer/QA gate remains open.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    for key in ("manifest", "grid", "cpp-root", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--max-new-shards", type=int, default=0)
    a = p.parse_args()
    if a.shard_size < 1 or a.max_new_shards < 0:
        raise ValueError("invalid shard limits")
    with np.load(a.grid, allow_pickle=False) as g:
        target_lat, target_lon = np.asarray(g["lat"]), np.asarray(g["lon"])
    if target_lat.shape != (256, 256) or target_lon.shape != (256, 256):
        raise RuntimeError("Hunan grid contract mismatch")
    frames = {"train": set(), "val": set()}
    with a.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in frames:
                continue
            for rel in row["data_relpaths"].split("|"):
                frames[row["split"]].add(Path(rel).stem)
    overlap = frames["train"] & frames["val"]
    if overlap:
        raise RuntimeError(f"train/val AGRI frames overlap: {sorted(overlap)[:5]}")
    items, missing = [], {"train": [], "val": []}
    for split in ("train", "val"):
        for stem in sorted(frames[split]):
            path = a.cpp_root / stem[:4] / stem[:8] / f"FY4B_AGRI_{stem}00.nc"
            if path.is_file():
                items.append((split, stem, path))
            else:
                missing[split].append(stem)
    if not items:
        raise RuntimeError("no CPP source products")
    a.output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": sha(a.manifest), "grid_sha256": sha(a.grid),
                "script_sha256": sha(__file__), "cpp_root": str(a.cpp_root),
                "shard_size": a.shard_size, "total_frames": len(items),
                "present_train": sum(s == "train" for s, _, _ in items),
                "present_val": sum(s == "val" for s, _, _ in items),
                "missing_train": len(missing["train"]), "missing_val": len(missing["val"])}
    index_path = a.output / "index.json"
    if index_path.exists():
        state = json.loads(index_path.read_text())
        if state["identity"] != identity:
            raise RuntimeError("regional CPP bank resume identity mismatch")
    else:
        state = {"state": "IN_PROGRESS", "test_used": False, "identity": identity,
                 "source_limit": "original CPP retrieval; producer/independent QA not certified",
                 "shards": []}
        atomic(index_path, state)
        atomic(a.output / "missing.json", missing)
    first = len(state["shards"])
    for old in state["shards"]:
        if sha(a.output / old["file"]) != old["sha256"]:
            raise RuntimeError("completed CPP shard drift")
    n_shards = (len(items)+a.shard_size-1)//a.shard_size
    rows = cols = None
    new_count = 0
    for shard_index in range(first, n_shards):
        if a.max_new_shards and new_count >= a.max_new_shards:
            break
        selected = items[shard_index*a.shard_size:(shard_index+1)*a.shard_size]
        cot_maps, valid_maps, names, splits, sources = [], [], [], [], []
        for split, stem, path in selected:
            with netCDF4.Dataset(path) as nc:
                lat = np.asarray(np.ma.filled(nc.variables["LAT"][:], np.nan))
                lon = np.asarray(np.ma.filled(nc.variables["LON"][:], np.nan))
                if rows is None:
                    rows = np.abs(lat[:, None]-target_lat[:, 0][None, :]).argmin(0)
                    cols = np.abs(lon[:, None]-target_lon[0, :][None, :]).argmin(0)
                    if (len(set(rows.tolist())) != 256 or len(set(cols.tolist())) != 256 or
                            np.any(np.abs(np.diff(rows)) != 1) or
                            np.any(np.abs(np.diff(cols)) != 1)):
                        raise RuntimeError("CPP grid axes are not unique contiguous 256px")
                error = max(float(np.max(np.abs(target_lat-lat[rows, None]))),
                            float(np.max(np.abs(target_lon-lon[None, cols]))))
                if error > 1e-4:
                    raise RuntimeError(f"CPP coordinate drift {stem}: {error}")
                raw = nc.variables["COT"][int(rows.min()):int(rows.max())+1,
                                          int(cols.min()):int(cols.max())+1]
                cot = np.asarray(np.ma.filled(raw, np.nan), np.float32)
                if rows[0] > rows[-1]:
                    cot = cot[::-1]
                if cols[0] > cols[-1]:
                    cot = cot[:, ::-1]
            if cot.shape != (256, 256):
                raise RuntimeError(f"CPP crop shape drift {stem}")
            valid = np.isfinite(cot) & (cot >= 0) & (cot <= 100)
            cot_maps.append(np.where(valid, cot, 0).astype(np.float32))
            valid_maps.append(valid)
            names.append(stem)
            splits.append(split)
            stat = path.stat()
            sources.append((int(stat.st_size), int(stat.st_mtime_ns)))
        output_name = f"cpp_{shard_index:04d}.npz"
        output_path = a.output / output_name
        if output_path.exists():
            raise RuntimeError("unregistered output exists: " + output_name)
        temporary = a.output / (output_name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, utc=np.asarray(names, dtype="S12"),
                                split=np.asarray(splits, dtype="S5"),
                                cot=np.stack(cot_maps), valid=np.stack(valid_maps),
                                source_size_mtime=np.asarray(sources, dtype=np.int64))
        temporary.replace(output_path)
        state["shards"].append({"file": output_name, "sha256": sha(output_path),
                                "frames": len(selected), "first_utc": names[0], "last_utc": names[-1],
                                "valid_fraction": float(np.mean(valid_maps))})
        atomic(index_path, state)
        new_count += 1
        print(json.dumps({"shard": shard_index, "frames": len(selected),
                          "completed_frames": sum(x["frames"] for x in state["shards"])}), flush=True)
    if len(state["shards"]) == n_shards:
        state["state"] = "COMPLETE_EXPLORATORY_CPP_BANK"
        atomic(index_path, state)
    print(json.dumps({"state": state["state"], "shards": len(state["shards"]),
                      "total_shards": n_shards}), flush=True)


if __name__ == "__main__":
    main()
