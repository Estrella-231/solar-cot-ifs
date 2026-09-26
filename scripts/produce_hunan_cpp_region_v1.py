"""Regional CPP teacher labels using the portable V2 tile coordinates.

No temporal fallback; observations rejected by V2 remain invalid. UTC inputs.
The full input mode is an equivalence witness, not the fast production mode.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BOOT_TIME = time.perf_counter()
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def grid_mapping(grid):
    with np.load(grid, allow_pickle=False) as z:
        lat, lon = z["lat"], z["lon"]
    sy = np.linspace(81, -81, 4051).astype(np.float32)
    sx = np.linspace(24, 186, 4051).astype(np.float32)
    rows = np.abs(sy[:, None] - lat[:, 0]).argmin(axis=0)
    cols = np.abs(sx[:, None] - lon[0, :]).argmin(axis=0)
    err = max(float(np.max(np.abs(lat - sy[rows, None]))),
              float(np.max(np.abs(lon - sx[None, cols]))))
    if lat.shape != (256, 256) or err > 1e-4:
        raise ValueError((lat.shape, err))
    if np.any(np.diff(rows) != 1) or np.any(np.diff(cols) != 1):
        raise ValueError("expected contiguous north-to-south west-to-east Hunan axes")
    ys = [i for i in range(63) if i*64 < rows[-1]+1 and i*64+128 > rows[0]]
    xs = [i for i in range(63) if i*64 < cols[-1]+1 and i*64+128 > cols[0]]
    bounds = (ys[0]*64, ys[-1]*64+128, xs[0]*64, xs[-1]*64+128)
    return rows, cols, ys, xs, bounds, err


def merge_tiles(tiles, ny, nx, fusion):
    stripes = []
    for y in range(ny):
        stripe = tiles[y*nx]
        for x in range(1, nx):
            stripe = fusion(stripe, tiles[y*nx+x], 64, Left_Right=True)
        stripes.append(stripe)
    result = stripes[0]
    for stripe in stripes[1:]:
        result = fusion(result, stripe, 64, Left_Right=False)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--time", action="append")
    p.add_argument("--times-json", type=Path, help="Frozen JSON list of UTC ISO timestamps")
    p.add_argument("--resume", action="store_true", help="Verify identity and output hashes before reuse")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--pad-batch", type=int, default=0,
                   help="Pad independent inference samples to the reference CUDA batch shape")
    p.add_argument("--networks", choices=("two", "four"), default="four")
    p.add_argument("--input-mode", choices=("full", "region", "cached"), default="full")
    p.add_argument("--reference-root", type=Path)
    p.add_argument("--input-reference-dir", type=Path)
    p.add_argument("--mapping-only", action="store_true")
    a = p.parse_args()
    if a.times_json:
        timestamps = json.loads(a.times_json.read_text(encoding="utf-8"))
        if not isinstance(timestamps, list) or not all(isinstance(t, str) for t in timestamps):
            p.error("times JSON must contain a list of UTC timestamp strings")
        a.time = (a.time or []) + timestamps
    if a.time and len(a.time) != len(set(a.time)):
        p.error("duplicate timestamps")
    if a.mapping_only:
        rows, cols, ys, xs, bounds, error = grid_mapping(a.grid)
        a.output.mkdir(parents=True, exist_ok=True)
        atomic_json(a.output / "tile_mapping.json", {
            "grid_sha256": sha(a.grid), "tile_rows": ys, "tile_cols": xs,
            "tile_count": len(ys)*len(xs), "input_bounds": list(bounds),
            "output_rows": [int(rows[0]), int(rows[-1])+1],
            "output_cols": [int(cols[0]), int(cols[-1])+1],
            "coordinate_error_deg": error, "test_used": False})
        print((a.output / "tile_mapping.json").read_text())
        return
    if not a.time:
        p.error("--time is required for production")
    os.environ.setdefault("GLOBAL_CPP_TORCH_THREADS", "4")
    os.environ.setdefault("GLOBAL_CPP_RESOURCE_ROOT", str(a.base))
    sys.path[:0] = [str(a.base), str(a.base / "Code"), str(a.base / "Model/SmaATUnet")]
    import torch
    import netCDF4
    from SmaAtUNet import SmaAt_UNet
    from Retrieval_FY4B_105E import Load_BT, Spec, XMAXMIN
    from Get_ERA_LCCS import Get_ERA_LCCS
    from Satellite_Retrieval import Reorder_ERA_For_FY
    from Data_Preprocess_FD import Img_Fusion
    from Quality_Control import classify_cloud_phase

    started = BOOT_TIME
    rows, cols, ys, xs, bounds, coordinate_error = grid_mapping(a.grid)
    print(json.dumps({"stage": "mapped", "tile_count": len(ys)*len(xs), "bounds": bounds}), flush=True)
    y0, y1, x0, x1 = bounds
    sy, sx = slice(rows[0]-y0, rows[-1]+1-y0), slice(cols[0]-x0, cols[-1]+1-x0)
    a.output.mkdir(parents=True, exist_ok=True)
    names = ("CLP", "CTH", "CER", "COT") if a.networks == "four" else ("CLP", "COT")
    model_paths = dict(zip(("CLP", "CTH", "CER", "COT"), Spec.Model_Paths))
    nets = {}
    for name in names:
        net = SmaAt_UNet(n_channels=16, n_classes=3 if name == "CLP" else 1).to(a.device)
        net.load_state_dict(torch.load(model_paths[name], map_location=a.device, weights_only=True)["model_state_dict"])
        nets[name] = net.eval()
    print(json.dumps({"stage": "models_loaded", "networks": names}), flush=True)
    if a.input_mode == "region":
        from hunan_cpp_region_inputs_v1 import RegionInputs
        region_inputs = RegionInputs(a.base, bounds)
    identity = {"schema": "hunan_cpp_region_v1", "grid_sha256": sha(a.grid),
                "script_sha256": sha(Path(__file__)), "networks": a.networks,
                "input_reader_sha256": sha(Path(__file__).with_name("hunan_cpp_region_inputs_v1.py"))
                if a.input_mode == "region" else None,
                "input_mode": a.input_mode, "weights": {n: sha(model_paths[n]) for n in names},
                "tile_rows": ys, "tile_cols": xs, "input_bounds": list(bounds),
                "coordinate_error_deg": coordinate_error, "fallback": "disabled",
                "test_used": False, "batch": a.batch, "device": a.device}
    identity["pad_batch"] = a.pad_batch
    identity["times_sha256"] = hashlib.sha256(json.dumps(a.time).encode()).hexdigest()
    report = {"identity": identity, "startup_seconds": time.perf_counter()-started, "frames": []}
    if (a.output / "identity.json").exists():
        if not a.resume:
            raise FileExistsError("existing production identity; use --resume to validate")
        previous_identity = json.loads((a.output / "identity.json").read_text())
        if previous_identity != identity:
            raise RuntimeError("resume identity changed; create a separate output directory")
        if (a.output / "report.json").exists():
            report = json.loads((a.output / "report.json").read_text())
    completed = {frame["utc"]: frame for frame in report["frames"]}
    atomic_json(a.output / "identity.json", identity)
    offset = XMAXMIN[1, :, None, None].astype(np.float32)
    scale = (XMAXMIN[0]-XMAXMIN[1])[:, None, None].astype(np.float32)
    for stamp in a.time:
        target = datetime.fromisoformat(stamp)
        tag = target.strftime("%Y%m%d%H%M%S")
        output = a.output / f"{tag}.npz"
        if output.exists():
            frame = completed.get(stamp)
            if not a.resume or frame is None or sha(output) != frame["output_sha256"]:
                raise RuntimeError(f"unverified existing output: {output}")
            if "reference" in frame and not frame["reference"]["passed"]:
                raise RuntimeError("cannot resume through failed equivalence")
            print(json.dumps({"stage": "verified_resume", "utc": stamp}), flush=True)
            continue
        if stamp in completed:
            raise RuntimeError(f"report entry missing its output: {output}")
        start = time.perf_counter()
        if a.input_mode == "full":
            bt = np.asarray(Load_BT(target), dtype=np.float32)
            bt = bt[:, y0:y1, x0:x1].copy()
            tbt = time.perf_counter()
            env = Reorder_ERA_For_FY(Get_ERA_LCCS(target, 81, -81, 186, 24))
            env = np.asarray(env[:, y0:y1, x0:x1], dtype=np.float32).copy()
        elif a.input_mode == "region":
            bt, env = region_inputs.load(target)
            tbt = start + region_inputs.last_timings["bt"]
        else:
            if not a.input_reference_dir:
                raise ValueError("cached witness requires --input-reference-dir")
            with np.load(a.input_reference_dir / f"{tag}.npz", allow_pickle=False) as z:
                cached_feed = z["feed"]
            bt, env = cached_feed[:6], cached_feed[6:]
            tbt = time.perf_counter()
        tenv = time.perf_counter()
        print(json.dumps({"stage": "inputs_loaded", "utc": stamp, "input_mode": a.input_mode,
                          "seconds": tenv-start}), flush=True)
        observation_valid = np.all(np.isfinite(bt), axis=0)[sy, sx]
        feed = np.concatenate((bt, env), axis=0)
        del bt, env
        input_audit = None
        if a.input_reference_dir:
            a.input_reference_dir.mkdir(parents=True, exist_ok=True)
            ipath = a.input_reference_dir / f"{tag}.npz"
            if a.input_mode == "full":
                if ipath.exists():
                    raise FileExistsError(ipath)
                np.savez_compressed(ipath, feed=feed)
            else:
                with np.load(ipath, allow_pickle=False) as z:
                    reference_feed = z["feed"]
                same_mask = bool(np.array_equal(np.isfinite(feed), np.isfinite(reference_feed)))
                differences = np.abs(feed-reference_feed)
                input_audit = {"finite_mask_same": same_mask,
                               "max_abs_by_channel": np.nanmax(differences, axis=(1, 2)).tolist()}
                if not same_mask:
                    raise RuntimeError("regional input validity differs from full input")
        outputs = {n: [] for n in names}
        tile_positions = [(iy*64-y0, ix*64-x0) for iy in ys for ix in xs]
        torch.cuda.reset_peak_memory_stats(a.device)
        with torch.inference_mode():
            for i in range(0, len(tile_positions), a.batch):
                batch = np.stack([feed[:, yy:yy+128, xx:xx+128] for yy, xx in tile_positions[i:i+a.batch]])
                batch = (np.nan_to_num(batch, nan=0., posinf=0., neginf=0.)-offset)/scale
                actual_count = len(batch)
                if a.pad_batch:
                    if actual_count > a.pad_batch:
                        raise ValueError("pad batch smaller than actual batch")
                    batch = np.pad(batch, ((0, a.pad_batch-actual_count), (0, 0), (0, 0), (0, 0)))
                tensor = torch.from_numpy(batch).to(a.device)
                for name, net in nets.items():
                    value = net(tensor).cpu().numpy()[:actual_count]
                    if name != "CLP":
                        value = value[:, 0]
                        lo, hi = {"COT": (0, 100), "CTH": (0, 17.5), "CER": (4, 60)}[name]
                        value = np.clip(value*(hi-lo)+lo, lo, hi)
                    outputs[name].extend(value)
        tinfer = time.perf_counter()
        scores = np.stack([merge_tiles(np.asarray(outputs["CLP"])[:, k], len(ys), len(xs), Img_Fusion)
                           for k in range(3)])[:, sy, sx]
        clp = classify_cloud_phase(scores)
        cot = merge_tiles(np.asarray(outputs["COT"]), len(ys), len(xs), Img_Fusion)[sy, sx]
        cot[np.isfinite(cot)] = np.clip(cot[np.isfinite(cot)], .01, 100)
        cot[clp == 0] = 0
        cot[~observation_valid] = np.nan
        clp[~observation_valid] = np.nan
        valid = observation_valid & np.isfinite(cot) & (cot >= 0) & (cot <= 100) & np.isfinite(clp)
        metrics = {"utc": stamp, "tile_count": len(tile_positions),
                   "seconds": {"bt": tbt-start, "environment": tenv-tbt,
                               "inference": tinfer-tenv},
                   "valid_fraction": float(valid.mean()),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(a.device)}
        if input_audit is not None:
            metrics["input_reference"] = input_audit
        if a.input_mode == "region":
            metrics["sources"] = region_inputs.last_sources
        if a.reference_root:
            reference = a.reference_root / tag[:4] / tag[:8] / f"FY4B_AGRI_{tag}.nc"
            with netCDF4.Dataset(reference) as ds:
                rcot = np.asarray(np.ma.filled(ds["COT"][rows[0]:rows[-1]+1, cols[0]:cols[-1]+1], np.nan))
                rclp = np.asarray(np.ma.filled(ds["CLP"][rows[0]:rows[-1]+1, cols[0]:cols[-1]+1], np.nan))
                rqa = np.asarray(ds["QA"][rows[0]:rows[-1]+1, cols[0]:cols[-1]+1])
            common = valid & np.isfinite(rcot) & (rqa == 0)
            delta = np.abs(cot[common]-rcot[common])
            mask_same = bool(np.array_equal(np.isfinite(cot), np.isfinite(rcot)))
            clp_same = bool(np.array_equal(clp[np.isfinite(rclp)], rclp[np.isfinite(rclp)]))
            close = bool(np.all(delta <= 1e-4+1e-5*np.abs(rcot[common])))
            same_phase = common & (clp == rclp)
            phase_delta = np.abs(cot[same_phase]-rcot[same_phase])
            metrics["reference"] = {"sha256": sha(reference), "common_pixels": int(common.sum()),
                                    "mae": float(delta.mean()) if len(delta) else None,
                                    "max_abs": float(delta.max()) if len(delta) else None,
                                    "mask_same": mask_same, "clp_same": clp_same,
                                    "phase_mismatch_pixels": int(np.sum(clp[np.isfinite(rclp)] != rclp[np.isfinite(rclp)])),
                                    "same_phase_mae": float(phase_delta.mean()) if len(phase_delta) else None,
                                    "passed": mask_same and clp_same and close and bool(common.any())}
        temp = output.with_suffix(".tmp.npz")
        np.savez_compressed(temp, cot=cot.astype(np.float32),
                            clp=np.where(np.isfinite(clp), clp, 255).astype(np.uint8),
                            observation_valid=observation_valid, valid=valid,
                            qa=np.zeros((256, 256), dtype=np.uint8))
        os.replace(temp, output)
        metrics["seconds"]["total"] = time.perf_counter()-start
        metrics["output_sha256"] = sha(output)
        report["frames"].append(metrics)
        atomic_json(a.output / "report.json", report)
        print(json.dumps(metrics), flush=True)
        del feed, outputs, scores, cot, clp
        gc.collect()
        if "reference" in metrics and not metrics["reference"]["passed"]:
            raise RuntimeError("regional equivalence failed; inspect report")
    atomic_json(a.output / "complete.json", {"frames": len(report["frames"]), "test_used": False})


if __name__ == "__main__":
    main()
