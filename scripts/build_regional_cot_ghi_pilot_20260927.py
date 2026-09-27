"""Build a paired regional GHI pilot and three future-COT baselines.

Deployable arrays use only observed history and fixed SimVP forecasts.
Future observed AGRI is loaded separately for an explicitly nondeployable
R-region oracle; CPP is read only for validation-reference diagnostics.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

STATIONS = ((163, 148), (73, 150))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def stratified_indices(rows, available, count):
    groups = defaultdict(list)
    for i in sorted(available):
        groups[rows[i]["BJT_start"][:7]].append(i)
    # All months represented; deterministic evenly spaced ranks, no CPP/GHI magnitude selection.
    selected = []
    quota = max(1, count // len(groups))
    for month in sorted(groups):
        ids = groups[month]
        selected += [ids[k] for k in np.unique(np.linspace(0, len(ids)-1, min(quota, len(ids))).astype(int))]
    remaining = sorted(set(available)-set(selected))
    need = min(count-len(selected), len(remaining))
    if need > 0:
        selected += [remaining[k] for k in np.unique(np.linspace(0, len(remaining)-1, need).astype(int))]
    return sorted(selected[:count])


def motion_shift(history):
    # C13 is index 10 in C01-C06,C09-C15. Phase correlation estimates last-step shift.
    a, b = history[-2, 10], history[-1, 10]
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    a = np.nan_to_num(a-a.mean())*window; b = np.nan_to_num(b-b.mean())*window
    cross = np.fft.fft2(b)*np.conj(np.fft.fft2(a))
    corr = np.fft.ifft2(cross/np.maximum(np.abs(cross), 1e-12)).real
    shift = np.array(np.unravel_index(corr.argmax(), corr.shape), dtype=float)
    shift[shift > 128] -= 256
    return shift


def transport(field, shift, lead):
    # Source is x - lead*shift; out-of-domain pixels remain missing, never wrap around.
    yy, xx = torch.meshgrid(torch.arange(256, device=field.device), torch.arange(256, device=field.device), indexing="ij")
    sy, sx = yy-lead*float(shift[0]), xx-lead*float(shift[1])
    grid = torch.stack((2*sx/255-1, 2*sy/255-1), -1).float()[None]
    moved = F.grid_sample(torch.expm1(field)[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    valid = (sy >= 0) & (sy <= 255) & (sx >= 0) & (sx <= 255)
    return torch.log1p(moved), valid


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for key in ("solar-root", "triad-root", "hunan-source", "data-root", "r-run", "r-cache", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--train-sequences", type=int, default=512)
    parser.add_argument("--val-sequences", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite pilot")
    solar = args.solar_root
    sc = json.loads((solar/"configs/s_frozen_hunan_seed42.json").read_text())
    assert sha(sc["manifest"]) == sc["manifest_sha256"] and sha(sc["normalization"]) == sc["normalization_sha256"]
    assert sha(args.data_root/"grid_static.npz") == sc["grid_sha256"]
    bank_code = solar/"experiments/cot_dynamics_20260924/scripts/build_geometry24_forecast_cot_20260924.py"
    bank = load(bank_code, "existing_geometry24_causal_bank")
    model_code = args.triad_root/"geometry24_v2/triad_models.py"
    checkpoint = args.triad_root/"geometry24_v2/runs/simvp_geometry24_formal_e12_v1/best.pt"
    assert (checkpoint.parent/"training_complete.json").exists()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sm = state["metadata"]
    assert sm["geometry_contract"] == "history_8_then_target_16_v2" and sm["region"] == "Hunan"
    assert sm["model"] == "simvp" and sm["model_code_sha256"] == sha(model_code)
    assert sm["manifest_sha256"] == sc["manifest_sha256"] and sm["normalization_sha256"] == sc["normalization_sha256"]
    s = load(model_code, "region_simvp").SimVPBackbone(args.hunan_source).cuda().eval().requires_grad_(False)
    s.load_state_dict(state["model"])
    rs = json.loads((args.r_run/"status.json").read_text())
    assert rs["state"] == "COMPLETE_EXPLORATORY_REGIONAL_R" and rs["test_used"] is False
    rstate = torch.load(args.r_run/"best.pt", map_location="cpu", weights_only=False)
    rcode = solar/"scripts/train_cot_repaired.py"
    assert rstate["metadata"]["r_arch_sha256"] == sha(rcode)
    rnorm = json.loads((args.r_cache/"norm.json").read_text())
    assert sha(args.r_cache/"norm.json") == rstate["metadata"]["normalization_sha256"]
    r = load(rcode, "region_retriever").COTUNet().cuda().eval().requires_grad_(False)
    r.load_state_dict(rstate["model"])
    rm = torch.tensor(rnorm["mean"], device="cuda")[None, :, None, None]
    rst = torch.tensor(rnorm["std"], device="cuda")[None, :, None, None]
    snorm = json.loads(Path(sc["normalization"]).read_text())
    sys.path.insert(0, str(args.hunan_source))
    from hunan_data import load_frame
    base = solar/"data/head_pack_trainval_20260908_v1"
    base_audit = json.loads((base/"audit.json").read_text())
    assert base_audit["test_used"] is False
    labels = {name: np.load(base/(name+".npy"), mmap_mode="r")
              for name in ("sequence", "split", "station", "lead", "ghi", "clear", "weight")}
    cpp_index = json.loads((args.r_cache/"index.json").read_text())
    assert cpp_index["test_used"] is False and cpp_index["identity"]["manifest_sha256"] == sc["manifest_sha256"]
    cpp_lookup = {}
    for shard in cpp_index["shards"]:
        d = args.r_cache/shard["directory"]
        for rec in json.loads((d/"records.json").read_text()):
            if rec["split"] == "val" and rec["included_for_R"]:
                cpp_lookup[rec["utc"]] = (d, rec["array_row"])
    args.output.mkdir(parents=True)
    started = time.monotonic()
    sums = defaultdict(lambda: np.zeros(3, float))
    profiles = []
    simvp_profile = []
    simvp_batch = None
    rbatch = 128  # FP32 inference: profile before any higher memory setting.
    def retrieve(raw):
        z = torch.nan_to_num((raw-rm)/rst, nan=0., posinf=0., neginf=0.)
        return torch.cat([r(part).float() for part in z.split(rbatch)])
    def crops(value):
        return torch.stack([value[..., row-32:row+32, col-32:col+32] for row, col in STATIONS], 0)
    selections = {}
    init_days = {}
    for split, value, limit in (("train", 0, args.train_sequences), ("val", 1, args.val_sequences)):
        ds = bank.CausalInputs(args.hunan_source, args.data_root, sc["manifest"], snorm,
                              "/tmp/slfu_hunan_triad_frames_"+sc["manifest_sha256"][:16], split)
        available = np.unique(labels["sequence"][labels["split"] == value]).astype(int).tolist()
        indices = stratified_indices(ds.base.rows, available, limit)
        assert len(indices) == limit
        selections[split] = indices
        init_days[split] = {str(k):(datetime.strptime(ds.base.rows[k]["BJT_start"], "%Y-%m-%d %H:%M:%S")
                                  +timedelta(minutes=105)).strftime("%Y-%m-%d") for k in indices}
        dst = args.output/split
        dst.mkdir()
        arrays = {}
        shapes = {"image_local": (2,16,16,64,64), "image_region": (16,16,32,32),
                  "cot_local_history": (2,8,1,64,64), "cot_local_forecast": (2,16,1,64,64),
                  "cot_region_history": (8,1,32,32), "cot_region_forecast": (16,1,32,32),
                  "cot_local_oracle": (2,16,1,64,64), "cot_region_oracle": (16,1,32,32)}
        for name, shape in shapes.items():
            arrays[name] = np.lib.format.open_memmap(dst/(name+".npy"), mode="w+", dtype=np.float32,
                                                    shape=(len(indices), *shape))
        selected_rows = np.flatnonzero((labels["split"] == value) & np.isin(labels["sequence"], indices))
        np.save(dst/"head_row_ids.npy", selected_rows)
        np.save(dst/"sequence_indices.npy", np.array(indices))
        prediction_cache = {}
        if simvp_batch is None:
            _, h0, g0 = ds[indices[0]]
            safe = []
            for size in (1,2,4,8,16,32):
                try:
                    hi = torch.tensor(np.repeat(h0[None],size,axis=0),device="cuda")
                    gi = torch.tensor(np.repeat(g0[None],size,axis=0),device="cuda")
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.monotonic()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for _ in range(2):
                            z = s(hi,gi)
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated()
                    rec = {"batch":size,"peak_gib":peak/2**30,
                           "sequences_per_second":2*size/(time.monotonic()-t0)}
                    simvp_profile.append(rec)
                    if peak < .75*torch.cuda.get_device_properties(0).total_memory:
                        safe.append(size)
                    else:
                        break
                except torch.cuda.OutOfMemoryError:
                    simvp_profile.append({"batch":size,"state":"OOM"})
                    break
                finally:
                    if "hi" in locals(): del hi
                    if "gi" in locals(): del gi
                    if "z" in locals(): del z
                    torch.cuda.empty_cache()
            if not safe: raise RuntimeError("no safe SimVP inference batch")
            simvp_batch = max(safe)
            write(args.output/"simvp_resource_profile.json", {"selected_batch":simvp_batch,
                  "rule":"largest profiled inference batch below75% VRAM", "profile":simvp_profile})
        for j, idx in enumerate(indices):
            if not prediction_cache:
                chunk = indices[j:j+simvp_batch]
                inputs = [ds[k] for k in chunk]
                hi = torch.tensor(np.stack([v[1] for v in inputs]), device="cuda")
                gi = torch.tensor(np.stack([v[2] for v in inputs]), device="cuda")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = s(hi,gi).float()
                outputs = predicted.cpu().numpy()
                profiles.append({"split":split,"sequences":len(chunk),"simvp_seconds":time.monotonic()-t0,
                                 "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30})
                prediction_cache = {k:(inputs[ii][1],inputs[ii][2],outputs[ii]) for ii,k in enumerate(chunk)}
                del hi,gi,predicted
            history, geometry, future_np = prediction_cache.pop(idx)
            h = torch.tensor(history, device="cuda")
            g = torch.tensor(geometry, device="cuda")
            smn = torch.tensor(ds.mean, device="cuda")[None]
            sst = torch.tensor(ds.std, device="cuda")[None]
            future = torch.tensor(future_np,device="cuda")
            assert future.shape == (16,13,256,256) and bool(torch.isfinite(future).all())
            raw_future = torch.cat((future*sst+smn, g[8:]), 1)
            raw_history = torch.cat((h*sst+smn, g[:8]), 1)
            # Restore invalid history input to zero after R normalization, rather than physical mean.
            rels = ds.base.rows[idx]["data_relpaths"].split("|")
            for ti, rel in enumerate(rels[:8]):
                agri, valid, geom = ds.historical_frame(rel)
                raw_history[ti, :13] = torch.tensor(np.where(valid, agri, np.nan), device="cuda")
            ch, cf = retrieve(raw_history), retrieve(raw_future)
            norm_image = torch.cat((future, g[8:]), 1)
            arrays["image_local"][j] = crops(norm_image).cpu().numpy()
            arrays["image_region"][j] = F.avg_pool2d(norm_image, 8).cpu().numpy()
            for name, tensor in (("history", ch), ("forecast", cf)):
                arrays["cot_local_"+name][j] = crops(tensor).cpu().numpy()
                arrays["cot_region_"+name][j] = F.avg_pool2d(tensor, 8).cpu().numpy()
            # Oracle-only branch: actual future images never enter deployable image features.
            observed = []
            for rel in rels[8:]:
                agri, valid, geom = load_frame(args.data_root, rel)
                observed.append(np.concatenate((np.where(valid, agri, np.nan), geom)))
            oracle = retrieve(torch.tensor(np.stack(observed), dtype=torch.float32, device="cuda"))
            assert bool(torch.isfinite(ch).all()) and bool(torch.isfinite(cf).all()) and bool(torch.isfinite(oracle).all())
            arrays["cot_local_oracle"][j] = crops(oracle).cpu().numpy()
            arrays["cot_region_oracle"][j] = F.avg_pool2d(oracle, 8).cpu().numpy()
            if split == "val":
                shift = motion_shift(history)
                for lead, rel in enumerate(rels[8:]):
                    stamp = Path(rel).stem
                    if stamp not in cpp_lookup:
                        continue
                    cd, row = cpp_lookup[stamp]
                    truth = np.load(cd/"cot.npy", mmap_mode="r")[row]
                    mask = np.load(cd/"mask.npy", mmap_mode="r")[row].copy()
                    adv, adv_valid = transport(ch[-1], shift, lead+1)
                    # All three baseline metrics use the same fixed valid target region;
                    # adv missing coordinates are recorded instead of changing each method's mask.
                    missing = mask & ~adv_valid.cpu().numpy()
                    sums[f"lead{lead+1}/transport_missing"] += (mask.sum(), missing.sum(), 0)
                    common = mask & adv_valid.cpu().numpy()
                    for name, pred in (("persistence", ch[-1]), ("transport", adv), ("simvp", cf[lead]), ("oracle", oracle[lead])):
                        error = np.expm1(pred[0].cpu().numpy())-truth
                        take = common & np.isfinite(error)
                        e = error[take].astype(np.float64)
                        sums[f"lead{lead+1}/{name}"] += (e.size, np.abs(e).sum(), np.square(e).sum())
            for array in arrays.values():
                array.flush()
            write(args.output/"status.json", {"state": "BUILDING_REGIONAL_PILOT", "split": split,
                "completed": j+1, "total": len(indices), "elapsed_seconds": time.monotonic()-started,
                "test_used": False})
            print("SEQUENCE", split, j+1, len(indices), flush=True)
        del arrays
    report = {"state": "COMPLETE_EXPLORATORY_REGIONAL_GHI_PILOT_BANK", "test_used": False,
        "selection": "deterministic BJT-month stratified manifest ranks among existing GHI cohort; no CPP availability selection",
        "selected_sequences": selections,"initialization_BJT_days":init_days,"manifest_sha256": sc["manifest_sha256"],
        "backbone_checkpoint_sha256": sha(checkpoint), "regional_R_checkpoint_sha256": sha(args.r_run/"best.pt"),
        "r_normalization_sha256": sha(args.r_cache/"norm.json"), "base_pack_audit_sha256": sha(base/"audit.json"),
        "image_contract": "fixed forecast AGRI+target geometry; local64 native resolution and spatial32 regional 8x8 cell means",
        "cot_contract": "8 historical +16 forecast regional R log1p COT; local64 and spatial32 grids; no global mean/p90",
        "oracle": "actual future AGRI -> same frozen regional R; isolated nondeployable diagnostic, not CPP truth",
        "lead_minutes": list(range(15,241,15)), "station_rc": STATIONS,
        "r_reference_QA": "historical CPP producer/physical QA unresolved; exploratory only",
        "profiles": profiles, "simvp_profile":simvp_profile,"simvp_batch":simvp_batch,
        "elapsed_seconds": time.monotonic()-started}
    diagnostics = {}
    for key, v in sums.items():
        if "missing" in key:
            diagnostics[key] = {"target_valid_pixels": int(v[0]), "missing_transport_pixels": int(v[1])}
        elif v[0]:
            diagnostics[key] = {"n_pixels": int(v[0]), "mae": float(v[1]/v[0]), "rmse": float(np.sqrt(v[2]/v[0]))}
    write(args.output/"future_cot_baselines.json", {"test_used": False, "reference": "CPP retrieval reference",
        "mask": "same intersection of target CPP mask and transport in-domain support for all four methods; coverage separately recorded",
        "metrics": diagnostics})
    write(args.output/"complete.json", report)
    write(args.output/"status.json", report)
    print(report["state"], flush=True)


if __name__ == "__main__":
    main()
