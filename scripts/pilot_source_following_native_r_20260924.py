"""Causal fixed-sample pilot: follow upstream AGRI and retrieve with native 16px R.

The last observed AGRI frame is cropped at the P16 motion-estimated upstream
source for each target lead. Frozen R always receives its trained 16x16 input;
future real AGRI enters only the separate offline evaluation reference.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def score(pred, reference, valid):
    if not valid.any():
        return None
    error = np.abs(pred[valid] - reference[valid])
    return {"pixels": int(valid.sum()), "mae_log1p_cot": float(error.mean())}


def main():
    p = argparse.ArgumentParser()
    for name in ("project", "data-root", "source-root", "pilot", "oracle", "history", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite native-R source-following pilot")
    torch.set_num_threads(2)
    sc = json.loads((args.project / "configs/s_frozen_hunan_seed42.json").read_text())
    rc = json.loads((args.project / "configs/r_frozen_repaired_seed42.json").read_text())
    pilot = json.loads((args.pilot / "pilot.json").read_text())
    oracle = json.loads((args.oracle / "complete.json").read_text())
    history_path = args.history / f"{args.split}_history_cot_log1p.npy"
    if (sha(Path(sc["manifest"])) != sc["manifest_sha256"] or
            sha(Path(sc["normalization"])) != sc["normalization_sha256"] or
            sha(Path(rc["checkpoint"])) != rc["checkpoint_sha256"] or
            pilot["state"] != "COMPLETE_P16_SUPPORT_PILOT" or pilot["test_used"] or
            pilot["manifest_sha256"] != sc["manifest_sha256"] or
            pilot["history_cot_sha256"] != sha(history_path) or
            oracle["state"] != "COMPLETE_REAL_FUTURE_R_ORACLE" or
            oracle["test_used"] or oracle["deployable"] or oracle["split"] != args.split or
            oracle["R_checkpoint_sha256"] != rc["checkpoint_sha256"]):
        raise RuntimeError("frozen source/manifest/split contract drift")
    model_source = args.project / "scripts/train_cot_repaired.py"
    spec = importlib.util.spec_from_file_location("native_source_following_r", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(rc["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["metadata"]["source_code_sha256"] != sha(model_source):
        raise RuntimeError("R implementation drift")
    model = module.COTUNet().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    mean = np.asarray(checkpoint["metadata"]["norm"]["mean"], np.float32)[:, None, None]
    std = np.asarray(checkpoint["metadata"]["norm"]["std"], np.float32)[:, None, None]
    simvp_norm = json.loads(Path(sc["normalization"]).read_text())
    simvp_mean = np.asarray(simvp_norm["mean"], np.float32)[:, None, None]
    sys.path.insert(0, str(args.source_root))
    from hunan_data import load_frame

    indices = np.load(args.pilot / "sequence_indices.npy", allow_pickle=False)
    p16 = np.load(args.pilot / "p16_cot_log1p_nan_outside_support.npy", allow_pickle=False)
    support = np.load(args.pilot / "p16_valid_support.npy", allow_pickle=False)
    history = np.load(history_path, mmap_mode="r", allow_pickle=False)
    ref_path = args.oracle / "real_future_cot_log1p.npy"
    if sha(ref_path) != oracle["cot_sha256"]:
        raise RuntimeError("oracle reference hash drift")
    reference = np.load(ref_path, mmap_mode="r", allow_pickle=False)[indices, :, :, 0]
    if (p16.shape != reference.shape or support.shape != p16.shape or
            p16.shape != (len(indices), 2, 16, 16, 16)):
        raise RuntimeError("fixed pilot/reference shape mismatch")
    with Path(sc["manifest"]).open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == args.split]
    if not np.isfinite(reference).all() or not np.array_equal(indices, np.unique(indices)):
        raise RuntimeError("invalid oracle or pilot sequence indices")
    motions = {(r["sequence_index"], r["station"]): r for r in pilot["records"]}
    stations = list(sc["station_patches"].items())
    if len(stations) != 2 or len(motions) != 2 * len(indices):
        raise RuntimeError("station/motion record mismatch")
    retrieved = np.full_like(reference, np.nan)
    available = np.zeros(reference.shape[:3], dtype=bool)
    crop_valid = np.full(reference.shape[:3], np.nan, np.float32)
    offsets = np.zeros((len(indices), 2, 16, 2), np.int16)
    offset_zero_drift = []
    with torch.no_grad():
        for pos, sequence in enumerate(indices):
            row = rows[int(sequence)]
            rel = row["data_relpaths"].split("|")[7]
            agri, valid, geometry = load_frame(args.data_root, rel)
            # The historical bank first zeros invalid SimVP-normalized AGRI,
            # then inverts that normalization before R. Its physical fill is
            # therefore the frozen SimVP channel mean, not R's zero input.
            physical = np.concatenate((np.where(valid, agri, simvp_mean), geometry), axis=0)
            if physical.shape != (16, 256, 256):
                raise RuntimeError("historical AGRI/geometry layout drift")
            normalized = (physical - mean) / std
            source_valid = np.concatenate((valid, np.isfinite(geometry)), axis=0)
            if not np.isfinite(normalized).all():
                raise RuntimeError("nonfinite native-R input")
            inputs, keys = [], []
            for station_index, (station_name, box) in enumerate(stations):
                r0, r1, c0, c1 = box
                record = motions[(int(sequence), station_name)]
                if record["seq_id"] != row["seq_id"]:
                    raise RuntimeError("motion/manifest sequence mismatch")
                dy, dx = record["dy_pixels_per_15min"], record["dx_pixels_per_15min"]
                for lead in range(1, 17):
                    sy, sx = int(np.rint(dy * lead)), int(np.rint(dx * lead))
                    offsets[pos, station_index, lead-1] = (sy, sx)
                    rr0, rr1, cc0, cc1 = r0-sy, r1-sy, c0-sx, c1-sx
                    if min(rr0, cc0) < 0 or max(rr1, cc1) > 256:
                        continue
                    fraction = float(source_valid[:, rr0:rr1, cc0:cc1].mean())
                    crop_valid[pos, station_index, lead-1] = fraction
                    if fraction < .95:
                        continue
                    inputs.append(normalized[:, rr0:rr1, cc0:cc1])
                    keys.append((station_index, lead-1, sy, sx))
            if inputs:
                for start in range(0, len(inputs), 32):
                    batch = np.stack(inputs[start:start+32]).astype(np.float32)
                    output = model(torch.from_numpy(batch)).numpy()[:, 0]
                    for local, (station_index, k, sy, sx) in enumerate(keys[start:start+32]):
                        value = output[local]
                        retrieved[pos, station_index, k] = value
                        available[pos, station_index, k] = True
                        if sy == sx == 0:
                            prior = history[sequence, station_index, 7, 0]
                            offset_zero_drift.append(float(np.abs(value-prior).max()))
    # The bank was evaluated on CUDA and this bounded witness on CPU; retain
    # the measured drift and reject a material mismatch rather than demanding
    # bit-identical convolution kernels across devices.
    if offset_zero_drift and max(offset_zero_drift) > 0.01:
        raise RuntimeError(f"native zero-offset R disagrees with frozen historical bank: max {max(offset_zero_drift)}")
    if not np.isfinite(retrieved[available]).all() or np.isfinite(retrieved[~available]).any():
        raise RuntimeError("invalid/missing source-following COT not preserved")
    persistence = np.broadcast_to(history[indices, :, 7, 0, None], reference.shape)
    report_leads = []
    for k in range(16):
        map_valid = available[:, :, k, None, None]
        full_valid = np.broadcast_to(map_valid, reference[:, :, k].shape)
        center = np.zeros_like(full_valid)
        center[..., 6:11, 6:11] = True
        item = {"lead_minutes": (k+1)*15,
                "available_source_patch_fraction": float(available[:, :, k].mean()),
                "mean_finite_input_fraction": float(np.nanmean(crop_valid[:, :, k]))}
        for label, area in (("full16", full_valid), ("center5", full_valid & center)):
            item[label] = {
                "AP": score(retrieved[:, :, k], reference[:, :, k], area),
                "P0_on_AP_rows": score(persistence[:, :, k], reference[:, :, k], area),
            }
            common = area & support[:, :, k]
            item[label]["AP_on_P16_support"] = score(retrieved[:, :, k], reference[:, :, k], common)
            item[label]["P16_on_same_support"] = score(p16[:, :, k], reference[:, :, k], common)
            item[label]["common_support_fraction_of_all_pixels"] = float(common.mean())
        report_leads.append(item)
    report = {"state":"COMPLETE_SOURCE_FOLLOWING_NATIVE_R_PILOT", "test_used":False,
              "split":args.split,"sequences":len(indices),"stations":[s for s,_ in stations],
              "source":"last observed AGRI frame and historical C13 motion only; integer upstream 16x16 crop",
              "R_input_contract":"frozen native 16x16; no larger-input extrapolation or stitched COT halo",
              "reference":"R(real future AGRI) only for offline comparison; not a deployed input or physical observation",
              "limitation":"small fixed pilot; upstream crop can move off station distribution; one rigid velocity per patch; not trained E",
              "zero_offset_max_abs_bank_drift":max(offset_zero_drift) if offset_zero_drift else None,
              "zero_offset_p95_abs_bank_drift":float(np.percentile(offset_zero_drift, 95)) if offset_zero_drift else None,
              "pilot_sha256":sha(args.pilot / "pilot.json"),
              "oracle_sha256":sha(args.oracle / "complete.json"),
              "history_sha256":sha(history_path),"R_checkpoint_sha256":rc["checkpoint_sha256"],
              "manifest_sha256":sc["manifest_sha256"],"script_sha256":sha(__file__),
              "per_lead":report_leads}
    args.output.mkdir(parents=True)
    np.save(args.output / "source_following_cot_log1p_nan_unavailable.npy", retrieved)
    np.save(args.output / "source_patch_available.npy", available)
    np.save(args.output / "rounded_motion_offset_pixels.npy", offsets)
    (args.output / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"state":report["state"],"split":args.split,
                      "sequences":len(indices),"lead15":report_leads[0]["center5"],
                      "lead240":report_leads[-1]["center5"]}), flush=True)


if __name__ == "__main__":
    main()
