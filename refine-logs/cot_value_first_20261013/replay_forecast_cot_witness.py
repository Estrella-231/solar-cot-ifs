#!/usr/bin/env python3
"""One-window causal SimVP -> monthly-R replay witness (validation only).

The sample is selected by CPP-supported lead coverage only. Forecast AGRI is
never compared to the target inside the model; target AGRI is used only to
form the already-supervised CPP reference through the saved validation R bank.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

EXPECTED = {
    "manifest": "f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2",
    "s_norm": "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d",
    "simvp": "f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4",
    "triad_models": "9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb",
    "triad_data": "8915b988d9db1b52d01a0c6cab9a7ee16bbde70835ebf6f6bf11732a9ee11502",
    "hunan_data": "f4b4aab55101bde575f474d5daa4a184466c16399b401c2ac67bf666c5f28b91",
    "r_checkpoint": "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001",
    "r_source": "7a2e5cb441cdb5132a537cd03580017801595bdea839719d1773398d267210ce",
}
CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("sili", "zhujia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key_from_rel(rel):
    t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
    return t.strftime("%Y-%m-%d %H:%M:%S")


def main():
    p = argparse.ArgumentParser()
    for name in ("solar_root", "data_root", "manifest", "norm", "simvp_checkpoint",
                 "simvp_source", "r_checkpoint", "r_source", "r_norm", "r_pack_rows",
                 "r_validation_predictions", "candidate", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved one-GPU replay witness")
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")

    checks = ((a.manifest, EXPECTED["manifest"]), (a.norm, EXPECTED["s_norm"]),
              (a.simvp_checkpoint, EXPECTED["simvp"]),
              (a.simvp_source / "triad_models.py", EXPECTED["triad_models"]),
              (a.simvp_source / "triad_data.py", EXPECTED["triad_data"]),
              (a.simvp_source / "hunan_data.py", EXPECTED["hunan_data"]),
              (a.r_checkpoint, EXPECTED["r_checkpoint"]), (a.r_source, EXPECTED["r_source"]))
    for path, expected in checks:
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"source/checkpoint hash mismatch: {path} {actual} != {expected}")

    a.output.mkdir(parents=True)
    sys.path.insert(0, str(a.simvp_source))
    sys.path.insert(0, str(a.r_source.parent))
    from triad_data import ForecastHunanDataset
    from triad_models import SimVPBackbone
    from train_station16_R import COTUNet

    norm = json.loads(a.norm.read_text())
    s_mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    s_std = np.asarray(norm["std"], np.float32)[:, None, None]
    sim_ck = torch.load(a.simvp_checkpoint, map_location="cpu", weights_only=False)
    sm = sim_ck["metadata"]
    if (sm.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1" or
            sm.get("manifest_sha256") != EXPECTED["manifest"] or
            sm.get("normalization_sha256") != EXPECTED["s_norm"] or
            sm.get("geometry_contract") != "history_8_then_target_16_v2" or
            sm.get("test_payloads_read") != 0):
        raise RuntimeError("SimVP checkpoint metadata violates the frozen Hunan validation contract")
    model = SimVPBackbone(a.simvp_source).cuda().eval().requires_grad_(False)
    model.load_state_dict(sim_ck["model"], strict=True)
    del sim_ck

    r_ck = torch.load(a.r_checkpoint, map_location="cpu", weights_only=False)
    rmeta = r_ck["metadata"]
    if (rmeta.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1" or
            rmeta.get("test_payloads_read") != 0 or
            rmeta.get("normalization_sha256") != sha(a.r_norm)):
        raise RuntimeError("monthly R checkpoint metadata violates the frozen train/val contract")
    rnorm = json.loads(a.r_norm.read_text())
    r_mean = torch.as_tensor(rnorm["mean"], device="cuda", dtype=torch.float32)[None, :, None, None]
    r_std = torch.as_tensor(rnorm["std"], device="cuda", dtype=torch.float32)[None, :, None, None]
    rmodel = COTUNet().cuda().float().eval().requires_grad_(False)
    rmodel.load_state_dict(r_ck["model"], strict=True)
    del r_ck

    # Build a key-only mapping from CPP-supervised validation R rows to their
    # saved reference and mask. These labels never enter the forecast model.
    with a.r_pack_rows.open(newline="", encoding="utf-8") as f:
        supervised = [r for r in csv.DictReader(f) if r["state"] == "SUPERVISED"]
    with np.load(a.r_validation_predictions, allow_pickle=False) as z:
        sample_indices = z["sample_indices"].astype(np.int64)
        ref_cot = z["reference_cot"].astype(np.float32)
        ref_mask = z["mask"].astype(bool)
    if len(sample_indices) != len(ref_cot) or len(ref_cot) != len(ref_mask):
        raise RuntimeError("saved R validation arrays have inconsistent sample counts")
    saved_val = {int(idx): j for j, idx in enumerate(sample_indices)}
    ref_by_key = {}
    for idx, row in enumerate(supervised):
        if row["split"] == "val" and idx in saved_val:
            j = saved_val[idx]
            ref_by_key[(row["station"], row["target_time_utc"])] = (ref_cot[j, 0], ref_mask[j, 0])

    with a.candidate.open(newline="", encoding="utf-8-sig") as f:
        candidate_rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    coords = {}
    for row in candidate_rows:
        k = (row["station"], row["target_time_utc"])
        coords[k] = tuple(map(int, row["station_grid_rc"].split(",")))

    dataset = ForecastHunanDataset(a.simvp_source, a.data_root, a.manifest, norm, "val", include_index=True)
    # Deterministically select the first window with CPP-supervised labels at
    # every lead for both stations. This is support-based smoke selection only.
    selected = None
    for i, row in enumerate(dataset.base.rows):
        rels = row["data_relpaths"].split("|")
        lead_keys = [key_from_rel(rel) for rel in rels[8:24]]
        if all((station, key) in ref_by_key and (station, key) in coords and
               bool(ref_by_key[(station, key)][1].any())
               for station in STATIONS for key in lead_keys):
            selected = i
            break
    if selected is None:
        raise RuntimeError("no validation window has CPP supervision for all 16 leads at both stations")

    seq_index, history_np, geometry_np, future_np, future_valid_np = dataset[selected]
    seq_row = dataset.base.rows[selected]
    rels = seq_row["data_relpaths"].split("|")
    if len(rels) != 24:
        raise RuntimeError("validation sequence is not 8+16 frames")
    history = torch.from_numpy(history_np[None]).cuda()
    geometry = torch.from_numpy(geometry_np[None]).cuda()
    if history.shape != (1, 8, 13, 256, 256) or geometry.shape != (1, 24, 3, 256, 256):
        raise RuntimeError("SimVP input shapes violate the audited 8+16 contract")
    torch.backends.cudnn.benchmark = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_norm = model(history, geometry)
    if pred_norm.shape != (1, 16, 13, 256, 256) or not torch.isfinite(pred_norm).all():
        raise RuntimeError("SimVP forecast output is malformed or non-finite")
    mean = torch.as_tensor(s_mean, device="cuda")[None, None]
    std = torch.as_tensor(s_std, device="cuda")[None, None]
    pred_physical = pred_norm.float() * std + mean
    # The target tensor is kept separate and is only read by the R->CPP
    # reference lookup below; the forecast call above receives history/geometry.
    target_physical = torch.from_numpy(future_np[None]).cuda().float() * std + mean
    geo_future = geometry[:, 8:].float()

    def crop_batch(fields, geos, keys):
        xs = []
        for station in STATIONS:
            crops = []
            for lead, key in enumerate(keys):
                row, col = coords[(station, key)]
                r0, c0 = row - 8, col - 8
                crops.append(torch.cat((fields[0, lead, :, r0:r0+16, c0:c0+16],
                                        geos[0, lead, :, r0:r0+16, c0:c0+16]), dim=0))
            xs.extend(crops)
        x = torch.stack(xs)
        x = (x - r_mean) / r_std
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    lead_keys = [key_from_rel(rel) for rel in rels[8:24]]
    forecast_x = crop_batch(pred_physical, geo_future, lead_keys)
    target_x = crop_batch(target_physical, geo_future, lead_keys)
    # Persistence uses only the final observed frame and its own solar geometry.
    hist_physical = history[:, -1].float() * std[:, 0] + mean[:, 0]
    hist_geo = geometry[:, 7:8].float()
    persist_fields = hist_physical[:, None].expand(-1, 1, -1, -1, -1)
    persist_geo = hist_geo
    persist_keys = [key_from_rel(rels[7])]
    persistence_x = crop_batch(persist_fields, persist_geo, persist_keys)

    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        forecast_log1p = rmodel(forecast_x.float())[:, 0].reshape(2, 16, 16, 16)
        target_log1p = rmodel(target_x.float())[:, 0].reshape(2, 16, 16, 16)
        persistence_log1p = rmodel(persistence_x.float())[:, 0].reshape(2, 1, 16, 16)
    if not all(torch.isfinite(x).all() for x in (forecast_log1p, target_log1p, persistence_log1p)):
        raise RuntimeError("R produced non-finite values")

    refs, masks = [], []
    for station in STATIONS:
        station_refs, station_masks = [], []
        for key in lead_keys:
            cot, mask = ref_by_key[(station, key)]
            station_refs.append(cot)
            station_masks.append(mask)
        refs.append(np.stack(station_refs))
        masks.append(np.stack(station_masks))
    refs, masks = np.stack(refs), np.stack(masks)
    if masks.shape != (2, 16, 16, 16) or int(masks.sum()) == 0:
        raise RuntimeError("CPP support failed to align to the witness forecast keys")
    if not np.isfinite(refs[masks]).all():
        raise RuntimeError("non-finite CPP reference under common support")

    pred_log = forecast_log1p.cpu().numpy().astype(np.float32)
    np.savez_compressed(a.output / "witness_predictions.npz",
                        forecast_log1p_cot=pred_log,
                        forecast_cot=np.expm1(pred_log.astype(np.float64)).astype(np.float32),
                        target_agri_r_log1p_cot=target_log1p.cpu().numpy().astype(np.float32),
                        persistence_log1p_cot=persistence_log1p.cpu().numpy().astype(np.float32),
                        cpp_reference_cot=refs.astype(np.float32), cpp_valid_mask=masks,
                        future_agri_valid=future_valid_np.astype(bool))
    lead_records = []
    for j, key in enumerate(lead_keys):
        target_utc = datetime.strptime(key, "%Y-%m-%d %H:%M:%S")
        lead_records.append({"lead_min": (j + 1) * 15, "target_time_utc": key,
                             "target_time_bjt": (target_utc + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                             "cpp_valid_pixels_by_station": [int(masks[s, j].sum()) for s in range(2)]})
    (a.output / "lead_keys.json").write_text(json.dumps(lead_records, indent=2) + "\n")
    receipt = {
        "state": "PASS_ONE_WINDOW_SIMVP_TO_MONTHLY_R_CPP_REPLAY_WITNESS",
        "split": "val_only", "test_payloads_read": 0,
        "selection": "first validation window with CPP-supervised support at all 16 leads for both stations; support-only, no forecast errors inspected",
        "manifest_sha256": sha(a.manifest), "simvp_norm_sha256": sha(a.norm),
        "simvp_checkpoint_sha256": sha(a.simvp_checkpoint), "r_checkpoint_sha256": sha(a.r_checkpoint),
        "r_norm_sha256": sha(a.r_norm), "r_source_sha256": sha(a.r_source),
        "simvp_model_source_sha256": sha(a.simvp_source / "triad_models.py"),
        "sequence_index": int(seq_index), "sequence_id": seq_row.get("seq_id"),
        "history_bjt_start": seq_row.get("BJT_start"),
        "history_end_utc": key_from_rel(rels[7]), "future_agri_loaded_for": "R-to-CPP offline reference only",
        "future_agri_passed_to_simvp": False, "future_cpp_passed_to_simvp": False,
        "input_shapes": {"history": list(history.shape), "geometry24": list(geometry.shape)},
        "forecast_shape": list(pred_norm.shape), "station_order": list(STATIONS),
        "lead_count": len(lead_keys), "cpp_common_support_pixels": int(masks.sum()),
        "finite": {"simvp_normalized": bool(torch.isfinite(pred_norm).all()),
                   "forecast_physical_agri": bool(torch.isfinite(pred_physical).all()),
                   "forecast_r_log1p_cot": bool(torch.isfinite(forecast_log1p).all()),
                   "target_r_log1p_cot": bool(torch.isfinite(target_log1p).all())},
        "gpu": torch.cuda.get_device_name(0),
        "outputs": ["witness_predictions.npz", "lead_keys.json"],
    }
    (a.output / "witness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
