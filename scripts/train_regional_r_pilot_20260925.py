"""Exploratory 256px COT retrieval smoke/profile on a stratified Hunan pilot.

Only train/val frames are read. The raw CPP reference has incomplete producer
provenance; this run cannot establish a physical or paper-level COT result.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(split, args, meta, by_stem, data_module):
    archive = args.cpp_pilot / meta["splits"][split]["archive"]
    if sha(archive) != meta["splits"][split]["archive_sha256"]:
        raise RuntimeError("CPP pilot archive drift")
    with np.load(archive, allow_pickle=False) as arr:
        stems = [s.decode("ascii") for s in arr["utc"]]
        cot, valid_cpp = arr["cot"], arr["valid"]
    x, valid_image, masks = [], [], []
    for i, stem in enumerate(stems):
        agri, valid, geometry = data_module.load_frame(args.data_root, by_stem[(split, stem)])
        if (agri.shape != (13, 256, 256) or geometry.shape != (3, 256, 256)
                or not np.isfinite(geometry).all()):
            raise RuntimeError("AGRI input contract mismatch")
        x.append(np.concatenate((agri, geometry)).astype(np.float32))
        valid_image.append(np.concatenate((valid, np.isfinite(geometry))).astype(bool))
        masks.append(valid_cpp[i] & valid.all(axis=0) & (geometry[2] > .5) &
                     (geometry[0] > np.cos(np.deg2rad(80))))
    x = np.stack(x)
    valid_image = np.stack(valid_image)
    y = np.log1p(cot).astype(np.float32)[:, None]
    mask = np.stack(masks)[:, None]
    if not mask.reshape(len(mask), -1).any(axis=1).all():
        raise RuntimeError("pilot includes frame with no valid label")
    return stems, x, valid_image, y, mask


def loss(pred, target, mask):
    values = F.huber_loss(pred.float(), target, reduction="none", delta=.1)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def main():
    p = argparse.ArgumentParser()
    for key in ("data-root", "manifest", "cpp-pilot", "hunan-data-code", "r-code", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite regional R pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("regional R profile requires one GPU")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    meta = json.loads((a.cpp_pilot / "manifest.json").read_text())
    if meta["test_used"] is not False or meta["manifest_sha256"] != sha(a.manifest):
        raise RuntimeError("CPP pilot metadata mismatch")
    by_stem = {}
    with a.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in ("train", "val"):
                continue
            for rel in row["data_relpaths"].split("|"):
                by_stem[(row["split"], Path(rel).stem)] = rel
    data_module = load_module("regional_hunan_data", a.hunan_data_code)
    retriever_module = load_module("regional_r_arch", a.r_code)
    train_stems, x_train, v_train, y_train, m_train = load_split("train", a, meta, by_stem, data_module)
    val_stems, x_val, v_val, y_val, m_val = load_split("val", a, meta, by_stem, data_module)
    total = np.where(v_train, x_train, 0).sum(axis=(0, 2, 3), dtype=np.float64)
    count = v_train.sum(axis=(0, 2, 3), dtype=np.float64)
    second = np.square(np.where(v_train, x_train, 0), dtype=np.float64).sum(axis=(0, 2, 3))
    if np.any(count == 0):
        raise RuntimeError("empty input channel")
    mean = (total/count).astype(np.float32)
    std = np.sqrt(np.maximum(second/count - (total/count)**2, 1e-8)).astype(np.float32)
    mean[15], std[15] = 0, 1
    def normalize(x, v):
        return np.where(v, (x-mean[None, :, None, None])/std[None, :, None, None], 0).astype(np.float32)
    x_train = torch.from_numpy(normalize(x_train, v_train)).cuda()
    x_val = torch.from_numpy(normalize(x_val, v_val)).cuda()
    y_train, y_val = torch.from_numpy(y_train).cuda(), torch.from_numpy(y_val).cuda()
    m_train, m_val = torch.from_numpy(m_train).cuda(), torch.from_numpy(m_val).cuda()
    profile = []
    device_total = torch.cuda.get_device_properties(0).total_memory
    for batch in (2, 4, 8, 16):
        torch.manual_seed(42)
        model = retriever_module.COTUNet().cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            z = model(x_train[:batch])
            value = loss(z, y_train[:batch], m_train[:batch])
            value.backward()
            opt.step()
        torch.cuda.synchronize()
        profile.append({"batch": batch, "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
                        "sequences_per_second": 3*batch/(time.monotonic()-start)})
        del model, opt, z, value
        torch.cuda.empty_cache()
    feasible = [r for r in profile if r["peak_allocated_gib"]*1024**3 < .75*device_total]
    if not feasible:
        raise RuntimeError("no safe batch from profile")
    chosen = max(feasible, key=lambda r: r["sequences_per_second"])["batch"]
    torch.manual_seed(42)
    model = retriever_module.COTUNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best = [], float("inf")
    best_state = None
    for epoch in range(3):
        model.train()
        order = torch.randperm(len(x_train), device="cuda")
        train_weighted = train_pixels = 0.
        for ids in order.split(chosen):
            opt.zero_grad(set_to_none=True)
            z = model(x_train[ids])
            value = loss(z, y_train[ids], m_train[ids])
            value.backward()
            opt.step()
            pixels = float(m_train[ids].sum())
            train_weighted += float(value.detach())*pixels
            train_pixels += pixels
        model.eval()
        preds = []
        with torch.no_grad():
            for ids in torch.arange(len(x_val), device="cuda").split(chosen):
                preds.append(model(x_val[ids]).float().cpu().numpy())
        pred = np.concatenate(preds)
        reference, mask = y_val.cpu().numpy(), m_val.cpu().numpy()
        val_huber = float(F.huber_loss(torch.from_numpy(pred), torch.from_numpy(reference),
                                       reduction="none", delta=.1).numpy()[mask].mean())
        val_mae = float(np.abs(pred-reference)[mask].mean())
        history.append({"epoch": epoch, "train_huber": train_weighted/train_pixels,
                        "val_huber": val_huber, "val_mae_log1p_cot": val_mae})
        if val_huber < best:
            best = val_huber
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prediction = pred.copy()
    a.output.mkdir(parents=True)
    torch.save({"model": best_state, "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                "scope": "exploratory_60train_12val_raw_cpp"}, a.output / "best.pt")
    np.save(a.output / "val_pred_log1p.npy", best_prediction)
    report = {"state": "COMPLETE_REGIONAL_R_256_PILOT", "test_used": False,
              "train_frames": len(train_stems), "val_frames": len(val_stems),
              "training_role": "pipeline and GPU sizing pilot only; not paper COT/GHI evidence",
              "raw_cpp_producer_binding": "unverified", "independent_cpp_qa": "unavailable",
              "batch_selected": chosen, "gpu": torch.cuda.get_device_name(0),
              "profile": profile, "epochs": history,
              "cpp_pilot_manifest_sha256": sha(a.cpp_pilot / "manifest.json"),
              "source_script_sha256": sha(__file__), "r_arch_source_sha256": sha(a.r_code),
              "val_pred_sha256": sha(a.output / "val_pred_log1p.npy")}
    (a.output / "complete.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"state": report["state"], "batch": chosen, "epochs": history}))


if __name__ == "__main__":
    main()
