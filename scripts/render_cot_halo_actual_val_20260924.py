"""Observed-AGRI witness: frozen 16x16 COT retriever versus 32x32 input.

This is a retrieval consistency diagnostic, not a forecast or CPP validation.
The four validation sequences are fixed by position before looking at COT.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite real-image witness")
    torch.set_num_threads(1)
    sc = json.loads((args.project / "configs/s_frozen_hunan_seed42.json").read_text())
    rc = json.loads((args.project / "configs/r_frozen_repaired_seed42.json").read_text())
    if sha(sc["manifest"]) != sc["manifest_sha256"] or sha(rc["checkpoint"]) != rc["checkpoint_sha256"]:
        raise RuntimeError("source asset hash drift")
    with Path(sc["manifest"]).open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "val"]
    indices = sorted({0, len(rows)//3, 2*len(rows)//3, len(rows)-1})
    box = sc["station_patches"]["sili"]
    r0, r1, c0, c1 = box
    model_source = args.project / "scripts/train_cot_repaired.py"
    spec = importlib.util.spec_from_file_location("cot_halo_witness_r", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(rc["checkpoint"], map_location="cpu", weights_only=False)
    if state["metadata"]["source_code_sha256"] != sha(model_source):
        raise RuntimeError("frozen R implementation drift")
    model = module.COTUNet().eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    mean = np.asarray(state["metadata"]["norm"]["mean"], np.float32)[:, None, None]
    std = np.asarray(state["metadata"]["norm"]["std"], np.float32)[:, None, None]
    sys.path.insert(0, str(args.source_root))
    from hunan_data import load_frame

    c13, native, extended, metadata = [], [], [], []
    with torch.no_grad():
        for index in indices:
            row = rows[index]
            rel = row["data_relpaths"].split("|")[7]
            agri, valid, geometry = load_frame(args.data_root, rel)
            physical = np.concatenate((np.where(valid, agri, np.nan), geometry), axis=0)
            normalized = (physical - mean) / std
            normalized[~np.isfinite(normalized)] = 0
            y16 = model(torch.from_numpy(normalized[None, :, r0:r1, c0:c1])).numpy()[0, 0]
            y32 = model(torch.from_numpy(normalized[None, :, r0-8:r1+8, c0-8:c1+8])).numpy()[0, 0]
            c13.append(np.where(valid[12, r0-8:r1+8, c0-8:c1+8], agri[12, r0-8:r1+8, c0-8:c1+8], np.nan))
            native.append(y16)
            extended.append(y32[8:24, 8:24])
            bjt = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") + timedelta(hours=8)
            metadata.append({"sequence_index": index, "seq_id": row["seq_id"], "history_end_bjt": bjt.isoformat(),
                             "station": "Sili", "day_mask_at_station": float(geometry[2, r0+8, c0+8]),
                             "native_vs_32_center_mae_log1p_cot": float(np.abs(y16-y32[8:24,8:24]).mean())})
    c13 = np.asarray(c13); native = np.asarray(native); extended = np.asarray(extended)
    delta = np.abs(native - extended)
    cot_max = float(np.percentile(np.concatenate((native.ravel(), extended.ravel())), 99))
    delta_max = float(np.percentile(delta, 99))
    ir_low, ir_high = np.nanpercentile(c13, [2, 98])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "source_arrays.npz", c13=c13, native_log1p_cot=native,
                        enlarged32_center_log1p_cot=extended, absolute_difference_log1p_cot=delta)
    report = {"state":"ACTUAL_VAL_AGRI_R_HALO_WITNESS", "test_used":False,
              "selection":"four fixed validation sequence positions 0, floor(n/3), floor(2n/3), n-1; Sili; observed history frame 7",
              "meaning":"same frozen R and AGRI frame; original 16x16 input versus 32x32 input, central 16x16 output",
              "not_a_forecast":True, "not_cpp_truth":True, "R_checkpoint_sha256":rc["checkpoint_sha256"],
              "manifest_sha256":sc["manifest_sha256"], "figure_values":"stored physical AGRI C13; log1p(COT) from R",
              "display_limits":{"c13_2_98_percentile":[float(ir_low),float(ir_high)],
                                "COT_common_0_99_percentile":[0,cot_max],"abs_difference_0_99_percentile":[0,delta_max]},
              "cases":metadata, "source_arrays_sha256":sha(args.output / "source_arrays.npz")}
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
    with plt.rc_context({"font.size":9, "figure.facecolor":"white", "savefig.facecolor":"white"}):
        fig, axes = plt.subplots(4, 4, figsize=(12.5, 11.5), layout="constrained")
        ims = [None]*4
        for i in range(4):
            panels = ((c13[i], "gray_r", float(ir_low), float(ir_high)),
                      (native[i], "magma", 0, cot_max),
                      (extended[i], "magma", 0, cot_max),
                      (delta[i], "viridis", 0, delta_max))
            for j, (array, cmap, low, high) in enumerate(panels):
                ax = axes[i,j]
                ims[j] = ax.imshow(array, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
                if j:
                    ax.plot(8, 8, marker="*", ms=8, mec="black", mew=.7, mfc="white")
                ax.set_xticks([0,8,15] if j else [0,16,31]); ax.set_yticks([0,8,15] if j else [0,16,31])
                ax.set_title(("Sili, " + metadata[i]["history_end_bjt"].replace("T", " ") +
                              (" (night)" if metadata[i]["day_mask_at_station"] == 0 else " (day)")) if j == 0 else
                             (f"MAE={metadata[i]['native_vs_32_center_mae_log1p_cot']:.3f}" if j == 3 else ""), fontsize=9)
        for j, title in enumerate(("Observed AGRI C13, 32x32", "R(original 16x16)",
                                   "R(32x32), central 16x16", "Absolute difference")):
            axes[0,j].set_xlabel(title, fontsize=10)
            axes[0,j].xaxis.set_label_position("top")
        fig.colorbar(ims[0], ax=axes[:,0], shrink=.75, label="Stored C13 value (common 2-98% display range)")
        fig.colorbar(ims[1], ax=axes[:,1:3], shrink=.75, label="R output: log(1+COT), common scale")
        fig.colorbar(ims[3], ax=axes[:,3], shrink=.75, label="|difference| in log(1+COT)")
        fig.suptitle("Real validation AGRI: retriever output changes with crop size", fontsize=14)
        fig.savefig(args.output / "r_halo_actual_val_sili.png", dpi=180)
        plt.close(fig)
    print("ACTUAL_VAL_AGRI_R_HALO_WITNESS_COMPLETE", json.dumps({"cases":len(metadata),
          "case_mae":[x["native_vs_32_center_mae_log1p_cot"] for x in metadata]}), flush=True)


if __name__ == "__main__":
    main()
