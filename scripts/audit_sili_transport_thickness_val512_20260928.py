"""Station-pixel audit of the completed Sili transport/thickness pilot.

Only the predeclared validation bank is loaded. CPP is treated as a retrieval
reference. The script emits per-sequence station tracks and validation metrics;
it never opens a test split.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_sili_transport_thickness_pilot_20260927 import (
    TransportThicknessNet, make_input, prepare,
)


METHODS = ("simvp", "persistence", "transport", "learned")
NAMES = {
    "simvp": "SimVP→R",
    "persistence": "Persistence",
    "transport": "C13 transport",
    "learned": "Transport + residual",
    "cpp": "CPP retrieval",
}


def summarize(ref, mask, preds):
    out = {"n_station_leads": int(mask.sum())}
    valid_ref = ref[mask]
    thick = mask & (ref >= 30)
    clear = mask & (ref < 5)
    out["thick_station_leads"] = int(thick.sum())
    out["clear_station_leads"] = int(clear.sum())
    for name, pred in preds.items():
        err = pred[mask] - valid_ref
        e_thick = pred[thick] - ref[thick]
        out[name] = {
            "mae": float(np.mean(np.abs(err))) if err.size else None,
            "rmse": float(np.sqrt(np.mean(err**2))) if err.size else None,
            "bias": float(np.mean(err)) if err.size else None,
            "thick_mae": float(np.mean(np.abs(e_thick))) if e_thick.size else None,
            "thick_rmse": float(np.sqrt(np.mean(e_thick**2))) if e_thick.size else None,
            "thick_miss_lt10": int(np.sum(thick & (pred < 10))),
            "thick_miss_rate_lt10": float(np.mean(pred[thick] < 10)) if e_thick.size else None,
            "clear_false_ge10": int(np.sum(clear & (pred >= 10))),
            "clear_false_rate_ge10": float(np.mean(pred[clear] >= 10)) if np.any(clear) else None,
            "iou_cot30": float(np.sum(mask & (ref >= 30) & (pred >= 30)) /
                                max(1, np.sum(mask & ((ref >= 30) | (pred >= 30))))),
        }
    return out


def peak_audit(ref, mask, preds, leads):
    rows = []
    for i in range(ref.shape[0]):
        v = mask[i]
        if not v.any():
            continue
        r = np.where(v, ref[i], np.nan)
        if not np.isfinite(r).any() or np.nanmax(r) < 30:
            continue
        peak_idx = int(np.nanargmax(r))
        row = {"sequence_index": i, "reference_peak_cot": float(r[peak_idx]),
               "reference_peak_lead_min": int(leads[peak_idx])}
        for name, p in preds.items():
            pv = np.where(v, p[i], np.nan)
            pidx = int(np.nanargmax(pv)) if np.isfinite(pv).any() else None
            row[name + "_at_reference_peak"] = float(p[i, peak_idx])
            row[name + "_max"] = float(np.nanmax(pv)) if pidx is not None else None
            row[name + "_max_lead_min"] = int(leads[pidx]) if pidx is not None else None
            row[name + "_peak_time_error_min"] = int(leads[pidx] - leads[peak_idx]) if pidx is not None else None
        rows.append(row)
    return rows


def choose_cases(ref, mask, history, days):
    valid_seq = [i for i in range(ref.shape[0]) if mask[i].any()]
    maxima = {i: float(np.nanmax(np.where(mask[i], ref[i], np.nan))) for i in valid_seq}
    last_valid = {i: bool(mask[i, -1]) for i in valid_seq}
    last = {i: float(ref[i, -1]) if last_valid[i] else np.nan for i in valid_seq}
    classes = {
        "cloud_growth": [i for i in valid_seq if history[i] < 5 and maxima[i] >= 30],
        "cloud_decay": [i for i in valid_seq if maxima[i] >= 30 and last_valid[i] and last[i] < 10],
        "persistent_thick": [i for i in valid_seq if maxima[i] >= 30 and last_valid[i] and last[i] >= 20],
        "clear_reference": [i for i in valid_seq if maxima[i] < 5],
    }
    picked, used_days = {}, set()
    for kind, candidates in classes.items():
        candidates = sorted(candidates, key=lambda i: (maxima[i], -i), reverse=kind != "clear_reference")
        for i in candidates:
            day = days[i]
            if day not in used_days:
                picked[kind] = i
                used_days.add(day)
                break
    return picked, {k: len(v) for k, v in classes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    bank_meta = json.loads((args.bank / "complete.json").read_text())
    if bank_meta.get("test_used") is not False or bank_meta.get("state") != "COMPLETE_SILI_COT_DYNAMICS_BANK":
        raise RuntimeError("Validation bank completion/provenance gate failed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"stage": "start", "device": device, "bank": str(args.bank),
                      "checkpoint": str(args.checkpoint)}), flush=True)
    print("stage=prepare_validation", flush=True)
    d = prepare(args.bank, "val", np.arange(512), device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TransportThicknessNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    predictions = {k: [] for k in METHODS}
    truths, masks = [], []
    with torch.inference_mode():
        for start in range(0, 512, 8):
            if start % 128 == 0:
                print(f"stage=inference_sequences_{start}_of_512", flush=True)
            ids = slice(start, min(start + 8, 512))
            h, f = d["h"][ids], d["f"][ids]
            y, m = d["y"][ids], d["mask"][ids]
            motion, adv = d["motion"][ids], d["adv"][ids]
            p = model(make_input(h, f, adv, motion), adv)
            pred = {
                "simvp": torch.expm1(f[:, :, 0]),
                "persistence": torch.expm1(h[:, -1, 0])[:, None].expand(-1, 16, -1, -1),
                "transport": torch.expm1(adv[:, 0]),
                "learned": torch.expm1(p[:, 0]),
            }
            for name in METHODS:
                predictions[name].append(pred[name][:, :, 32, 32].cpu().numpy())
            truths.append(torch.expm1(y)[:, :, 32, 32].cpu().numpy())
            masks.append((m & d["support"][ids, 0])[:, :, 32, 32].cpu().numpy())
    pred = {k: np.concatenate(v) for k, v in predictions.items()}
    ref = np.concatenate(truths)
    mask = np.concatenate(masks).astype(bool)
    hist = torch.expm1(d["h"][:, -1, 0, 32, 32]).cpu().numpy()
    leads = np.asarray(bank_meta["lead_minutes"], dtype=int)
    selected = list(bank_meta["selected_sequences"]["val"])
    days_map = bank_meta["initialization_BJT_days"]["val"]
    days = [days_map[str(i)] for i in selected]
    if len(days) != 512 or not np.array_equal(np.asarray(selected), np.arange(512)):
        # Data arrays are saved in selection order, independent of original row id.
        if len(days) != 512:
            raise RuntimeError("Validation sequence/date manifest length mismatch")
    day_for_row = days
    print("stage=aggregate_station_metrics", flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    metrics = summarize(ref, mask, pred)
    per_lead = []
    for t, lead in enumerate(leads):
        item = {"lead_min": int(lead), "n": int(mask[:, t].sum()),
                "thick_n": int((mask[:, t] & (ref[:, t] >= 30)).sum()),
                "clear_n": int((mask[:, t] & (ref[:, t] < 5)).sum())}
        for name, p in pred.items():
            q = mask[:, t]
            thick = q & (ref[:, t] >= 30)
            clear = q & (ref[:, t] < 5)
            e = p[:, t] - ref[:, t]
            item[name + "_rmse"] = float(np.sqrt(np.mean(e[q] ** 2))) if q.any() else None
            item[name + "_thick_rmse"] = float(np.sqrt(np.mean(e[thick] ** 2))) if thick.any() else None
            item[name + "_miss_lt10"] = int(np.sum(thick & (p[:, t] < 10)))
            item[name + "_clear_false_ge10_rate"] = float(np.mean(p[clear, t] >= 10)) if clear.any() else None
        per_lead.append(item)
    peaks = peak_audit(ref, mask, pred, leads)
    picked, case_counts = choose_cases(ref, mask, hist, day_for_row)
    metrics["peak_event_count"] = len(peaks)
    metrics["peak_event_mean"] = {
        name: {key: float(np.mean([r[name + key] for r in peaks if r.get(name + key) is not None]))
               for key in ("_at_reference_peak", "_max", "_peak_time_error_min")
               if any(r.get(name + key) is not None for r in peaks)}
        for name in METHODS
    }
    metrics["case_candidate_counts"] = case_counts
    metrics["selected_cases"] = {k: {"row": i, "original_sequence_index": int(selected[i]),
                                     "init_day_BJT": day_for_row[i],
                                     "reference_peak_cot": float(np.nanmax(np.where(mask[i], ref[i], np.nan)))}
                                  for k, i in picked.items()}
    metrics["source"] = {"best_epoch": int(ckpt["epoch"]), "test_used": False,
                         "split": "validation", "station_patch_pixel": [32, 32],
                         "reference": "CPP COT retrieval, not independent in-situ cloud truth",
                         "support": "CPP valid mask intersected with pure-transport in-domain support"}
    print("stage=write_json_and_tracks", flush=True)
    (output / "summary.json").write_text(json.dumps(metrics, indent=2))
    (output / "lead_metrics.json").write_text(json.dumps(per_lead, indent=2))
    (output / "peak_events.json").write_text(json.dumps(peaks, indent=2))
    np.savez_compressed(output / "station_tracks.npz", sequence_index=np.asarray(selected),
                        init_day_BJT=np.asarray(day_for_row), lead_minutes=leads,
                        cpp=ref, cpp_valid=mask, history_last=hist,
                        **pred)
    with (output / "station_tracks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["val_row", "sequence_index", "init_day_BJT", "lead_min", "cpp", "valid",
                    "history_last", *METHODS])
        for i in range(512):
            for t, lead in enumerate(leads):
                w.writerow([i, selected[i], day_for_row[i], int(lead), float(ref[i,t]), int(mask[i,t]),
                            float(hist[i]), *[float(pred[k][i,t]) for k in METHODS]])
    print("stage=plot_event_tracks", flush=True)
    names = list(picked)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, kind in zip(axes.ravel(), names):
        i = picked[kind]
        ax.plot(np.r_[0, leads], np.r_[hist[i], np.where(mask[i], ref[i], np.nan)], "k-o", lw=2, label=NAMES["cpp"] + "/history")
        for method in METHODS:
            ax.plot(leads, pred[method][i], marker=".", label=NAMES[method])
        ax.axhline(30, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{kind.replace('_', ' ')} | init day {day_for_row[i]} | CPP peak {metrics['selected_cases'][kind]['reference_peak_cot']:.1f}")
        ax.set_xlabel("Forecast lead (min; targets +15 to +240)")
        ax.set_ylabel("Station-pixel COT")
        ax.grid(alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels() if names else ([], [])
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Sili station-pixel trajectories — validation only, CPP retrieval reference")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output / "station_event_tracks.png", dpi=180)
    fig.savefig(output / "station_event_tracks.pdf")
    plt.close(fig)

    print("stage=plot_lead_metrics", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for method in METHODS:
        axes[0].plot(leads, [x[method + "_rmse"] for x in per_lead], marker="o", label=NAMES[method])
        axes[1].plot(leads, [x[method + "_miss_lt10"] / max(1, x["thick_n"]) for x in per_lead], marker="o", label=NAMES[method])
    axes[0].set_ylabel("Station COT RMSE")
    axes[1].set_ylabel("Thick-reference miss rate\n(prediction < 10, CPP ≥ 30)")
    axes[1].set_xlabel("Forecast lead (min; targets +15 to +240)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle("Sili station-pixel skill by lead — validation only")
    fig.tight_layout()
    fig.savefig(output / "station_lead_metrics.png", dpi=180)
    fig.savefig(output / "station_lead_metrics.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(output), "summary": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
