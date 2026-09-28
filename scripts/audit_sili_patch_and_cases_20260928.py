"""Paired local-64 patch and physical COT case audit; validation only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from train_sili_transport_thickness_pilot_20260927 import (
    TransportThicknessNet, make_input, prepare,
)


METHODS = ("simvp", "persistence", "transport", "learned")
LABELS = {"simvp": "SimVP→R", "persistence": "Persistence",
          "transport": "Transport / step 0", "learned": "Learned residual"}
BINS = (("0-5", 0., 5.), ("5-10", 5., 10.), ("10-30", 10., 30.),
        ("30-60", 30., 60.), ("60-100", 60., 100.00001))


def agg(mask, truth, predictions):
    out = {"n": int(mask.sum())}
    if not mask.any():
        return out
    for name, pred in predictions.items():
        e = pred[mask] - truth[mask]
        thick = mask & (truth >= 30)
        clear = mask & (truth < 5)
        union = mask & ((truth >= 30) | (pred >= 30))
        out[name] = {
            "rmse": float(np.mean(e**2)**0.5),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "thick_n": int(thick.sum()),
            "thick_rmse": float(np.mean((pred[thick]-truth[thick])**2)**0.5) if thick.any() else None,
            "thick_miss_lt10": int(np.sum(thick & (pred < 10))),
            "thick_iou_cot30": float(np.sum(mask & (truth >= 30) & (pred >= 30)) / max(1, union.sum())),
            "clear_false_ge10_rate": float(np.mean(pred[clear] >= 10)) if clear.any() else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--station-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.bank / "complete.json").read_text())
    assert meta.get("state") == "COMPLETE_SILI_COT_DYNAMICS_BANK" and meta.get("test_used") is False
    assert not a.output.exists()
    torch.set_num_threads(2)
    d = prepare(a.bank, "val", np.arange(512), "cuda")
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    model = TransportThicknessNet().cuda().eval()
    model.load_state_dict(ck["model"])

    pred_parts = {k: [] for k in METHODS}
    truth_parts, mask_parts, support_parts = [], [], []
    with torch.inference_mode():
        for start in range(0, 512, 8):
            ids = slice(start, start + 8)
            h, f, y, m = d["h"][ids], d["f"][ids], d["y"][ids], d["mask"][ids]
            motion, adv = d["motion"][ids], d["adv"][ids]
            learned = model(make_input(h, f, adv, motion), adv)
            block = {
                # forecast is [B,T,1,H,W]; retain all forecast leads.
                "simvp": torch.expm1(f[:, :, 0]),
                "persistence": torch.expm1(h[:, -1, 0])[:, None].expand(-1, 16, -1, -1),
                "transport": torch.expm1(adv[:, 0]),
                "learned": torch.expm1(learned[:, 0]),
            }
            for k in METHODS:
                pred_parts[k].append(block[k].cpu().numpy())
            # prepare() returns target and mask as [B,T,H,W]. Preserve all
            # leads here; indexing [:,0] silently discarded 15 of 16 leads.
            truth_parts.append(torch.expm1(y).cpu().numpy())
            mask_parts.append(m.cpu().numpy().astype(bool))
            # Keep all 16 lead-time support masks: support is [N,1,T,H,W].
            support_parts.append(d["support"][ids, 0].cpu().numpy().astype(bool))
    truth = np.concatenate(truth_parts)
    ref_valid = np.concatenate(mask_parts)
    support = np.concatenate(support_parts)
    predictions = {k: np.concatenate(v) for k, v in pred_parts.items()}
    expected = (512, len(meta["lead_minutes"]), 64, 64)
    for name, value in {"truth": truth, "reference_mask": ref_valid,
                        "transport_support": support, **predictions}.items():
        if value.shape != expected:
            raise ValueError(f"{name} shape {value.shape}, expected {expected}")
    common = ref_valid & support

    # The model's zero-initialized final layer gives clamp(transport, 0, 100).
    # Compare this exact station/patch baseline with raw transport before scoring.
    step0 = np.clip(predictions["transport"], 0, 100)
    exact_support_equal = bool(np.array_equal(step0[common], predictions["transport"][common]))
    scoring = dict(predictions)
    scoring["transport"] = step0

    per_lead = []
    for t, lead in enumerate(meta["lead_minutes"]):
        q = common[:, t]
        per_lead.append({"lead_min": int(lead), "cpp_valid_pixels": int(ref_valid[:, t].sum()),
                         "transport_supported_pixels": int(q.sum()),
                         "transport_support_fraction": float(q.sum()/max(1, ref_valid[:, t].sum())),
                         "thick_cpp_valid_pixels": int((ref_valid[:, t] & (truth[:, t] >= 30)).sum()),
                         "thick_supported_pixels": int((q & (truth[:, t] >= 30)).sum()),
                         "paired_common_support": agg(q, truth[:, t],
                            {k: v[:, t] for k, v in scoring.items()})})
    regimes = {}
    for label, lo, hi in BINS:
        regimes[label] = agg(common & (truth >= lo) & (truth < hi), truth, scoring)
    regimes["all"] = agg(common, truth, scoring)
    regimes["thick_ge30"] = agg(common & (truth >= 30), truth, scoring)

    # Reuse the event classes selected by the station audit. Render CPP, model,
    # persistence, transport, and residual at each case's peak station lead.
    audit = json.loads(a.station_audit.read_text())
    selected = audit["selected_cases"]
    cases = []
    for kind, item in selected.items():
        row = int(item["row"])
        station_y = truth[row, :, 32, 32]
        station_ok = ref_valid[row, :, 32, 32]
        possible = np.flatnonzero(station_ok)
        peak_lead = int(possible[np.argmax(station_y[possible])]) if len(possible) else 0
        cases.append({"kind": kind, "row": row, "sequence_index": int(item["original_sequence_index"]),
                      "init_day_BJT": item["init_day_BJT"],
                      "reference_peak_cot": float(item["reference_peak_cot"]),
                      "lead_min": int(meta["lead_minutes"][peak_lead]), "lead_index": peak_lead,
                      "reference_valid_pixels": int(ref_valid[row, peak_lead].sum()),
                      "supported_valid_pixels": int(common[row, peak_lead].sum())})

    a.output.mkdir(parents=True, exist_ok=False)
    report = {"state": "COMPLETE_SILI_LOCAL64_VALIDATION_AUDIT", "test_used": False,
              "split": "validation", "station": "Sili", "station_pixel_local_rc": [32, 32],
              "checkpoint_epoch": int(ck["epoch"]), "cpp_reference_not_independent_truth": True,
              "transport_support_fraction_by_lead": [x["transport_support_fraction"] for x in per_lead],
              "zero_residual_equals_clipped_transport_on_common_support": exact_support_equal,
              "common_support_total_pixels": int(common.sum()),
              "cpp_valid_total_pixels": int(ref_valid.sum()),
              "per_lead": per_lead, "by_cpp_cot_regime_common_support": regimes,
              "cases": cases}
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "case_manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    case_rows = np.asarray([x["row"] for x in cases], dtype=np.int64)
    case_leads = np.asarray([x["lead_index"] for x in cases], dtype=np.int64)
    case_truth = np.stack([truth[r, t] for r, t in zip(case_rows, case_leads)])
    case_valid = np.stack([ref_valid[r, t] for r, t in zip(case_rows, case_leads)])
    case_common = np.stack([common[r, t] for r, t in zip(case_rows, case_leads)])
    case_pred = {k: np.stack([v[r, t] for r, t in zip(case_rows, case_leads)])
                 for k, v in scoring.items()}
    np.savez_compressed(a.output / "case_maps.npz", rows=case_rows, lead_indices=case_leads,
                        truth=case_truth, reference_valid=case_valid,
                        common_support=case_common, **case_pred)

    fig, axes = plt.subplots(len(cases), 6, figsize=(15, 2.8*len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[None, :]
    cols = ["CPP retrieval", "SimVP→R", "Persistence", "Transport / step 0", "Learned residual", "Learned − CPP"]
    for j, title in enumerate(cols):
        axes[0, j].set_title(title, fontsize=10)
    cot_im = diff_im = None
    for r, case in enumerate(cases):
        row, t = case["row"], case["lead_index"]
        target = np.ma.masked_where(~ref_valid[row, t], truth[row, t])
        fields = [target, predictions["simvp"][row, t], predictions["persistence"][row, t],
                  step0[row, t], predictions["learned"][row, t],
                  np.ma.masked_where(~ref_valid[row, t], predictions["learned"][row, t]-truth[row, t])]
        for c, field in enumerate(fields):
            ax = axes[r, c]
            if c < 5:
                im = ax.imshow(field, origin="upper", cmap="magma", vmin=0, vmax=100, interpolation="nearest")
                if c == 0:
                    cot_im = im
            else:
                im = ax.imshow(field, origin="upper", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50), interpolation="nearest")
                diff_im = im
            ax.scatter([32], [32], marker="+", color="cyan", s=45, linewidths=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{case['kind']}\n{case['init_day_BJT']}\n+{case['lead_min']} min", fontsize=8)
    fig.colorbar(cot_im, ax=axes[:, :5].ravel().tolist(), shrink=.75,
                 label="COT", location="bottom", pad=.02)
    fig.colorbar(diff_im, ax=axes[:, 5].ravel().tolist(), shrink=.75,
                 label="Learned − CPP COT", location="bottom", pad=.02)
    fig.suptitle("Sili local 64×64 COT cases — validation only; cyan cross = station pixel", fontsize=13)
    fig.savefig(a.output / "case_maps.png", dpi=180)
    fig.savefig(a.output / "case_maps.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(a.output), "cases": cases,
                      "support_exact_zero_residual_equals_transport": exact_support_equal,
                      "common_support_total_pixels": int(common.sum()),
                      "cpp_valid_total_pixels": int(ref_valid.sum())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
