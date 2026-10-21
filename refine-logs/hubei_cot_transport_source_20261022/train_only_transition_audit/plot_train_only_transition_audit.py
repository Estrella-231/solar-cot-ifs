#!/usr/bin/env python3
"""Plot the train-only station-center COT transition diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SITES = ("Niushou", "Wangzhai", "Mizhuang")
COLORS = {"Niushou": "#0072B2", "Wangzhai": "#D55E00", "Mizhuang": "#009E73"}
MARKERS = {"Niushou": "o", "Wangzhai": "s", "Mizhuang": "^"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data.get("state") != "TRAIN_ONLY_CPP_REFERENCE_COT_TRANSITION_AUDIT":
        raise ValueError("input is not the expected train-only transition audit")
    if any(data.get(k) != 0 for k in ("test_payloads_read", "buffer_payloads_read",
                                      "validation_model_or_label_values_read")):
        raise ValueError("scope receipt does not confirm held-out value isolation")

    leads = sorted(int(k) for k in data["transition_counts_by_lead"])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / "train_only_transition_audit_attempt1.png",
               out / "train_only_transition_audit_attempt1.pdf",
               out / "train_only_transition_audit_attempt1_manifest.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("refusing to overwrite an existing figure or manifest")

    with mpl.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.75))

        ax = axes[0]
        for site in SITES:
            values = data["site_by_lead"][site]
            rmse = [values[str(lead)]["history_persistence_rmse"] for lead in leads]
            ax.plot(leads, rmse, color=COLORS[site], marker=MARKERS[site],
                    linewidth=1.7, markersize=3.8, label=site)
        ax.set_title("Historical persistence error")
        ax.set_xlabel("Forecast lead (min)")
        ax.set_ylabel("RMSE vs future CPP COT")
        ax.set_xlim(10, 245)
        ax.set_ylim(0, 16)
        ax.set_xticks([15, 60, 120, 180, 240])
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.6)
        ax.legend(frameon=False, ncols=1, loc="lower left")

        ax = axes[1]
        specs = (
            ("thick_arrival_or_growth", "History <30 → future CPP ≥30", "#D55E00", "o"),
            ("thick_retained", "History ≥30 → future CPP ≥30", "#0072B2", "s"),
            ("thick_decay", "History ≥30 → future CPP <30", "#009E73", "^"),
        )
        for key, label, color, marker in specs:
            rate = []
            for lead in leads:
                counts = data["transition_counts_by_lead"][str(lead)]
                total = sum(counts.values())
                rate.append(100.0 * counts[key] / total if total else float("nan"))
            ax.plot(leads, rate, color=color, marker=marker, linewidth=1.6,
                    markersize=3.5, label=label)
        ax.set_title("Thick-threshold transitions at station pixels")
        ax.set_xlabel("Forecast lead (min)")
        ax.set_ylabel("Share of valid center records (%)")
        ax.set_xlim(10, 245)
        ax.set_ylim(0, 14)
        ax.set_xticks([15, 60, 120, 180, 240])
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.6)
        ax.legend(frameon=False, loc="lower right")

        fig.suptitle("Train-only Hubei station-center COT diagnostic", fontsize=11, y=0.97)
        fig.text(0.5, 0.012,
                 "Train split; 16×16 cache; frozen-R history vs CPP retrieval target.\n"
                 "Pairs overlap; CPP is not independent truth. No validation/test/buffer values read.",
                 ha="center", va="bottom", fontsize=6.8, color="#444444", linespacing=1.15)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.83, wspace=0.18)
        fig.savefig(targets[0], dpi=300, facecolor="white", bbox_inches=None)
        fig.savefig(targets[1], facecolor="white", bbox_inches=None)
        plt.close(fig)

    manifest = {
        "input": str(args.input),
        "source_state": data["state"],
        "scope": data["scope"],
        "transforms": [
            "per-lead site-center persistence RMSE read from FP64 audit JSON",
            "threshold event rate = event count / all valid station-center records at each lead",
        ],
        "uncertainty": "none; descriptive train-only overlapping records, no independent-event inference",
        "held_out_value_reads": {
            "validation": data["validation_model_or_label_values_read"],
            "test": data["test_payloads_read"],
            "buffer": data["buffer_payloads_read"],
        },
        "outputs": [p.name for p in targets[:2]],
        "alt_text": (
            "Two-panel train-only plot. Historical persistence RMSE against future CPP COT increases with lead at all "
            "three Hubei station pixels. The share crossing from historical COT below 30 to future CPP COT at least "
            "30 rises from about 2.3 percent at 15 minutes to 6.6 percent at four hours, while thick-cloud retention "
            "falls. Records overlap and CPP is a retrieval reference, not independent truth."
        ),
    }
    targets[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in targets]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
