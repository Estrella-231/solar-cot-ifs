#!/usr/bin/env python3
"""Map paired validation GHI changes to actual COT gate locations.

Uses only the saved validation forensic NPZ and paired sample CSV; no test data.
Writes an overlap-aware station 3x3 table, figure, and short Markdown report.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/gate_ghi_forensics_20261015"
OUT = ROOT / "results/gate_point_ghi_correspondence_20261015"


def metric_row(name, mask, y, raw, gate):
    n = int(mask.sum())
    if not n:
        return {"group": name, "n": 0}
    dse = (gate[mask] - y[mask]) ** 2 - (raw[mask] - y[mask]) ** 2
    return {
        "group": name,
        "n": n,
        "worsened_fraction": float(np.mean(dse > 0)),
        "raw_rmse": float(np.sqrt(np.mean((raw[mask] - y[mask]) ** 2))),
        "gate_rmse": float(np.sqrt(np.mean((gate[mask] - y[mask]) ** 2))),
        "delta_rmse": float(
            np.sqrt(np.mean((gate[mask] - y[mask]) ** 2))
            - np.sqrt(np.mean((raw[mask] - y[mask]) ** 2))
        ),
        "delta_sse": float(dse.sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT / "paired_sample_diagnostics.csv")
    z = np.load(INPUT / "validation_forensics.npz", allow_pickle=False)
    if len(df) != len(z["utc"]):
        raise ValueError("CSV and NPZ row counts differ")
    if not np.array_equal(df["target_time_utc"].astype(str).values, z["utc"]):
        raise ValueError("CSV/NPZ UTC keys are not in identical order")
    if not np.array_equal(df["station"].values, z["station"]):
        raise ValueError("CSV/NPZ station keys are not in identical order")

    rows = []
    per_pixel = []
    station_data = {}
    for station in ("sili", "zhujia"):
        s = z["station"] == station
        y = z["ghi"][s]
        raw = z["raw_saved"][s]
        gate = z["gate_saved"][s]
        cleared = z["cleared"][s]
        falseclear = z["falseclear"][s]
        center = cleared[:, 7:10, 7:10]
        center_falseclear = falseclear[:, 7:10, 7:10]
        n_center = center.sum((1, 2))
        n_patch = cleared.sum((1, 2))
        n_falsecenter = center_falseclear.sum((1, 2))
        masks = {
            "all_validation": np.ones(len(y), dtype=bool),
            "any_gate_in_GHI_3x3": n_center > 0,
            "gate_only_outside_GHI_3x3": (n_center == 0) & (n_patch > 0),
            "no_gate_anywhere": n_patch == 0,
            "CPP_confirmed_cloud_cleared_in_3x3": n_falsecenter > 0,
            "3x3_gate_without_CPP_confirmed_cloud": (n_center > 0)
            & (n_falsecenter == 0),
        }
        for label, mask in masks.items():
            row = metric_row(label, mask, y, raw, gate)
            row.update({"station": station, "comparison": "separately_trained_raw_vs_gate_heads"})
            rows.append(row)

        # Same trained head, same normalization: change only the COT input.
        for label, mask in masks.items():
            p_raw = z["gatehead_rawinput"][s]
            p_gate = z["gatehead_gateinput"][s]
            row = metric_row(label, mask, y, p_raw, p_gate)
            row.update({"station": station, "comparison": "fixed_gate_head_gate_input_vs_raw_input"})
            rows.append(row)

        station_data[station] = {
            "count": center.sum(0),
            "falseclear_count": center_falseclear.sum(0),
            "associated_delta_sse": np.array(
                [
                    ((gate[center[:, rr, cc]] - y[center[:, rr, cc]]) ** 2
                     - (raw[center[:, rr, cc]] - y[center[:, rr, cc]]) ** 2).sum()
                    for rr in range(3)
                    for cc in range(3)
                ]
            ).reshape(3, 3),
        }
        for rr in range(3):
            for cc in range(3):
                mask = center[:, rr, cc]
                if not mask.any():
                    continue
                dse = (gate[mask] - y[mask]) ** 2 - (raw[mask] - y[mask]) ** 2
                per_pixel.append(
                    {
                        "station": station,
                        "row": rr + 7,
                        "col": cc + 7,
                        "n_samples_pixel_cleared": int(mask.sum()),
                        "n_CPP_confirmed_cloud_pixel_cleared": int(
                            center_falseclear[:, rr, cc].sum()
                        ),
                        "paired_worsened_fraction": float(np.mean(dse > 0)),
                        "paired_delta_sse_associated_not_additive": float(dse.sum()),
                    }
                )

    pd.DataFrame(rows).to_csv(OUT / "group_comparison.csv", index=False)
    pixel_df = pd.DataFrame(per_pixel)
    pixel_df.to_csv(OUT / "center3x3_gate_pixel_association.csv", index=False)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for r, station in enumerate(("sili", "zhujia")):
        data = station_data[station]
        for c, (key, title, fmt) in enumerate(
            [
                ("count", "Gate-cleared samples at each pixel", "d"),
                ("associated_delta_sse", "Paired ΔSSE associated with gate at pixel", ".0f"),
            ]
        ):
            ax = axs[r, c]
            arr = data[key]
            im = ax.imshow(arr, cmap="magma" if c == 0 else "RdBu_r")
            ax.set_xticks(range(3), [7, 8, 9])
            ax.set_yticks(range(3), [7, 8, 9])
            ax.set_xlabel("column in 16×16 patch")
            ax.set_ylabel("row in 16×16 patch")
            ax.set_title(f"{station.title()} · {title}")
            for rr in range(3):
                for cc in range(3):
                    ax.text(cc, rr, format(arr[rr, cc], fmt), ha="center", va="center", color="white", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(
        "Actual COT gate points and paired GHI error changes\n"
        "Pixel-associated ΔSSE overlaps across samples; it is not additive attribution",
        fontsize=14,
    )
    fig.savefig(OUT / "gate_point_ghi_correspondence.png", dpi=180)
    plt.close(fig)

    # Human-readable headline values.
    report = [
        "# GHI degradation mapped to actual COT gate points",
        "",
        "This is a validation-only analysis of the saved paired seed42 outputs. It uses no test payload.",
        "The station GHI head reads only rows/columns 7–9 of the 16×16 COT patch (center 3×3).",
        "",
        "## Grouped paired result",
        "",
        "ΔRMSE is gate minus raw, in W/m². ΔSSE is the sum of per-sample squared-error changes.",
        "",
        "| Station | Comparison | Gate in center 3×3 | N | Worse fraction | ΔRMSE | ΔSSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    results = pd.DataFrame(rows)
    for station in ("sili", "zhujia"):
        for comparison in (
            "separately_trained_raw_vs_gate_heads",
            "fixed_gate_head_gate_input_vs_raw_input",
        ):
            for group in ("any_gate_in_GHI_3x3", "gate_only_outside_GHI_3x3", "no_gate_anywhere"):
                q = results[(results.station == station) & (results.comparison == comparison) & (results.group == group)].iloc[0]
                report.append(
                    f"| {station} | {comparison} | {group} | {int(q.n)} | {q.worsened_fraction:.3f} | {q.delta_rmse:+.3f} | {q.delta_sse:+.1f} |"
                )
    report += [
        "",
        "CPP confirms gate-cleared cloudy pixels in the center 3×3 for only 13 Sili samples and 17 Zhujia samples. Those are subsets of the gated-center group; missing CPP is unknown, not clear.",
        "The fixed-head comparison changes only the COT input and therefore isolates the input intervention for this frozen head. The separately retrained-head comparison also contains training/normalization differences.",
        "",
        "## Point map",
        "",
        "![Gate point correspondence](gate_point_ghi_correspondence.png)",
        "",
        "The per-pixel ΔSSE map includes every sample where that pixel was cleared. A sample can contribute to several pixels, so these cells overlap and must not be summed as causal contributions. See `center3x3_gate_pixel_association.csv` for the exact counts.",
        "",
        "## Representative deterioration cases",
        "",
        "- Sili, 2025-07-08 14:45 BJT: observation 462.0, raw head 680.4, gate head 716.5 W/m²; Δsquared error +17,071. The center gate cleared (8,8), (8,9), (9,8), (9,9), with no CPP-confirmed cloudy pixel among them.",
        "- Zhujia, 2024-09-04 14:00 BJT: observation 433.3, raw head 617.8, gate head 813.8 W/m²; Δsquared error +110,721. Eight center pixels were cleared; CPP marked all available cleared locations clear.",
        "- The largest individual regression with an actual CPP-confirmed cloudy center pixel was Sili 2025-07-08 14:30: the point (7,9) had R COT 2.66 versus CPP COT 6.99; observed/raw/gate GHI were 527.7/754.3/779.7 W/m².",
        "",
        "## Interpretation boundary",
        "",
        "Gate-induced input changes are associated with some GHI error increases, especially where the gate changes pixels read by the head. However, most CPP-confirmed false-clear samples are not the dominant source of the aggregate regression. Separate retraining also changes outputs on samples whose COT was not gated. This analysis identifies where to inspect; it does not establish a physical cause.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT)
    print(pd.DataFrame(rows).query("comparison == 'separately_trained_raw_vs_gate_heads' and group == 'all_validation'").to_string(index=False))


if __name__ == "__main__":
    main()
