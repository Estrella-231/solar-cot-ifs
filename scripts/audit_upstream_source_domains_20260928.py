"""No-training source-domain support comparison on the fixed Sili val cohort.

Uses the already estimated last-two-frame motion vectors from 128 validation
sequences. It measures backward-source coordinate availability only; it does
not establish wind accuracy, AGRI validity, COT skill, or GHI skill.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/regional_station_cot_20260927"
OUT = RESULTS / "upstream_source_domain_20260928"
SILI_ROW, SILI_COL = 163, 148
SIZES = (64, 128, 256)


def support_for_points(size: int, shift_rc: list[float], lead: int, rr: np.ndarray, cc: np.ndarray):
    # Historical source = target - lead * observed-history shift.
    sy = rr - lead * float(shift_rc[0])
    sx = cc - lead * float(shift_rc[1])
    top = SILI_ROW - size // 2
    left = SILI_COL - size // 2
    return ((sy >= top) & (sy <= top + size - 1) &
            (sx >= left) & (sx <= left + size - 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((RESULTS / "sili_transport_variants/motion_vectors_v2.json").read_text(encoding="utf-8"))
    # Fixed 128-sequence validation pilot, independent of CPP/GHI availability.
    offsets = np.arange(-2, 3)
    rr, cc = np.meshgrid(SILI_ROW + offsets, SILI_COL + offsets, indexing="ij")
    rows = []
    for size in SIZES:
        for lead in range(1, 17):
            n_points = n_supported = n_all5 = n_station = n_full64 = 0
            for item in catalog:
                shift = item["last1"]
                valid = support_for_points(size, shift, lead, rr, cc)
                n_points += valid.size
                n_supported += int(valid.sum())
                n_all5 += int(valid.all())
                station = support_for_points(size, shift, lead,
                                             np.asarray([[SILI_ROW]]), np.asarray([[SILI_COL]]))
                n_station += int(station.item())
                r64 = np.arange(SILI_ROW - 32, SILI_ROW + 32)[:, None]
                c64 = np.arange(SILI_COL - 32, SILI_COL + 32)[None, :]
                local = support_for_points(size, shift, lead, r64, c64)
                n_full64 += int(local.all())
            rows.append({
                "source_domain_pixels": size,
                "lead_minutes": lead * 15,
                "sequences": len(catalog),
                "station_pixel_geometric_support_fraction": n_station / len(catalog),
                "station_5x5_pixelwise_geometric_support_fraction": n_supported / n_points,
                "records_with_all_station_5x5_sources_fraction": n_all5 / len(catalog),
                "records_with_entire_target_64x64_sources_fraction": n_full64 / len(catalog),
            })
    summary = {
        "test_used": False,
        "training_performed": False,
        "cohort": "same fixed 128 Sili validation sequences as the regional station COT pilot; includes sequences without valid CPP targets",
        "motion": "saved full-precision last-step regional C13 phase-correlation displacement from observed history; source coordinates are target minus lead times displacement",
        "station_grid_rc": [SILI_ROW, SILI_COL],
        "domain_definition": "64x64, 128x128, and 256x256 square source windows centred on the same station pixel",
        "interpretation_limit": "geometric coordinate inclusion only; ignores real AGRI validity, R error, cloud-layer wind, deformation, evolution, CPP accuracy, and GHI impact",
        "lead_metrics": rows,
        "at_240_min": [r for r in rows if r["lead_minutes"] == 240],
    }
    (OUT / "source_domain_support.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for size in SIZES:
        vals = [r for r in rows if r["source_domain_pixels"] == size]
        ax.plot([r["lead_minutes"] for r in vals],
                [r["records_with_all_station_5x5_sources_fraction"] * 100 for r in vals],
                marker="o", label=f"{size}×{size}")
    ax.set(xlabel="Lead from history end (min)", ylabel="Validation sequences with all 25 backtraced sources (%)",
           title="Larger upstream domains reduce station-neighbourhood source loss")
    ax.set_ylim(0, 102); ax.grid(alpha=.25); ax.legend(title="Source domain")
    fig.savefig(OUT / "source_domain_support.png", dpi=180)
    fig.savefig(OUT / "source_domain_support.pdf")
    print(json.dumps({"output": str(OUT), "at_240_min": summary["at_240_min"]}))


if __name__ == "__main__":
    main()
