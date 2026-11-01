"""Show real AGRI contexts for the same-pixel COT recrop diagnostic."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cot_staged_diagnostic_20260924/source_cases"
OUT = SOURCE / "border_context_audit_20260925"


def main():
    report = json.loads((OUT / "same_pixel_recrop_validation.json").read_text())
    rgb_limits = json.loads((SOURCE / "four_case_temporal_rgb_limits.json").read_text())
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.5), constrained_layout=True)
    for row, case in enumerate(report["cases"]):
        full = np.load(SOURCE / "original_agri_20ch" / f"{case['timestamp_utc']}.npy", mmap_mode="r")
        for col, variant in enumerate(case["variants"]):
            r0, c0 = variant["crop_origin_row"], variant["crop_origin_col"]
            channels = full[[2, 1, 0], r0:r0+16, c0:c0+16]
            rgb = np.stack([np.clip((channels[i] - rgb_limits[str(ch)][0]) /
                                    (rgb_limits[str(ch)][1] - rgb_limits[str(ch)][0]), 0, 1)
                            for i, ch in enumerate((2, 1, 0))], axis=-1)
            axis = axes[row, col]
            axis.imshow(rgb, interpolation="nearest", origin="upper")
            axis.scatter([variant["pixel_local_col"]], [variant["pixel_local_row"]],
                         marker="x", c="red", s=100, linewidths=2)
            axis.set(title=f"Case {case['case_id']} | pixel row {variant['pixel_local_row']}\n"
                           f"P(cloud)={variant['probability']:.3f} | COT={variant['final_cot']:.1f}",
                     xticks=[0, 4, 8, 12, 15], yticks=[0, 4, 8, 12, 15])
            axis.grid(color="white", alpha=.35, linewidth=.6)
    fig.suptitle("Same observed AGRI pixel, shifted 16×16 crop (R=C03, G=C02, B=C01)")
    fig.savefig(OUT / "same_pixel_recrop_rgb.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
