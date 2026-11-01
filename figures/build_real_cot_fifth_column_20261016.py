"""Build a five-column paired COT trajectory using the existing rendered maps.

The raw validation packs are not present in this local checkout. The two source
figures were rendered from the same frozen trajectory; this script reuses their
map panels and verifies the matching sample/time manifests before composition.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FORECAST_DIR = ROOT / "cpp_vs_forecast_cot_trajectory_20260922"
REAL_DIR = ROOT / "cpp_vs_real_agri_cot_trajectory_20260922"
OUT_DIR = ROOT / "forecast_vs_real_retrieved_cot_trajectory_20261016_v2"


def load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def crop_square(image, box):
    crop = image.crop(box)
    if crop.width != crop.height:
        side = min(crop.size)
        left = (crop.width - side) // 2
        top = (crop.height - side) // 2
        crop = crop.crop((left, top, left + side, top + side))
    return np.asarray(crop)


def main():
    fm = load_manifest(FORECAST_DIR / "manifest.json")
    rm = load_manifest(REAL_DIR / "manifest.json")
    shared = ("station", "sequence", "candidate_count", "init_time_bjt", "target_time_bjt", "leads_minutes")
    mismatches = {key: (fm.get(key), rm.get(key)) for key in shared if fm.get(key) != rm.get(key)}
    if mismatches:
        raise RuntimeError(f"Source figures do not show the same trajectory: {mismatches}")
    if fm["state"] != "COMPLETE_VALIDATION_CPP_VS_FORECAST_COT_TRAJECTORY_VISUAL":
        raise RuntimeError(f"Unexpected forecast source state: {fm['state']}")
    if rm["state"] != "COMPLETE_VALIDATION_CPP_VS_REAL_AGRI_COT_TRAJECTORY_VISUAL":
        raise RuntimeError(f"Unexpected real-AGRI source state: {rm['state']}")

    forecast_img = Image.open(FORECAST_DIR / "cpp_vs_forecast_cot_trajectory.png").convert("RGB")
    real_img = Image.open(REAL_DIR / "cpp_vs_real_agri_cot_trajectory.png").convert("RGB")
    if forecast_img.size != (3330, 2580) or real_img.size != (2700, 2580):
        raise RuntimeError(f"Unexpected source image dimensions: {forecast_img.size}, {real_img.size}")

    # The four left panels are copied from the forecast-vs-CPP rendering;
    # the fifth is the frozen R(observed future AGRI) panel from its paired
    # same-sample rendering. All crops contain only the map axes, not labels.
    row_y_forecast = ((193, 725), (798, 1330), (1407, 1939), (2010, 2542))
    x_forecast = ((63, 598), (620, 1152), (1176, 1707), (1731, 2265))
    row_y_real = ((193, 709), (801, 1317), (1408, 1924), (2017, 2533))
    x_real_cot = (1140, 1656)
    columns = []
    for y_f, y_r in zip(row_y_forecast, row_y_real):
        row = [crop_square(forecast_img, (x0, y_f[0], x1, y_f[1])) for x0, x1 in x_forecast]
        row.append(crop_square(real_img, (x_real_cot[0], y_r[0], x_real_cot[1], y_r[1])))
        columns.append(row)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 5, figsize=(13.0, 10.0))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.085, wspace=0.10, hspace=0.13)
    titles = (
        "Observed future AGRI\n(false colour)",
        "SimVP forecast AGRI\n(false colour)",
        "CPP COT reference",
        "Forecast COT\nR(SimVP AGRI)",
        "Real-image retrieved COT\nR(observed future AGRI)",
    )
    for r in range(4):
        for c in range(5):
            axes[r, c].imshow(columns[r][c], interpolation="nearest")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_linewidth(0.6)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f"+{fm['leads_minutes'][r]} min", fontsize=9)

    fig.suptitle(f"Validation COT trajectory: {fm['station']}, init {fm['init_time_bjt']}", fontsize=13)
    fig.text(
        0.5, 0.018,
        "COT maps use a common 0–8.04 physical scale; stars mark patch [8,8]. "
        "R(observed future AGRI) is an offline diagnostic, unavailable at forecast issue time. "
        "CPP is a retrieval reference, not independent physical truth.",
        ha="center", fontsize=8, color="#444444",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"forecast_vs_real_retrieved_cot_trajectory.{ext}", dpi=300)
    plt.close(fig)

    manifest = {
        "state": "COMPLETE_PAIRED_RENDERED_COT_COMPARISON",
        "station": fm["station"],
        "sequence": fm["sequence"],
        "init_time_bjt": fm["init_time_bjt"],
        "target_time_bjt": fm["target_time_bjt"],
        "leads_minutes": fm["leads_minutes"],
        "columns": [
            "observed future AGRI false-colour",
            "SimVP forecast AGRI false-colour",
            "CPP COT retrieval reference",
            "frozen R(SimVP forecast AGRI)",
            "frozen R(observed future AGRI), offline diagnostic only",
        ],
        "cot_scale": "shared source-rendered physical COT scale 0-8.04",
        "limits": [
            "fifth column uses future observed AGRI and is not causally available at forecast issue time",
            "CPP is a retrieval reference, not independent physical truth",
            "composition reuses matching map panels from the two source figures; it does not rerun retrieval",
        ],
        "source_manifests": {
            "forecast_vs_cpp": str(FORECAST_DIR / "manifest.json"),
            "cpp_vs_real_agri": str(REAL_DIR / "manifest.json"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
