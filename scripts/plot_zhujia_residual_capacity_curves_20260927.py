"""Plot the saved, validation-only curves for the matched Zhujia residual pilot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "zhujia_residual_capacity_control_20260927"
OUT = RUN / "training_curves.png"
BASELINE_RMSE = 127.29536960923751


def read_curve(name: str) -> list[dict]:
    path = RUN / f"{name}_validation_curve.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    curves = {k: read_curve(k) for k in ("agri_residual", "cot_residual")}
    colors = {"agri_residual": "#0072B2", "cot_residual": "#D55E00"}
    labels = {"agri_residual": "AGRI-only residual", "cot_residual": "AGRI + forecast-COT residual"}
    fig, (ax_rmse, ax_loss) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True, layout="constrained")

    for key, records in curves.items():
        steps = np.asarray([r["step"] for r in records], dtype=int)
        rmse = np.asarray([r["val_rmse"] for r in records], dtype=float)
        loss = np.asarray([r["train_loss_mean_last100"] for r in records[1:]], dtype=float)
        loss_steps = steps[1:]
        ax_rmse.plot(steps, rmse, color=colors[key], label=labels[key], lw=1.8,
                     marker="o", ms=3, markevery=2)
        ax_loss.plot(loss_steps, loss, color=colors[key], label=labels[key], lw=1.8,
                     marker="s" if key == "agri_residual" else "^", ms=3, markevery=2)

    ax_rmse.axhline(BASELINE_RMSE, color="#333333", lw=1.2, ls="--",
                    label=f"Frozen no-COT baseline / COT selected step 0 ({BASELINE_RMSE:.2f})")
    ax_rmse.scatter([276], [126.32321890220607], color=colors["agri_residual"], marker="*", s=110,
                    edgecolor="white", linewidth=0.5, zorder=5, label="AGRI-only best: step 276")
    ax_rmse.set_ylabel("Validation GHI RMSE (W m$^{-2}$)")
    ax_rmse.set_ylim(120, 155)
    ax_rmse.grid(True, color="#d9d9d9", lw=0.7, alpha=0.8)
    ax_rmse.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax_rmse.set_title("Zhujia residual capacity control: saved training curves")

    ax_loss.set_ylabel("Training loss\n(last-100-update mean)")
    ax_loss.set_xlabel("Optimizer updates")
    ax_loss.set_xlim(0, 610)
    ax_loss.grid(True, color="#d9d9d9", lw=0.7, alpha=0.8)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=8.5)
    fig.text(0.01, 0.002,
             "Validation only; one seed; 1,624 paired Zhujia rows across 70 initialization dates. "
             "Lines show every saved epoch, with every second marker drawn. No uncertainty band (single run).",
             fontsize=8, color="#333333")

    fig.savefig(OUT, dpi=180, facecolor="white")
    # Matplotlib may emit RGBA even with an opaque white face; publish RGB explicitly.
    tmp = OUT.with_name("training_curves_rgb.tmp.png")
    with Image.open(OUT) as image:
        image.convert("RGB").save(tmp, dpi=image.info.get("dpi", (180, 180)))
    tmp.replace(OUT)
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    print(f"saved {OUT}")
    print(f"saved {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
