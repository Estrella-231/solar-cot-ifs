#!/usr/bin/env python3
"""Summarize monthly R-COT in-sample vs date-held-out input shift.

This reads only the already-produced train-only per-frame audit CSV and its
scope receipt. It does not open AGRI, CPP labels/QC, validation predictions,
test, or buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REQUIRED = {
    "date_bjt", "day_pixels", "common_inrange_pixels",
    "full_r_common_inrange_mean", "oof_r_common_inrange_mean",
    "common_inrange_RMSE", "full_r_day_gt100", "oof_r_day_gt100",
}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-frame", type=Path, default=here.parent / "evidence/remote_215541/oof_train/evidence/independent_train_only_v3/per_frame.csv")
    ap.add_argument("--scope-summary", type=Path, default=here.parent / "evidence/remote_215541/oof_train/evidence/independent_train_only_v3/summary.json")
    ap.add_argument("--analysis-summary", type=Path, default=here.parent / "evidence/oof_history_input_origin_analysis.json")
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args()

    scope = json.loads(a.scope_summary.read_text(encoding="utf-8"))
    analysis = json.loads(a.analysis_summary.read_text(encoding="utf-8"))
    assert scope.get("state") == "COMPLETE_OOF_HISTORY_INPUT_DISTRIBUTION_DIAGNOSTIC"
    assert int(scope.get("validation_rows_inferred", -1)) == 0
    for key in ("test_payloads_read", "buffer_payloads_read", "cpp_target_or_quality_payloads_read", "ghi_labels_read"):
        assert int(scope.get(key, -1)) == 0, (key, scope.get(key))

    groups: dict[str, dict] = defaultdict(lambda: {
        "frames": 0, "dates": set(), "pairs": 0,
        "full_sum": 0.0, "oof_sum": 0.0, "squared_error_sum": 0.0,
        "full_gt100": 0, "oof_gt100": 0, "day_pixels": 0,
    })
    with a.per_frame.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        assert not missing, f"missing columns: {sorted(missing)}"
        for row in reader:
            day_pixels = int(row["day_pixels"] or 0)
            pairs = int(row["common_inrange_pixels"] or 0)
            if day_pixels <= 0 or pairs <= 0:
                continue
            month = row["date_bjt"][:6]
            full = float(row["full_r_common_inrange_mean"])
            oof = float(row["oof_r_common_inrange_mean"])
            rmse = float(row["common_inrange_RMSE"])
            assert all(math.isfinite(x) for x in (full, oof, rmse))
            g = groups[month]
            g["frames"] += 1
            g["dates"].add(row["date_bjt"])
            g["pairs"] += pairs
            g["full_sum"] += pairs * full
            g["oof_sum"] += pairs * oof
            g["squared_error_sum"] += pairs * rmse * rmse
            g["full_gt100"] += int(row["full_r_day_gt100"] or 0)
            g["oof_gt100"] += int(row["oof_r_day_gt100"] or 0)
            g["day_pixels"] += day_pixels

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for month, g in sorted(groups.items()):
        n = g["pairs"]
        rows.append({
            "month_bjt": month,
            "daylight_frames_with_common_support": g["frames"],
            "train_dates": len(g["dates"]),
            "common_inrange_pixel_pairs": n,
            "full_r_weighted_mean_cot": g["full_sum"] / n,
            "oof_r_weighted_mean_cot": g["oof_sum"] / n,
            "oof_minus_full_weighted_mean_cot": (g["oof_sum"] - g["full_sum"]) / n,
            "pooled_common_support_map_disagreement_rmse_cot": math.sqrt(g["squared_error_sum"] / n),
            "full_r_daytime_gt100_pct": 100.0 * g["full_gt100"] / g["day_pixels"],
            "oof_r_daytime_gt100_pct": 100.0 * g["oof_gt100"] / g["day_pixels"],
        })

    total_pairs = sum(g["pairs"] for g in groups.values())
    total_full = sum(g["full_sum"] for g in groups.values()) / total_pairs
    total_oof = sum(g["oof_sum"] for g in groups.values()) / total_pairs
    total_map_rmse = math.sqrt(sum(g["squared_error_sum"] for g in groups.values()) / total_pairs)
    assert total_pairs == int(analysis["common_daytime_inrange_pixels"])
    assert math.isclose(total_full, float(analysis["full_r_common_mean"]), rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(total_oof, float(analysis["oof_r_common_mean"]), rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(total_map_rmse, float(analysis["overall_pixel_weighted_map_RMSE"]), rel_tol=0.0, abs_tol=1e-8)

    csv_path = a.out / "monthly_shift.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Train-only monthly R-COT input-origin shift",
        "",
        "This descriptive CPU audit groups the existing date-held-out R-vs-full-R comparison by BJT month. It reads only the frozen train-only `per_frame.csv` audit output and its scope summary. No AGRI, CPP target/QC, validation prediction, test, or buffer payload is read.",
        "",
        "The values below use only common daytime pixels for which both R outputs are finite and in [0,100]. Means and disagreement RMSE are weighted by the number of such pixel pairs in each frame. Frames overlap in time and are not independent samples; no inferential interval is reported.",
        "",
        f"Aggregation check against the saved train-only analysis summary passed: {total_pairs:,} pixel pairs; full-R mean {total_full:.6f}; date-held-out R mean {total_oof:.6f}; pooled map-disagreement RMSE {total_map_rmse:.6f} COT.",
        "",
        "| BJT month | Train dates | Frames | Pixel pairs | Full-R mean | Date-held-out R mean | OOF − full | Map-disagreement RMSE | Full-R >100 (%) | OOF-R >100 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['month_bjt']} | {r['train_dates']} | {r['daylight_frames_with_common_support']} | {r['common_inrange_pixel_pairs']:,} | "
            f"{r['full_r_weighted_mean_cot']:.2f} | {r['oof_r_weighted_mean_cot']:.2f} | "
        f"{r['oof_minus_full_weighted_mean_cot']:+.2f} | {r['pooled_common_support_map_disagreement_rmse_cot']:.2f} | "
            f"{r['full_r_daytime_gt100_pct']:.3f} | {r['oof_r_daytime_gt100_pct']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The source gap varies by month: September–October show the largest negative OOF-minus-full mean differences, while February reverses sign in this train-only comparison. This is a seasonal feature-distribution warning for downstream models trained on full-R in-sample histories and applied to held-out R histories. It does not show which R output is physically more accurate, does not measure future-COT forecast skill, and does not prove that this shift caused the validation result.",
        "",
        "The monthly validation cohort remains frozen for tuning. The next model experiment, if later unlocked by the source/plan gate, should train its COT forecaster with date-held-out R histories on train dates and preserve one fixed monthly-validation evaluation; do not fit monthly offsets or weights to these validation dates.",
        "",
        f"Source audit state: {scope['state']}; train rows={scope['train_rows']}; OOF rows={scope['oof_rows']}; validation rows inferred={scope['validation_rows_inferred']}; test/buffer/CPP-target/QC/GHI-label reads=0.",
        f"CSV: `{csv_path.name}`. Reproduce with `python {Path(__file__).name}` from this directory.",
    ]
    report_path = a.out / "MONTHLY_SHIFT_AUDIT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
