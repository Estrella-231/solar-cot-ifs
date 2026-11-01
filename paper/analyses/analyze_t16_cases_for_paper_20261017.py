from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/monthly_R_station16_20261013_v1/results/t16_partial_cases_v1/t16_case_figure_audit.json"
OUT = Path(__file__).resolve().parent / "t16_event_cases_20261017"


def main() -> None:
    audit = json.loads(SOURCE.read_text(encoding="utf-8"))
    if audit.get("state") != "PASS_PARTIAL_T16_JJA_PHYSICAL_VALIDATION_FIGURES":
        raise ValueError("unexpected T16 case audit state")
    if audit.get("test_payload_opened") is not False:
        raise ValueError("T16 audit does not confirm the validation-only boundary")
    if audit.get("case_selection_used_GHI_values_or_errors") is not False:
        raise ValueError("cases were not selected independently of GHI errors")
    if len(audit.get("cases", [])) != 4:
        raise ValueError("expected four frozen JJA cases")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in audit["cases"]:
        a = float(case["case_A_mean_seed_RMSE_wm2"])
        ac = float(case["case_AC_mean_seed_RMSE_wm2"])
        rows.append({
            "case_id": case["case_id"],
            "station": case["station"],
            "init_time_BJT": case["init_time_BJT"],
            "selection_type": case["case_type"],
            "mean_abs_cpp_cot_change_per_15min": case["D_mean_abs_COT_change"],
            "common_cpp_pixels": case["common_CPP_pixels"],
            "A_seed_equal_within_case_RMSE_wm2": a,
            "AC_seed_equal_within_case_RMSE_wm2": ac,
            "AC_minus_A_RMSE_wm2": ac - a,
            "case_figure": case["case_png"],
        })
    with (OUT / "case_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# T16 event cases used in the manuscript",
        "",
        "Source: frozen validation audit `t16_case_figure_audit.json`. Case keys were selected before GHI values/prediction errors were inspected. The audit reports the mean of the three seed-specific within-case RMSEs over the 16 canonical future targets (+15 to +240 min). Negative AC−A favours AGRI + COT. These four temporally adjacent validation sequences are descriptive cases, not independent samples and have no event-level confidence intervals.",
        "",
        "| Station | Init (BJT) | Selection | D (COT / 15 min) | AGRI RMSE | AGRI + COT RMSE | Δ (AC−A) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['station']} | {r['init_time_BJT']} | {r['selection_type']} | "
            f"{float(r['mean_abs_cpp_cot_change_per_15min']):.3f} | "
            f"{float(r['A_seed_equal_within_case_RMSE_wm2']):.3f} | "
            f"{float(r['AC_seed_equal_within_case_RMSE_wm2']):.3f} | "
            f"{float(r['AC_minus_A_RMSE_wm2']):+.3f} |"
        )
    lines += [
        "",
        "The COT-augmented head has lower descriptive RMSE in one case and higher RMSE in three. This does not change the paired cohort result or establish an event-level causal mechanism. All four examples are from JJA because the available CPP archive does not provide complete 17-frame sequences for the other seasons. The figure shows CPP reference COT and R-retrieved COT as separate fields; its GHI curves use observed AGRI and are not forecast results.",
        "",
        "Figure source and rendering audit: `figures/t16_partial_cases_20261013/`; T16 case audit confirms `test_payload_opened=false`.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
