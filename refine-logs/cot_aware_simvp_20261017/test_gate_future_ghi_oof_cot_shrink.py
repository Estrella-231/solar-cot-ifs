#!/usr/bin/env python3
"""Synthetic contract checks for the OOF AC50 follow-up gate."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from gate_future_ghi_oof_cot_shrink import evaluate


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pack_path, run_path, audit_path = (root / n for n in ("pack.json", "run.json", "audit.json"))
        run_output, audit_output = root / "new-run", root / "new-audit.json"
        pack = {"state": "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK",
                "train_forecast_oof": True, "forecast_bank_oof_fold_count": 5,
                "test_payload_opened": False, "buffer_payload_opened": False,
                "future_observed_agri_used_as_input": False, "cpp_used_as_input": False}
        pack_path.write_text(json.dumps(pack), encoding="utf-8")
        run = {"state": "COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT",
               "train_forecast_oof": True,
               "pack_manifest_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest()}
        run_path.write_text(json.dumps(run), encoding="utf-8")
        audit = {"state": "COMPLETE_PAIRED_FUTURE_GHI_COT_SOURCE_VALIDATION_AUDIT",
                 "train_forecast_oof": True, "test_payload_opened": False,
                 "run_manifest_sha256": "bound-to-run",
                 "pairwise_delta_rmse_wm2": {"AC_minus_AH": 0.2},
                 "date_block_bootstrap_95pct_ci_wm2": {"AC_minus_AH": [-0.1, 0.5]},
                 "per_station_mean_seed_rmse": {
                     "sili": {"AC": 101., "AH": 100.},
                     "zhujia": {"AC": 99., "AH": 100.}},
                 "lead_metrics": [
                     {"station": s, "lead_minutes": lead,
                      "AC_rmse_wm2": 101. if s == "sili" else 99.,
                      "AH_rmse_wm2": 100.}
                     for s in ("sili", "zhujia") for lead in (15, 30, 45, 60, 75)]}
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        result = evaluate(pack_path, run_path, audit_path, run_output, audit_output)
        assert result["ac50_followup_allowed"] is True
        assert result["station_gate_pass"] is False
        assert result["early_lead_equal_station_delta_wm2"] == 0.0

        audit["pairwise_delta_rmse_wm2"]["AC_minus_AH"] = -1.
        audit["date_block_bootstrap_95pct_ci_wm2"]["AC_minus_AH"] = [-1.2, -0.2]
        audit["per_station_mean_seed_rmse"] = {
            "sili": {"AC": 99., "AH": 100.},
            "zhujia": {"AC": 99., "AH": 100.}}
        for row in audit["lead_metrics"]:
            row["AC_rmse_wm2"] = 99.
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        try:
            evaluate(pack_path, run_path, audit_path, run_output, audit_output)
        except ValueError as exc:
            assert "gate passed" in str(exc)
        else:
            raise AssertionError("must refuse AC50 when the registered OOF gate passes")
    print("FUTURE_GHI_OOF_AC50_GATE_WITNESS_PASS")


if __name__ == "__main__":
    main()
