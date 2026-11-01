#!/usr/bin/env python3
"""Synthetic-only tests for the fail-closed Hubei test evaluator."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from evaluate_hubei_forecast_heads_frozen_test import (
    ARMS, SEEDS, STATIONS, CANONICAL_TEST_LEDGER_PATH,
    canonical_dates_sha256, canonical_keys_sha256,
    evaluate, sha256_file,
)


class FrozenTestEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_root = self.root / "synthetic_predictions"
        self.output = self.root / "scores.json"
        self.receipt_path = self.root / "release.json"
        self.split_path = self.root / "frozen_split.csv"
        self.truth_manifest_path = self.root / "truth_manifest.json"
        self.ledger_path = self.root / "canonical-test-ledger.json"
        self.ledger_patch = patch(
            "evaluate_hubei_forecast_heads_frozen_test.CANONICAL_TEST_LEDGER_PATH",
            self.ledger_path,
        )
        self.ledger_patch.start()
        self.addCleanup(self.ledger_patch.stop)
        self.synthetic_root_patch = patch(
            "evaluate_hubei_forecast_heads_frozen_test.SYNTHETIC_ONLY_ROOT",
            self.root,
        )
        self.synthetic_root_patch.start()
        self.addCleanup(self.synthetic_root_patch.stop)
        self.script_sha = sha256_file(Path(__file__).with_name("evaluate_hubei_forecast_heads_frozen_test.py"))
        self.dates = ["20260601", "20260602"]
        self.split_path.write_text(
            "date_BJT,split\n20260530,train\n20260601,test\n20260602,test\n",
            encoding="utf-8",
        )
        self.expected_keys = [
            (f"{d}0800", f"{d}{8 + lead // 60:02d}{lead % 60:02d}", lead)
            for d in self.dates for lead in range(15, 241, 15)
        ]
        self.write_truth_bundle("Niushou")
        self.write_truth_bundle("Wangzhai")
        self.write_truth_bundle("Mizhuang")
        manifest = {
            "state": "HUBEI_INDEPENDENT_TRUTH_MANIFEST_V1",
            "stations": {
                station: {"path": f"truth_{station}.npz", "sha256": sha256_file(self.root / f"truth_{station}.npz")}
                for station in STATIONS
            },
        }
        self.truth_manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        self.receipt = {
            "state": "HUBEI_TEST_EVALUATION_RELEASE_PASS",
            "evaluation_scope": "SYNTHETIC_ONLY",
            "test_unseal_verdict": "PASS",
            "test_payloads_read": 0,
            "buffer_payloads_read": 0,
            "split_sha256": sha256_file(self.split_path),
            "test_pack_manifest_sha256": sha256_file(self.truth_manifest_path),
            "model_manifest_sha256": "c" * 64,
            "test_target_dates_bjt": self.dates,
            "test_target_dates_sha256": canonical_dates_sha256(self.dates),
            "expected_key_count_by_station": {station: len(self.expected_keys) for station in STATIONS},
            "expected_keys_sha256_by_station": {
                station: canonical_keys_sha256(self.expected_keys) for station in STATIONS
            },
            "evaluator_sha256": self.script_sha,
            "stations": list(STATIONS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
        }
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_truth_bundle(self, station: str, *, support_dates: list[str] | None = None,
                           shift_date: str | None = None) -> None:
        dates = support_dates or self.dates
        inits: list[str] = []
        targets: list[str] = []
        leads: list[int] = []
        for day in dates:
            init = datetime.strptime(f"{day}0800", "%Y%m%d%H%M")
            if day == shift_date:
                init += timedelta(minutes=15)
            for lead in range(15, 241, 15):
                inits.append(init.strftime("%Y%m%d%H%M"))
                targets.append((init + timedelta(minutes=lead)).strftime("%Y%m%d%H%M"))
                leads.append(lead)
        np.savez_compressed(
            self.root / f"truth_{station}.npz",
            station=np.asarray([station] * len(leads)),
            truth_ghi_wm2=np.full(len(leads), 300.0),
            clear_sky_ghi_wm2=np.full(len(leads), 800.0),
            init_time_bjt=np.asarray(inits), target_time_bjt=np.asarray(targets),
            lead_minutes=np.asarray(leads, dtype=np.int16),
        )

    def write_bundle(self, seed: int, station: str, arm: str, *, target_date: str = "20260601",
                     target_delta_minutes: int = 15, station_value: str | None = None,
                     support_dates: list[str] | None = None,
                     shift_date: str | None = None) -> None:
        path = self.run_root / f"seed{seed}" / station / f"{arm}_test_predictions_with_keys.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        actual_station = station_value or station
        if support_dates is not None:
            dates = support_dates
        else:
            dates = ["20260601", "20260602"] if target_date == "20260601" else [target_date, target_date]
        inits: list[str] = []
        targets: list[str] = []
        leads: list[int] = []
        for day in dates:
            init = datetime.strptime(f"{day}0800", "%Y%m%d%H%M")
            if day == shift_date:
                init += timedelta(minutes=15)
            for lead in range(15, 241, 15):
                inits.append(init.strftime("%Y%m%d%H%M"))
                targets.append((init + timedelta(minutes=lead + target_delta_minutes - 15)).strftime("%Y%m%d%H%M"))
                leads.append(lead)
        np.savez_compressed(
            path,
            station=np.asarray([actual_station] * len(leads)),
            pred_ghi_wm2=np.full(len(leads), 280.0),
            clear_sky_ghi_wm2=np.full(len(leads), 800.0),
            init_time_bjt=np.asarray(inits),
            target_time_bjt=np.asarray(targets),
            lead_minutes=np.asarray(leads, dtype=np.int16),
        )

    def fill_complete_fixture(self) -> None:
        for seed in SEEDS:
            for station in STATIONS:
                for arm in ARMS:
                    self.write_bundle(seed, station, arm)

    def fill_sparse_common_fixture(self) -> None:
        """Build exact, paired support with only selected valid forecast leads."""
        sparse_keys = [key for key in self.expected_keys if key[2] in (15, 60, 240)]
        for station in STATIONS:
            np.savez_compressed(
                self.root / f"truth_{station}.npz",
                station=np.asarray([station] * len(sparse_keys)),
                truth_ghi_wm2=np.full(len(sparse_keys), 300.0),
                clear_sky_ghi_wm2=np.full(len(sparse_keys), 800.0),
                init_time_bjt=np.asarray([key[0] for key in sparse_keys]),
                target_time_bjt=np.asarray([key[1] for key in sparse_keys]),
                lead_minutes=np.asarray([key[2] for key in sparse_keys], dtype=np.int16),
            )
        for seed in SEEDS:
            for station in STATIONS:
                for arm in ARMS:
                    path = self.run_root / f"seed{seed}" / station / f"{arm}_test_predictions_with_keys.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        station=np.asarray([station] * len(sparse_keys)),
                        pred_ghi_wm2=np.full(len(sparse_keys), 280.0),
                        clear_sky_ghi_wm2=np.full(len(sparse_keys), 800.0),
                        init_time_bjt=np.asarray([key[0] for key in sparse_keys]),
                        target_time_bjt=np.asarray([key[1] for key in sparse_keys]),
                        lead_minutes=np.asarray([key[2] for key in sparse_keys], dtype=np.int16),
                    )
        manifest = {
            "state": "HUBEI_INDEPENDENT_TRUTH_MANIFEST_V1",
            "stations": {
                station: {"path": f"truth_{station}.npz", "sha256": sha256_file(self.root / f"truth_{station}.npz")}
                for station in STATIONS
            },
        }
        self.truth_manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        self.receipt["test_pack_manifest_sha256"] = sha256_file(self.truth_manifest_path)
        self.receipt["expected_key_count_by_station"] = {station: len(sparse_keys) for station in STATIONS}
        self.receipt["expected_keys_sha256_by_station"] = {
            station: canonical_keys_sha256(sparse_keys) for station in STATIONS
        }
        self.receipt["evaluator_sha256"] = sha256_file(Path(__file__).with_name("evaluate_hubei_forecast_heads_frozen_test.py"))
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def test_invalid_release_fails_before_any_bundle_open(self) -> None:
        self.receipt["test_unseal_verdict"] = "FAIL_NO_GO"
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with patch("evaluate_hubei_forecast_heads_frozen_test.np.load", side_effect=AssertionError("payload opened before gate")):
            with self.assertRaisesRegex(ValueError, "not PASS"):
                evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())

    def test_external_paths_are_rejected_before_receipt_or_payload_read(self) -> None:
        outside = self.root.parent / f"{self.root.name}_outside"
        outside.mkdir()
        outside_receipt = outside / "release.json"
        outside_receipt.write_text("not json", encoding="utf-8")
        with patch("evaluate_hubei_forecast_heads_frozen_test.np.load", side_effect=AssertionError("payload opened")):
            with self.assertRaisesRegex(ValueError, "outside fixture root"):
                evaluate(self.run_root, self.output, outside_receipt, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())

    def test_wrong_evaluator_hash_fails_closed(self) -> None:
        self.receipt["evaluator_sha256"] = "d" * 64
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact evaluator"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)

    def test_wrong_split_fails_closed(self) -> None:
        wrong_split = self.root / "wrong_split.csv"
        wrong_split.write_text("date_BJT,split\n20260601,train\n20260602,test\n", encoding="utf-8")
        with patch("evaluate_hubei_forecast_heads_frozen_test.np.load", side_effect=AssertionError("payload opened before gate")):
            with self.assertRaisesRegex(ValueError, "split SHA"):
                evaluate(self.run_root, self.output, self.receipt_path, wrong_split, self.truth_manifest_path, 100)

    def test_refuses_live_scope_even_with_pass_receipt(self) -> None:
        self.receipt["evaluation_scope"] = "INDEPENDENT_TEST"
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "synthetic-only"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())

    def test_accepts_complete_synthetic_fixture_and_records_hashes(self) -> None:
        self.fill_complete_fixture()
        report = evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertEqual(report["state"], "SYNTHETIC_CONTRACT_TEST_PASSED")
        self.assertEqual(report["evaluation_type"], "synthetic_proxy")
        self.assertEqual(report["test_payloads_read"], 30)
        self.assertEqual(report["buffer_payloads_read"], 0)
        self.assertEqual(len(report["prediction_bundle_sha256"]), 27)
        self.assertEqual(len(report["independent_truth_sha256"]), 3)
        self.assertEqual(set(report["paired_rmse_differences_wm2"]), {
            "FAC_minus_FA", "FAP_minus_FA", "FAP_minus_FAC",
        })
        self.assertEqual(set(report["paired_date_cluster_bootstrap_by_comparison"]),
                         set(report["paired_rmse_differences_wm2"]))
        self.assertEqual(set(report["paired_two_day_block_bootstrap_by_comparison"]),
                         set(report["paired_rmse_differences_wm2"]))
        self.assertEqual(
            report["paired_two_day_block_bootstrap_by_comparison"]["FAC_minus_FA"]["block_length_days"],
            2,
        )
        self.assertAlmostEqual(report["paired_rmse_differences_wm2"]["FAC_minus_FA"], 0.0)
        self.assertEqual(report["three_station_common_support_sensitivity"]["n_keys_per_station"],
                         {station: len(self.expected_keys) for station in STATIONS})
        json.loads(self.output.read_text(encoding="utf-8"))

    def test_rejects_ground_truth_embedded_in_prediction_bundle(self) -> None:
        self.fill_complete_fixture()
        path = self.run_root / "seed42" / "Niushou" / "FA_test_predictions_with_keys.npz"
        np.savez_compressed(
            path,
            station=np.asarray(["Niushou"] * len(self.expected_keys)),
            truth_ghi_wm2=np.full(len(self.expected_keys), 300.0),
            pred_ghi_wm2=np.full(len(self.expected_keys), 280.0),
            clear_sky_ghi_wm2=np.full(len(self.expected_keys), 800.0),
            init_time_bjt=np.asarray([key[0] for key in self.expected_keys]),
            target_time_bjt=np.asarray([key[1] for key in self.expected_keys]),
            lead_minutes=np.asarray([key[2] for key in self.expected_keys], dtype=np.int16),
        )
        with self.assertRaisesRegex(ValueError, "must not contain ground truth"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())

    def test_truth_hash_is_checked_before_truth_payload_is_opened(self) -> None:
        manifest = json.loads(self.truth_manifest_path.read_text(encoding="utf-8"))
        manifest["stations"]["Niushou"]["sha256"] = "0" * 64
        self.truth_manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        self.receipt["test_pack_manifest_sha256"] = sha256_file(self.truth_manifest_path)
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with patch("evaluate_hubei_forecast_heads_frozen_test.np.load", side_effect=AssertionError("truth payload opened before hash validation")):
            with self.assertRaisesRegex(ValueError, "truth file hash mismatch"):
                evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.ledger_path.exists())

    def test_one_use_ledger_blocks_a_second_payload_read(self) -> None:
        self.fill_complete_fixture()
        first_report = evaluate(
            self.run_root, self.output, self.receipt_path, self.split_path,
            self.truth_manifest_path, 100,
        )
        self.assertTrue(self.ledger_path.is_file())
        self.assertEqual(first_report["one_use_ledger_sha256"], sha256_file(self.ledger_path))
        second_output = self.root / "scores_second_attempt.json"
        with patch("evaluate_hubei_forecast_heads_frozen_test.np.load", side_effect=AssertionError("payload reopened on retry")):
            with self.assertRaises(FileExistsError):
                evaluate(
                    self.run_root, second_output, self.receipt_path, self.split_path,
                    self.truth_manifest_path, 100,
                )
        self.assertFalse(second_output.exists())
        copied_split = self.root / "copied_split_elsewhere.csv"
        copied_split.write_bytes(self.split_path.read_bytes())
        copied_output = self.root / "scores_copied_split_attempt.json"
        with patch("evaluate_hubei_forecast_heads_frozen_test.np.load", side_effect=AssertionError("payload reopened on copied split")):
            with self.assertRaises(FileExistsError):
                evaluate(
                    self.run_root, copied_output, self.receipt_path, copied_split,
                    self.truth_manifest_path, 100,
                )
        self.assertFalse(copied_output.exists())

    def test_ledger_path_is_not_a_cli_option_or_output_override(self) -> None:
        import subprocess
        import sys

        script = Path(__file__).with_name("evaluate_hubei_forecast_heads_frozen_test.py")
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertNotIn("--ledger", help_result.stdout)
        self.assertEqual(CANONICAL_TEST_LEDGER_PATH.name, "hubei_frozen_test_once.ledger.json")
        self.assertNotEqual(self.ledger_path, self.split_path.with_name(
            self.split_path.name + ".hubei-test-once.ledger.json"))

    def test_rejects_prediction_keys_not_matching_independent_truth(self) -> None:
        self.fill_complete_fixture()
        self.write_truth_bundle("Niushou", shift_date="20260602")
        manifest = json.loads(self.truth_manifest_path.read_text(encoding="utf-8"))
        manifest["stations"]["Niushou"]["sha256"] = sha256_file(self.root / "truth_Niushou.npz")
        self.truth_manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        self.receipt["test_pack_manifest_sha256"] = sha256_file(self.truth_manifest_path)
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not exactly match independent truth"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())

    def test_reports_three_station_common_support_when_key_coverage_differs(self) -> None:
        self.write_truth_bundle("Mizhuang", shift_date="20260602")
        manifest = json.loads(self.truth_manifest_path.read_text(encoding="utf-8"))
        manifest["stations"]["Mizhuang"]["sha256"] = sha256_file(self.root / "truth_Mizhuang.npz")
        self.truth_manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        self.receipt["test_pack_manifest_sha256"] = sha256_file(self.truth_manifest_path)
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        for seed in SEEDS:
            for station in STATIONS:
                for arm in ARMS:
                    self.write_bundle(
                        seed, station, arm,
                        shift_date="20260602" if station == "Mizhuang" else None,
                    )
        shifted_mizhuang = [
            ("202606020815", (datetime.strptime("202606020815", "%Y%m%d%H%M") + timedelta(minutes=lead)).strftime("%Y%m%d%H%M"), lead)
            for lead in range(15, 241, 15)
        ]
        mizhuang_keys = [key for key in self.expected_keys if key[1].startswith("20260601")] + shifted_mizhuang
        self.receipt["expected_keys_sha256_by_station"]["Mizhuang"] = canonical_keys_sha256(mizhuang_keys)
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")

        report = evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        common = report["three_station_common_support_sensitivity"]
        self.assertEqual(common["n_keys_per_station"], {station: 16 for station in STATIONS})
        self.assertEqual(common["target_dates_bjt"], ["20260601"])
        self.assertIsNone(common["paired_date_cluster_bootstrap_by_comparison"])
        self.assertEqual(common["bootstrap_status"], "unavailable_fewer_than_two_common_target_dates")

    def test_rejects_validation_date_in_test_predictions(self) -> None:
        self.fill_complete_fixture()
        self.write_bundle(42, "Niushou", "FA", target_date="20260603")
        with self.assertRaisesRegex(ValueError, "outside frozen test dates"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)
        self.assertFalse(self.output.exists())

    def test_rejects_mismatched_arm_support(self) -> None:
        self.fill_complete_fixture()
        path = self.run_root / "seed43" / "Wangzhai" / "FAC_test_predictions_with_keys.npz"
        np.savez_compressed(path,
            station=np.asarray(["Wangzhai"]),
            pred_ghi_wm2=np.asarray([280.0]), clear_sky_ghi_wm2=np.asarray([800.0]),
            init_time_bjt=np.asarray(["202606010800"]), target_time_bjt=np.asarray(["202606010815"]),
            lead_minutes=np.asarray([15], dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "Prediction keys do not exactly match independent truth"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)

    def test_rejects_float_lead_without_lossless_integer_conversion(self) -> None:
        self.fill_complete_fixture()
        path = self.run_root / "seed42" / "Niushou" / "FA_test_predictions_with_keys.npz"
        np.savez_compressed(path,
            station=np.asarray(["Niushou", "Niushou"]),
            pred_ghi_wm2=np.asarray([280.0, 280.0]),
            clear_sky_ghi_wm2=np.asarray([800.0, 800.0]),
            init_time_bjt=np.asarray(["202606010800", "202606020800"]),
            target_time_bjt=np.asarray(["202606010815", "202606020815"]),
            lead_minutes=np.asarray([15.5, 15.0]))
        with self.assertRaisesRegex(ValueError, "Lossy non-integer lead"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)

    def test_rejects_broadcastable_column_vector(self) -> None:
        self.fill_complete_fixture()
        path = self.run_root / "seed42" / "Niushou" / "FA_test_predictions_with_keys.npz"
        np.savez_compressed(path,
            station=np.asarray(["Niushou", "Niushou"]),
            pred_ghi_wm2=np.asarray([[280.0], [280.0]]),
            clear_sky_ghi_wm2=np.asarray([800.0, 800.0]),
            init_time_bjt=np.asarray(["202606010800", "202606020800"]),
            target_time_bjt=np.asarray(["202606010815", "202606020815"]),
            lead_minutes=np.asarray([15, 15], dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "inconsistent prediction bundle"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)

    def test_rejects_subset_of_frozen_expected_keys(self) -> None:
        self.fill_complete_fixture()
        path = self.run_root / "seed42" / "Niushou" / "FA_test_predictions_with_keys.npz"
        np.savez_compressed(path,
            station=np.asarray(["Niushou"]),
            pred_ghi_wm2=np.asarray([280.0]), clear_sky_ghi_wm2=np.asarray([800.0]),
            init_time_bjt=np.asarray(["202606010800"]), target_time_bjt=np.asarray(["202606010815"]),
            lead_minutes=np.asarray([15], dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "Prediction keys do not exactly match independent truth"):
            evaluate(self.run_root, self.output, self.receipt_path, self.split_path, self.truth_manifest_path, 100)

    def test_accepts_frozen_sparse_valid_lead_support(self) -> None:
        self.fill_sparse_common_fixture()
        report = evaluate(self.run_root, self.output, self.receipt_path, self.split_path,
                          self.truth_manifest_path, 100)
        self.assertEqual(report["three_station_common_support_sensitivity"]["n_keys_per_station"],
                         {station: 6 for station in STATIONS})
        self.assertEqual(report["test_target_dates_bjt"], self.dates)
        self.assertEqual(report["test_payloads_read"], 30)


if __name__ == "__main__":
    unittest.main()
