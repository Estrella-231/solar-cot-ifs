#!/usr/bin/env python3
"""Synthetic tests for the Hubei one-use ledger primitive."""
from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hubei_test_ledger import REQUIRED_BINDINGS, claim_one_use


class OneUseLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ledger = self.root / "test_once.json"
        self.bindings = {field: (str(index + 1) * 64) for index, field in enumerate(REQUIRED_BINDINGS)}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_claim_records_all_frozen_bindings(self) -> None:
        record = claim_one_use(self.ledger, self.bindings)
        saved = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertEqual(saved, record)
        self.assertEqual(record["state"], "CLAIMED_BEFORE_TEST_PAYLOAD_READ")
        self.assertEqual(record["bindings"], self.bindings)

    def test_second_claim_is_rejected(self) -> None:
        claim_one_use(self.ledger, self.bindings)
        with self.assertRaises(FileExistsError):
            claim_one_use(self.ledger, self.bindings)

    def test_invalid_binding_does_not_consume_ledger(self) -> None:
        bad = dict(self.bindings, split_sha256="not-a-hash")
        with self.assertRaisesRegex(ValueError, "invalid SHA-256"):
            claim_one_use(self.ledger, bad)
        self.assertFalse(self.ledger.exists())

    def test_concurrent_claim_has_exactly_one_winner(self) -> None:
        def attempt() -> str:
            try:
                claim_one_use(self.ledger, self.bindings)
                return "claimed"
            except FileExistsError:
                return "already_claimed"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: attempt(), range(8)))
        self.assertEqual(results.count("claimed"), 1)
        self.assertEqual(results.count("already_claimed"), 7)


if __name__ == "__main__":
    unittest.main()
