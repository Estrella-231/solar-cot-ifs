#!/usr/bin/env python3
"""Fail-closed, atomic one-use claim for an eventual frozen Hubei test run."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


REQUIRED_BINDINGS = (
    "release_receipt_sha256",
    "split_sha256",
    "truth_manifest_sha256",
    "model_manifest_sha256",
    "evaluator_sha256",
    "test_target_dates_sha256",
)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def claim_one_use(ledger_path: Path, bindings: dict[str, str]) -> dict[str, Any]:
    """Atomically consume the ledger path before any test payload file is opened.

    The existence of the file means the evaluation slot is consumed, including
    when the caller crashes immediately after the claim. Never auto-delete or
    repair a partial ledger: that is the intentional fail-closed behavior.
    """
    missing = [field for field in REQUIRED_BINDINGS if field not in bindings]
    if missing:
        raise ValueError(f"One-use ledger bindings missing fields: {missing}")
    invalid = [field for field in REQUIRED_BINDINGS if not _valid_sha256(bindings[field])]
    if invalid:
        raise ValueError(f"One-use ledger bindings contain invalid SHA-256: {invalid}")
    ledger_path = Path(ledger_path)
    if not ledger_path.parent.is_dir():
        raise FileNotFoundError(f"One-use ledger parent must already exist: {ledger_path.parent}")

    payload = {
        "state": "CLAIMED_BEFORE_TEST_PAYLOAD_READ",
        "claim_id": str(uuid.uuid4()),
        "claimed_unix_time_ns": time.time_ns(),
        "bindings": {field: bindings[field].lower() for field in REQUIRED_BINDINGS},
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(ledger_path, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    # Best-effort durability of the directory entry on POSIX; file creation is
    # still protected by O_EXCL on platforms without O_DIRECTORY.
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is not None:
        directory_fd = os.open(ledger_path.parent, os.O_RDONLY | directory_flag)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return payload


def ledger_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
