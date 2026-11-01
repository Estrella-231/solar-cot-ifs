#!/usr/bin/env python3
"""Validation-only SimVP->R replay for the completed MSE/temporal/spatial arms.

This is a controlled continuation of the existing R07 plan. It uses the same
full monthly validation manifest, frozen monthly R, CPP reference mapping,
geometry, batch size, and output schema as the baseline all-validation replay.
"""
import argparse
import sys
from pathlib import Path

from profile_forecast_agri3x3_allval_20261014 import EXPECTED, main as _unused


def main():
    # Reuse the frozen evaluator implementation while allowing only the
    # explicitly hashed candidate SimVP checkpoint to differ.
    original = sys.argv[:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expected-simvp-sha", required=True)
    known, remaining = parser.parse_known_args()
    candidate_sha = known.expected_simvp_sha.lower()
    if len(candidate_sha) != 64 or any(c not in "0123456789abcdef" for c in candidate_sha):
        raise ValueError("--expected-simvp-sha must be a lowercase SHA-256 digest")
    EXPECTED["simvp"] = candidate_sha
    sys.argv = [original[0], *remaining]
    try:
        from profile_forecast_agri3x3_allval_20261014 import main as run
        run()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
