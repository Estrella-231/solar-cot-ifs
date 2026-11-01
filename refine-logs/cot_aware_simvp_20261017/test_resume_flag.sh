#!/usr/bin/env bash
set -euo pipefail

RESUME_FLAG=""
set -- --batch-sequences 8 --r-batch-patches 1024 ${RESUME_FLAG:+$RESUME_FLAG}
[[ "$#" -eq 4 ]]

RESUME_FLAG=--resume
set -- --batch-sequences 8 --r-batch-patches 1024 ${RESUME_FLAG:+$RESUME_FLAG}
[[ "$#" -eq 5 && "$5" == "--resume" ]]

echo "FUTURE_BANK_RESUME_FLAG_WITNESS_PASS"
