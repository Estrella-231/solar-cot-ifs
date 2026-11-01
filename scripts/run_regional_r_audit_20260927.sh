#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
EXP=$ROOT/experiments/regional_station_cot_20260925
exec /public/home/slfu/miniconda3/envs/swc/bin/python -u "$EXP/scripts/audit_regional_r_pair_20260927.py" \
 --cache "$EXP/agri_cpp_pair_bank_v1" \
 --baseline "$EXP/r256_full_exploratory_seed42_node22" \
 --candidate "$EXP/r256_full_exploratory_seed42_batch512_node22" \
 --output "$EXP/r_pair_audit_20260927_v1"
