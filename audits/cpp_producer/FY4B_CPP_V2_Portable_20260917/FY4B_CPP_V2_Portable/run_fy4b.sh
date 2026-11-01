#!/usr/bin/env bash
set -euo pipefail

# Portable FY4B launcher. The first argument selects the FY4B sub-satellite
# point; all remaining arguments are passed unchanged to Run_Pipeline.py.
# Examples:
#   ./run_fy4b.sh 133E --time 2023-07-01T00:00:00
#   ./run_fy4b.sh 105E --start 2025-08-01T00:00:00 --end 2025-08-01T23:45:00 --freq 15min

PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PACKAGE_ROOT"

if [[ $# -lt 1 ]]; then
    echo "用法: $0 {105E|133E} [Run_Pipeline.py 参数...]" >&2
    exit 2
fi

case "${1^^}" in
    105E) SATELLITE="FY4B_105E" ;;
    133E) SATELLITE="FY4B_133E" ;;
    *) echo "错误：星下点必须是 105E 或 133E，收到：$1" >&2; exit 2 ;;
esac
shift

if [[ -f config/runtime.env ]]; then
    set -a
    # shellcheck disable=SC1091
    source config/runtime.env
    set +a
fi

export GLOBAL_CPP_RESOURCE_ROOT="${GLOBAL_CPP_RESOURCE_ROOT:-$PACKAGE_ROOT}"
export GLOBAL_CPP_TEST_ROOT="${GLOBAL_CPP_TEST_ROOT:-$PACKAGE_ROOT/Test}"
export GLOBAL_CPP_RESULT_ROOT="${GLOBAL_CPP_RESULT_ROOT:-$PACKAGE_ROOT/Test/Result}"
export GLOBAL_CPP_GLOBAL_RESULTS="${GLOBAL_CPP_GLOBAL_RESULTS:-$PACKAGE_ROOT/Test/Results}"
export GLOBAL_CPP_CHECK_DIR="${GLOBAL_CPP_CHECK_DIR:-$PACKAGE_ROOT/Test/Check}"
export GLOBAL_CPP_LOG_DIR="${GLOBAL_CPP_LOG_DIR:-$PACKAGE_ROOT/Logs}"

exec python Code/Run_Pipeline.py satellite --satellite "$SATELLITE" "$@"
