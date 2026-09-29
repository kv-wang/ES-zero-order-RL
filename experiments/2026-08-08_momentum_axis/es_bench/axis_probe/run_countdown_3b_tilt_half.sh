#!/usr/bin/env bash
# Countdown 3B: piecewise tilt then vanilla ES (SINGLE GPU).
# First TILT_SWITCH_FRAC of steps use tilt with TILT_KAPPA (default 0.2);
# remaining steps use κ=0 (vanilla ES). Beta fixed at 0.9.
#
# Wrapper around run_countdown_3b.sh with TILT_HALF_VANILLA=1.
#
# Usage:  DRY_RUN=1 ./run_countdown_3b_tilt_half.sh
#         SMOKE=1   ./run_countdown_3b_tilt_half.sh
#         ./run_countdown_3b_tilt_half.sh
#         TILT_KAPPA=0.2 TILT_SWITCH_FRAC=0.5 ./run_countdown_3b_tilt_half.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

export TILT_HALF_VANILLA="${TILT_HALF_VANILLA:-1}"
export TILT_LAMBDA_COSINE="${TILT_LAMBDA_COSINE:-0}"
export TILT_KAPPA="${TILT_KAPPA:-0.2}"
export TILT_SWITCH_FRAC="${TILT_SWITCH_FRAC:-0.5}"
export TILT_MOM_BETA="${TILT_MOM_BETA:-0.9}"

exec bash "$HERE/run_countdown_3b.sh" "$@"
