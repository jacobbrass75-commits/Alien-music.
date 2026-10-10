#!/usr/bin/env bash
# The first experiment: one small model per environment, trained in parallel.
# Results land in runs/<env>/first/. Works on Windows (Git Bash), Linux and macOS.
# Pass --wait to hold off until the PC is idle (Windows only; uses scripts/wait_until_idle.ps1).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python || command -v python3)
if [ "${1:-}" = "--wait" ]; then
  if command -v powershell >/dev/null; then
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/wait_until_idle.ps1
  else
    echo "--wait is Windows-only; starting now."
  fi
fi

CORES=$("$PY" -c "import os; print(os.cpu_count() or 1)")
export OMP_NUM_THREADS=$(( CORES / 3 > 0 ? CORES / 3 : 1 ))
echo "$(date +%H:%M) starting first experiment ($OMP_NUM_THREADS threads per run)"
"$PY" train.py signal   --iters 800  --name first > runs_first_signal.log   2>&1 &
"$PY" train.py rhythm   --iters 2000 --name first > runs_first_rhythm.log   2>&1 &
"$PY" train.py composer --iters 2000 --name first > runs_first_composer.log 2>&1 &
wait
mkdir -p runs && mv runs_first_*.log runs/ 2>/dev/null
echo "$(date +%H:%M) done"
"$PY" scripts/summarize.py runs --last 100
