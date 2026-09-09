#!/usr/bin/env sh
# claim6_probe_evasion -- see claim.txt for what this shows, expected/ for reference values.
#
#   ./run.sh --smoke   minutes. Pipeline check only; does NOT evaluate the claim.
#   ./run.sh --quick   under an hour. Evaluates the claim at a reduced budget.
#   ./run.sh --full    the paper configuration. Hours.
#
# Extra flags pass through, e.g. --dataset-root /path/to/datasets
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -d "$ROOT/.venv" ] && . "$ROOT/.venv/bin/activate" 2>/dev/null || true
exec python "$ROOT/artifact/scripts/claim_runner.py" --claim claim6_probe_evasion "$@"
