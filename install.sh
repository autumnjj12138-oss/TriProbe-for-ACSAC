#!/usr/bin/env bash
# One-click setup for the TriProbe artifact.
#
# Creates a virtual environment, installs pinned dependencies, and checks that
# the datasets are present. It does NOT download the datasets: all three require
# accepting a provider licence, so the URLs and the expected layout are printed
# instead. See infrastructure/datasets.md.
#
# Usage:  bash install.sh            # CUDA build of torch (recommended)
#         CPU_ONLY=1 bash install.sh # CPU build, ~20x slower
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "ERROR: $PY not found. Set PYTHON=/path/to/python3.11"; exit 1; }

VER="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
case "$VER" in
  3.10|3.11|3.12) ;;
  *) echo "WARNING: tested on Python 3.11 (found $VER). Continuing anyway." ;;
esac

echo "[1/4] creating virtualenv in .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
python -m pip install --quiet --upgrade pip

echo "[2/4] installing torch"
if [ "${CPU_ONLY:-0}" = "1" ]; then
  pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
else
  # cu126 matches the reference environment. Change the index URL if your
  # driver needs a different CUDA build; the code itself is version-agnostic.
  pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
fi

echo "[3/4] installing remaining dependencies"
pip install --quiet -r requirements.txt

echo "[4/4] verifying"
python - <<'PYEOF'
import sys
import numpy, pandas, sklearn, torch
print(f"  python       {sys.version.split()[0]}")
print(f"  torch        {torch.__version__}  cuda={torch.version.cuda}")
print(f"  cuda avail   {torch.cuda.is_available()}", end="")
print(f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "  (CPU only: expect ~20x slower)")
print(f"  numpy        {numpy.__version__}")
print(f"  pandas       {pandas.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
try:
    from sklearn.cluster import HDBSCAN  # noqa: F401
    print("  HDBSCAN      available (DeepSight baseline uses it)")
except ImportError:
    print("  HDBSCAN      MISSING -> DeepSight falls back to AgglomerativeClustering")
PYEOF

echo
echo "Checking datasets (not downloaded by this script):"
MISSING=0
for d in cicids2017 unsw_nb15 nsl_kdd; do
  if [ -d "artifact/dataset/$d" ] && [ -n "$(ls -A "artifact/dataset/$d" 2>/dev/null)" ]; then
    echo "  OK      artifact/dataset/$d"
  else
    echo "  MISSING artifact/dataset/$d"
    MISSING=1
  fi
done
if [ "$MISSING" = "1" ]; then
  echo
  echo "One or more datasets are absent. See infrastructure/datasets.md for"
  echo "download URLs, the expected directory layout, and checksums."
  echo "claims/claim1..6 cannot run until they are in place."
fi

echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Then try the fastest check:     bash claims/claim3_density_cliff/run.sh --smoke"
