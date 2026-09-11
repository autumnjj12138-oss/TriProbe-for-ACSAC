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
  # torch 2.6.0 ships wheels for cp39 through cp313, so all of these work.
  # The reference measurements were taken on 3.11.
  3.10|3.11|3.12|3.13) ;;
  *) echo "WARNING: tested on Python 3.10-3.13 (found $VER). Continuing anyway." ;;
esac

activate_venv() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
}

echo "[1/4] creating virtualenv in .venv"
if [ "${NO_VENV:-0}" = "1" ]; then
  echo "  NO_VENV=1 set: installing into the current environment instead"
elif "$PY" -m venv .venv 2>/tmp/triprobe_venv.err; then
  activate_venv
  python -m pip install --quiet --upgrade pip
else
  # Debian, Ubuntu and Google Colab ship venv without ensurepip, because
  # python3-venv is packaged separately. The venv directory itself is fine, so
  # rebuild it without pip and bootstrap pip into it.
  echo "  venv creation failed:"
  sed 's/^/    /' /tmp/triprobe_venv.err
  echo "  retrying with --without-pip and bootstrapping pip"
  rm -rf .venv
  if "$PY" -m venv --without-pip .venv 2>/dev/null; then
    activate_venv
    if ! curl -sS https://bootstrap.pypa.io/get-pip.py | python -; then
      echo
      echo "ERROR: could not bootstrap pip into the virtualenv."
      echo "Pick one:"
      echo "  1. install the venv package, then re-run:"
      echo "       sudo apt-get install -y python3-venv   # Debian, Ubuntu, Colab"
      echo "  2. skip the virtualenv and install into the current environment:"
      echo "       NO_VENV=1 bash install.sh"
      echo "     Use this only in a disposable environment such as a Colab"
      echo "     runtime or a container, since it changes the system packages."
      exit 1
    fi
  else
    echo
    echo "ERROR: this Python cannot create a virtualenv."
    echo "Pick one:"
    echo "  1. install the venv package, then re-run:"
    echo "       sudo apt-get install -y python3-venv   # Debian, Ubuntu, Colab"
    echo "  2. skip the virtualenv and install into the current environment:"
    echo "       NO_VENV=1 bash install.sh"
    echo "     Use this only in a disposable environment such as a Colab"
    echo "     runtime or a container, since it changes the system packages."
    exit 1
  fi
fi

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
echo
if [ -d .venv ]; then
  echo "Setup complete. Activate with:  source .venv/bin/activate"
else
  echo "Setup complete, installed into the current environment (NO_VENV=1)."
fi
echo "Fastest pipeline check:         bash claims/claim3_density_cliff/run.sh --smoke"
echo "Recommended evaluation path:    bash claims/run_all.sh --quick"
