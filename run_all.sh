#!/usr/bin/env bash
#
# run_all.sh -- reproduce every figure and every results CSV from a clean clone.
# Guide, Part 5, "Definition of done", item 8.
#
#   ./run_all.sh              # tests, then execute notebooks 01-08 and 10
#   ./run_all.sh --tests-only # just the pytest suite
#   ./run_all.sh --no-install # skip venv creation (use the current environment)
#
# Tiers 1 and 2 are pure NumPy and run on any laptop in a few minutes. Tier 3
# (notebook 09) needs a CUDA GPU and a checkpoint download, so it is NOT run
# here -- see the notice printed at the end. Its outputs are committed to
# results/ and figures/, and notebook 10 reads them from there.

set -euo pipefail
cd "$(dirname "$0")"

INSTALL=1
TESTS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --no-install) INSTALL=0 ;;
    --tests-only) TESTS_ONLY=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

hr() { printf '\n%s\n' "------------------------------------------------------------"; }

# --- 1. environment --------------------------------------------------------
if [ "$INSTALL" -eq 1 ]; then
  hr; echo "1. environment"
  if [ ! -d .venv ]; then
    # Homebrew python@3.14 is broken on macOS 26 (pyexpat/pip); uv sidesteps it.
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 .venv
    else
      python3 -m venv .venv
    fi
  fi
  # shellcheck disable=SC1091
  if [ -f .venv/bin/activate ]; then . .venv/bin/activate; else . .venv/Scripts/activate; fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install -q -r requirements.txt
  else
    python -m pip install -q -r requirements.txt
  fi
  echo "   $(python --version), $(python -c 'import numpy; print("numpy", numpy.__version__)')"
else
  hr; echo "1. environment  (skipped, --no-install)"
fi

# --- 2. tests --------------------------------------------------------------
hr; echo "2. test suite"
python -m pytest

if [ "$TESTS_ONLY" -eq 1 ]; then
  hr; echo "done (--tests-only)"; exit 0
fi

# --- 3. notebooks ----------------------------------------------------------
# In order: each milestone's notebook writes the results/ CSVs the next ones and
# the final report read. 09 is skipped (GPU); 10 consumes everything.
NOTEBOOKS=(
  01_schedule
  02_tier1_testbed
  03_solvers
  04_arm_c
  05_convergence
  06_stability
  07_tier2
  08_crossover
  10_report
)

hr; echo "3. notebooks"
for nb in "${NOTEBOOKS[@]}"; do
  echo "   executing notebooks/${nb}.ipynb"
  python -m jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    "notebooks/${nb}.ipynb"
done

# --- 4. Tier 3 notice ------------------------------------------------------
hr
echo "4. Tier 3 (notebook 09) -- NOT run here"
echo
echo "   notebooks/09_tier3_cifar10.ipynb needs a CUDA GPU and downloads the"
echo "   google/ddpm-cifar10-32 checkpoint. Run it on a Kaggle T4 (Internet ON,"
echo "   Accelerator: GPU T4 x1), then copy back:"
echo "       results/tier3_error.csv  results/tier3_decomposition.csv"
echo "       figures/09_tier3_error_vs_nfe.png  figures/09_tier3_decomposition.png"
echo "   and re-run notebooks/10_report.ipynb to fill in the Tier-3 rows."
if [ -f results/tier3_error.csv ]; then
  echo
  echo "   [present] results/tier3_error.csv -- the report above includes Tier 3."
else
  echo
  echo "   [pending] results/tier3_error.csv -- the report above marks Tier 3 pending."
fi

hr
echo "done. figures/ and results/ are regenerated."
