#!/usr/bin/env bash
# ============================================================
# Trackey installer
#
# Usage:
#   ./scripts/install.sh           # installs [all] extras
#   ./scripts/install.sh dev       # also installs dev deps
#   ./scripts/install.sh tracking  # specific extra only
# ============================================================

set -euo pipefail

EXTRA="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "=== Trackey Install (extra: [$EXTRA]) ==="
echo ""

# Step 1: Install main package + standard extras (torchreid is NOT in pyproject.toml)
echo ">> Step 1: Installing trackey[$EXTRA]..."
pip install -e "${REPO_ROOT}[${EXTRA}]"

# Step 2: Install torchreid with --no-build-isolation
# Its old setup.py imports numpy at build time, which only works if numpy is
# already in the environment (which it is after step 1).
if [[ "$EXTRA" == "all" || "$EXTRA" == "reid" ]]; then
    echo ""
    echo ">> Step 2: Installing torchreid (--no-build-isolation required)..."
    pip install --no-build-isolation \
        "git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
fi

echo ""
echo "=== Installation complete! ==="
echo "Run with: trackey run <your_config.yaml>"
