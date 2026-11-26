#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints
# GitHub CLI must be installed and able to read this repository's Releases.
gh release download baseline-v1 --repo BalaneML/AdvancedAI --pattern 'D1-baseline-best.pkl' --pattern 'D2-nolcf-best.pkl' --dir checkpoints --skip-existing
.venv/bin/python tools/verify_checkpoints.py
