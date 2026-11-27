#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints
for name in D1-baseline-best.pkl D2-nolcf-best.pkl; do
  wget -c --tries=3 --timeout=60 -O "checkpoints/$name" "https://github.com/BalaneML/AdvancedAI/releases/download/baseline-v1/$name"
done
.venv/bin/python tools/verify_checkpoints.py
