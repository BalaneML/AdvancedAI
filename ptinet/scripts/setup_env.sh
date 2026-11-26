#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?Usage: bash scripts/setup_env.sh cuda|rocm-wsl}
python3.12 -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip
if [[ "$mode" == cuda ]]; then
  "$PY" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
elif [[ "$mode" == rocm-wsl ]]; then
  for name in 'torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' 'torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' 'triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'; do
    mkdir -p downloads/wheels
    wget -c -O "downloads/wheels/$name" "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/${name//+/%2B}"
  done
  "$PY" -m pip install downloads/wheels/*.whl
  export HSA_ENABLE_DXG_DETECTION=1
  "$PY" - <<'PY'
import sys
from pathlib import Path
lib=Path(sys.prefix)/'lib/python3.12/site-packages/torch/lib'
for p in lib.glob('libhsa-runtime64.so*'):
    if p.is_file() and not p.name.endswith('.bundled_backup'):
        backup=p.with_name(p.name+'.bundled_backup')
        if backup.exists():raise RuntimeError('Existing runtime backup; inspect before modifying')
        p.rename(backup)
PY
else
  echo 'Choose cuda or rocm-wsl'; exit 2
fi
"$PY" -m pip install -r requirements.txt
HSA_ENABLE_DXG_DETECTION=1 "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'
