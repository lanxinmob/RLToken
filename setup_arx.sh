#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RLT_UV="${RLT_UV:-../pi05/.tools/bin/uv}"
[[ -x "$RLT_UV" ]] || RLT_UV=uv
if [[ ! -x .venv/bin/python ]]; then
  "$RLT_UV" venv --python "${RLT_PYTHON:-python}" .venv
fi
"$RLT_UV" pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 -r arx-lock.txt
"$RLT_UV" pip check --python .venv/bin/python
