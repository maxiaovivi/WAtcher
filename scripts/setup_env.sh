#!/usr/bin/env bash
# Create .venv on top of an existing RoboTwin Python environment (torch, SAPIEN 3.0.0b1, CuRobo, mplib)
# and install WAtcher into it. The base environment is not modified.
#   scripts/setup_env.sh /path/to/robotwin/env
set -euo pipefail
BASE=${1:?usage: scripts/setup_env.sh /path/to/robotwin/env}
cd "$(dirname "$0")/.."
"$BASE/bin/python" -m venv --system-site-packages .venv
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${TMPDIR:-/tmp}/pip-cache} .venv/bin/pip install -q -e '.[wam,text,dev]'
.venv/bin/python -c "import torch, sapien, watcher; print('torch', torch.__version__, '| sapien', sapien.__version__, '| watcher ok')"
