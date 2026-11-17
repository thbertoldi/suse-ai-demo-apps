#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=${PYTHON:-python3}
case "${1:-core}" in
  core)
    "$python_bin" -m pip install -r tests/requirements.txt -r agent-service/requirements-dev.txt \
      -r rag-service/requirements.txt -r traffic-gen/requirements.txt
    ;;
  kubeflow) "$python_bin" -m pip install -r demo/kubeflow/requirements-dev.txt ;;
  training)
    "$python_bin" -m pip install -r demo/training/requirements-cpu.txt
    "$python_bin" -m pip install -r demo/training/requirements.txt
    ;;
  *) exit 2 ;;
esac
