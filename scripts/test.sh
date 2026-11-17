#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
case "${1:-core}" in
  core)
    "$python_bin" -m unittest discover -s tests -v
    PYTHONPATH=agent-service "$python_bin" -m pytest -p no:cacheprovider agent-service/tests -q
    PYTHONPATH=rag-service "$python_bin" -m unittest discover -s rag-service/tests -v
    "$python_bin" -m pytest -p no:cacheprovider traffic-gen/tests -q
    ;;
  kubeflow) PYTHONPATH=demo/kubeflow "$python_bin" -m pytest -p no:cacheprovider demo/kubeflow/tests -q ;;
  training) "$python_bin" -m unittest discover -s demo/training/tests -v ;;
  *) exit 2 ;;
esac
