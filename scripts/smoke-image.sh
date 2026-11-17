#!/usr/bin/env bash
set -euo pipefail
service=${1:?service is required}
image=${2:?image is required}
root_dir=$(cd "$(dirname "$0")/.." && pwd)
case "$service" in
  gateway)
    # The gateway must start even while its upstream services are unavailable.
    docker run --rm --network none --entrypoint /bin/sh "$image" -c \
      '/gateway & pid=$!; sleep 2; kill -0 "$pid"; kill -TERM "$pid"; wait "$pid"'
    ;;
  rag-service)
    docker run --rm --network none "$image" python -c 'import app.main, app.product_traffic, app.vectordb.factory'
    docker run --rm --network none -v "$root_dir/rag-service/tests:/app/tests:ro" \
      "$image" python -m unittest discover -s tests -v
    ;;
  llm-service|agent-service) docker run --rm --network none "$image" python -c 'import app.main' ;;
  traffic-gen) docker run --rm --network none "$image" python -c 'import main, webui' ;;
  iris-pipeline) docker run --rm --network none "$image" python -m steps --help ;;
  training)
    docker run --rm --network none "$image" python -c 'import train, torch; assert torch.version.cuda is None'
    docker run --rm --network none -v "$root_dir/demo/training/tests:/app/tests:ro" \
      "$image" python -m unittest discover -s tests -v
    ;;
  *) exit 2 ;;
esac
