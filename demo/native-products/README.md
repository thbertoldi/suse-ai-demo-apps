# Instrumenting the MLflow workers

The demo chart sends real MLflow API traffic and trains a model whose checkpoint
is stored through MLflow. It does not install or replace the MLflow server.
Server-side request telemetry requires instrumentation in each Python worker,
in addition to the Collector's availability probes.

`mlflow_sitecustomize.py` initializes OpenTelemetry in the launcher and child
interpreters. Package it with the pinned dependencies in `requirements.txt`.
For a Python 3.12 MLflow image, build a derived image using:

```dockerfile
ARG MLFLOW_IMAGE
FROM python:3.12-slim AS telemetry
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --target /otel -r /tmp/requirements.txt
COPY mlflow_sitecustomize.py /otel/sitecustomize.py

FROM ${MLFLOW_IMAGE}
COPY --from=telemetry /otel /otel
ENV PYTHONPATH=/otel
```

Save this Dockerfile next to the two files and build with your pinned MLflow
image as `--build-arg MLFLOW_IMAGE=...`. The Python minor version in the telemetry
stage must match the MLflow workers. Preserve any application-specific
`PYTHONPATH` entries after `/otel` if the base image needs them. This recipe
preserves the base image's entrypoint and existing MLflow configuration.

Configure the server workload with `OTEL_SERVICE_NAME=mlflow`, its actual
`service.namespace`, pod UID/node resource attributes, and the Collector's
OTLP endpoint. Set `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` and
`OTEL_EXPORTER_OTLP_INSECURE=true` only for a plaintext in-cluster receiver.
Use the supported `suse.ai.component.type=ml-registry.mlflow` resource attribute
for product classification. Do not use a static service-instance identity across
workers/pods or disable a verified TLS connection.

MLflow must allow the exact Service host/port used by clients through
`MLFLOW_SERVER_ALLOWED_HOSTS`. It must serve artifacts and return
`mlflow-artifacts:/` artifact URIs for training; a server-local `file:` artifact
store cannot be uploaded through this demo's HTTP artifact client.

After rollout, exercise an actual run API and check for a server span from the
worker. A healthy `/health` response alone does not demonstrate worker telemetry.
Use the chart's optional Secret-backed MLflow bearer token settings when the
server requires authentication.
