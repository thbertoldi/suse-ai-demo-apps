"""Initialize OTel in both the MLflow launcher and its Python HTTP workers.

Mount as /otel/sitecustomize.py alongside the SDK and put /otel on PYTHONPATH.
The normal opentelemetry-instrument bootstrap removes its own directory from
PYTHONPATH, so child interpreters otherwise start without instrumentation.
"""
from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()
