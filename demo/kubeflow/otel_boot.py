"""Minimal OTLP tracer bootstrap for pipeline steps.

Resource attributes (incl. suse.ai.component.name=kubeflow-pipelines) come from
the OTEL_RESOURCE_ATTRIBUTES env var set on the pipeline task.
"""
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor


def init_tracer():
    provider = TracerProvider()
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    return trace.get_tracer("kubeflow-pipelines")
