"""OpenTelemetry bootstrap and bounded custom instruments for KFP steps."""

from __future__ import annotations

import os
from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


tracer = trace.get_tracer("suse-ai.kubeflow-pipelines")
meter = metrics.get_meter("suse-ai.kubeflow-pipelines")

step_runs = meter.create_counter(
    "suse.ai.kubeflow.pipeline.step.runs",
    unit="{run}",
    description="Pipeline step executions by step and outcome.",
)
step_duration = meter.create_histogram(
    "suse.ai.kubeflow.pipeline.step.duration",
    unit="s",
    description="Pipeline step execution duration.",
)
model_accuracy = meter.create_histogram(
    "suse.ai.kubeflow.model.accuracy",
    unit="1",
    description="Observed evaluation accuracy for the demo model.",
)
smoke_tests = meter.create_counter(
    "suse.ai.kubeflow.deployment.smoke_test",
    unit="{test}",
    description="KServe deployment smoke-test outcomes.",
)


@dataclass
class TelemetryProviders:
    tracer_provider: TracerProvider
    meter_provider: MeterProvider

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def init_telemetry() -> TelemetryProviders:
    """Configure trace and metric export using standard OTEL environment vars."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
    resource = Resource.create(
        {
            "service.name": os.environ.get(
                "OTEL_SERVICE_NAME",
                "kubeflow-pipelines-demo",
            ),
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    metric_readers = []
    if endpoint:
        insecure = endpoint.startswith("http://")
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
        )
        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint, insecure=insecure),
                export_interval_millis=int(
                    os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "10000")
                ),
            )
        )

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    HTTPXClientInstrumentor().instrument()
    return TelemetryProviders(tracer_provider, meter_provider)


def init_tracer():
    """Backward-compatible helper for existing local callers."""
    init_telemetry()
    return trace.get_tracer("suse-ai.kubeflow-pipelines")
