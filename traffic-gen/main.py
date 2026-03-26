import os
import random
import signal
import logging
import time

import grpc
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

from generated import demo_pb2, demo_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAG_QUERIES = [
    "What is a Kubernetes pod?",
    "How do Linux containers work?",
    "What is a container runtime?",
    "Explain Kubernetes deployments",
    "What are container images?",
    "What is OpenTelemetry?",
    "Explain distributed tracing",
    "What is a Kubernetes service?",
]

CHAT_MESSAGES = [
    "Explain microservices in one sentence",
    "What is observability?",
    "What is the difference between monitoring and observability?",
    "Explain the CAP theorem briefly",
    "What is a service mesh?",
    "What are the benefits of containerization?",
]

AGENT_MESSAGES = [
    "Search our docs for what a Kubernetes pod is, then tell me the current time",
    "Calculate 256 * 384 and search for information about container runtimes",
    "What time is it right now?",
    "Search the docs about distributed tracing and also look up what OpenTelemetry is on the web",
    "Calculate 1024 / 16 and tell me the current time",
    "Search our knowledge base for information about container images",
]

running = True


def shutdown(signum, frame):
    global running
    logger.info("Received shutdown signal")
    running = False


def main():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "traffic-gen")
    resource = Resource.create({"service.name": service_name})

    trace_exporter = OTLPSpanExporter(insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    set_global_textmap(CompositeHTTPPropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    GrpcInstrumentorClient().instrument()

    gateway_addr = os.environ.get("GATEWAY_ADDR", "gateway:50051")
    interval = int(os.environ.get("INTERVAL_SECONDS", "5"))

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    channel = grpc.insecure_channel(gateway_addr)
    stub = demo_pb2_grpc.DemoServiceStub(channel)

    query_types = ["rag", "chat", "agent"]
    query_lists = {
        "rag": RAG_QUERIES,
        "chat": CHAT_MESSAGES,
        "agent": AGENT_MESSAGES,
    }
    query_indices = {"rag": 0, "chat": 0, "agent": 0}

    type_idx = 0
    logger.info(f"Starting traffic generation to {gateway_addr} every ~{interval}s")

    while running:
        query_type = query_types[type_idx % len(query_types)]
        type_idx += 1
        query_list = query_lists[query_type]
        query_text = query_list[query_indices[query_type] % len(query_list)]
        query_indices[query_type] += 1

        try:
            if query_type == "rag":
                logger.info(f"Sending RAG query: {query_text}")
                resp = stub.Query(demo_pb2.QueryRequest(query=query_text, top_k=3), timeout=120)
                logger.info(f"RAG response model={resp.model}, sources={len(resp.sources)}")
            elif query_type == "chat":
                logger.info(f"Sending Chat message: {query_text}")
                resp = stub.Chat(demo_pb2.ChatRequest(message=query_text), timeout=120)
                logger.info(f"Chat response model={resp.model}")
            elif query_type == "agent":
                logger.info(f"Sending Agent message: {query_text}")
                resp = stub.AgentChat(demo_pb2.AgentChatRequest(message=query_text), timeout=180)
                logger.info(f"Agent response model={resp.model}, tools_used={len(resp.tool_calls_made)}")
        except grpc.RpcError as e:
            logger.warning(f"gRPC error: {e.code()} {e.details()}")
        except Exception as e:
            logger.warning(f"Error: {e}")

        jitter = random.uniform(0, 2)
        sleep_time = interval + jitter
        start = time.monotonic()
        while running and (time.monotonic() - start) < sleep_time:
            time.sleep(0.5)

    logger.info("Shutting down OTel...")
    channel.close()
    tracer_provider.shutdown()
    meter_provider.shutdown()
    logger.info("Done")


if __name__ == "__main__":
    main()
