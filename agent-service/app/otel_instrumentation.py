import os
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

from opentelemetry import trace, metrics

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

token_usage_histogram = meter.create_histogram(
    name="gen_ai.client.token.usage",
    description="Token usage per GenAI call",
    unit="{token}",
)

operation_duration_histogram = meter.create_histogram(
    name="gen_ai.client.operation.duration",
    description="Duration of GenAI operations",
    unit="s",
    explicit_bucket_boundaries_advisory=(
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
    ),
)

tool_calls_counter = meter.create_counter(
    name="suse.ai.agent.tool.calls",
    description="Agent tool calls by tool and outcome",
    unit="{call}",
)

tool_duration_histogram = meter.create_histogram(
    name="suse.ai.agent.tool.duration",
    description="Agent tool execution duration",
    unit="s",
    explicit_bucket_boundaries_advisory=(
        0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30,
    ),
)

agent_iterations_histogram = meter.create_histogram(
    name="suse.ai.agent.iterations",
    description="LLM iterations used per agent invocation",
    unit="1",
)

agent_runs_counter = meter.create_counter(
    name="suse.ai.agent.runs",
    description="Agent invocations by outcome and truncation",
    unit="{run}",
)

def _capture_content() -> bool:
    # Standard OTel GenAI opt-in; fall back to the legacy var for backward compat.
    val = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    if val is None:
        val = os.environ.get("ENABLE_OTEL_CONTENT_EVENTS", "false")
    return val.strip().lower() in ("true", "1", "yes")


ENABLE_CONTENT_EVENTS = _capture_content()


def model_endpoint_attributes():
    endpoint = urlsplit(os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1"))
    attributes = {"server.address": endpoint.hostname or ""}
    port = endpoint.port or {"http": 80, "https": 443}.get(endpoint.scheme)
    if port is not None:
        attributes["server.port"] = port
    return attributes


@contextmanager
def invoke_agent_span(agent_name: str, model: str):
    agent_id = str(uuid.uuid4())
    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.id": agent_id,
            "gen_ai.request.model": model,
            "gen_ai.provider.name": os.environ.get("LLM_PROVIDER", "vllm"),
            **model_endpoint_attributes(),
        },
    ) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise


@contextmanager
def execute_tool_span(tool_name: str, tool_call_id: str, tool_description: str = ""):
    started = time.monotonic()
    with tracer.start_as_current_span(
        f"execute_tool {tool_name}",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": tool_call_id,
            "gen_ai.tool.description": tool_description,
        },
    ) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise
        finally:
            outcome = (
                "error"
                if span.status.status_code == trace.StatusCode.ERROR
                else "success"
            )
            attributes = {"tool_name": tool_name, "outcome": outcome}
            tool_calls_counter.add(1, attributes)
            tool_duration_histogram.record(
                time.monotonic() - started,
                attributes,
            )


def record_agent_execution(iterations: int, truncated: bool, outcome: str) -> None:
    attributes = {
        "outcome": outcome,
        "truncated": str(truncated).lower(),
    }
    agent_iterations_histogram.record(iterations, attributes)
    agent_runs_counter.add(1, attributes)


def record_tool_result(span, arguments: str, result: str):
    if ENABLE_CONTENT_EVENTS:
        span.set_attribute("gen_ai.tool.call.arguments", arguments)
        span.set_attribute("gen_ai.tool.call.result", result)
