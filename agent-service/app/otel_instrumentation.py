import os
import uuid
from contextlib import contextmanager

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
)

def _capture_content() -> bool:
    # Standard OTel GenAI opt-in; fall back to the legacy var for backward compat.
    val = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    if val is None:
        val = os.environ.get("ENABLE_OTEL_CONTENT_EVENTS", "false")
    return val.strip().lower() in ("true", "1", "yes")


ENABLE_CONTENT_EVENTS = _capture_content()


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


def record_tool_result(span, arguments: str, result: str):
    if ENABLE_CONTENT_EVENTS:
        span.set_attribute("gen_ai.tool.call.arguments", arguments)
    span.set_attribute("gen_ai.tool.call.result", result)
