import os
import time
import uuid
from contextlib import contextmanager

from opentelemetry import trace, metrics

tracer = trace.get_tracer("gen_ai")
meter = metrics.get_meter("gen_ai")

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

tool_calls_counter = meter.create_counter(
    name="suse.ai.agent.tool.calls",
    description="Agent tool calls by tool and outcome",
    unit="{call}",
)

tool_duration_histogram = meter.create_histogram(
    name="suse.ai.agent.tool.duration",
    description="Agent tool execution duration",
    unit="s",
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

ENABLE_CONTENT_EVENTS = os.environ.get("ENABLE_OTEL_CONTENT_EVENTS", "false").lower() == "true"


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
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
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
