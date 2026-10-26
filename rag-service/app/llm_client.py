import json
import os
import time
import requests
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
    explicit_bucket_boundaries_advisory=(
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120,
    ),
)

ENABLE_CONTENT_EVENTS = os.environ.get("ENABLE_OTEL_CONTENT_EVENTS", "false").lower() == "true"


def chat_completion(
    base_url: str,
    model: str,
    provider: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    with tracer.start_as_current_span(
        f"chat {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
            "gen_ai.request.max_tokens": max_tokens,
            "gen_ai.request.temperature": temperature,
        },
    ) as span:
        if ENABLE_CONTENT_EVENTS:
            span.add_event("gen_ai.input.messages", attributes={
                "gen_ai.input.messages": json.dumps(messages),
            })

        start_time = time.monotonic()
        common_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
        }
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            response_model = data.get("model", model)
            usage = data.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            choices = data.get("choices", [])
            finish_reasons = [c.get("finish_reason", "") for c in choices]
            response_id = data.get("id", "")

            span.set_attribute("gen_ai.response.model", response_model)
            span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
            span.set_attribute("gen_ai.response.id", response_id)

            # Missing usage is unknown, not zero. Preserve an explicitly reported
            # zero while rejecting malformed, negative or boolean token counts.
            for field, token_type in (("prompt_tokens", "input"), ("completion_tokens", "output")):
                count = usage.get(field)
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    span.set_attribute(f"gen_ai.usage.{token_type}_tokens", count)
                    token_usage_histogram.record(count, attributes={
                        **common_attrs, "gen_ai.token.type": token_type,
                    })

            if ENABLE_CONTENT_EVENTS and choices:
                output_messages = [
                    {"role": "assistant", "content": c.get("message", {}).get("content", "")}
                    for c in choices
                ]
                span.add_event("gen_ai.output.messages", attributes={
                    "gen_ai.output.messages": json.dumps(output_messages),
                })

            span.set_status(trace.StatusCode.OK)
            return data
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            error_type = type(e).__name__
            if isinstance(e, requests.HTTPError) and e.response is not None:
                error_type = str(e.response.status_code)
            span.set_attribute("error.type", error_type)
            common_attrs["error.type"] = error_type
            raise
        finally:
            duration = time.monotonic() - start_time
            operation_duration_histogram.record(duration, attributes=common_attrs)
