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
        except requests.RequestException as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            error_type = type(e).__name__
            if hasattr(e, "response") and e.response is not None:
                error_type = str(e.response.status_code)
            span.set_attribute("error.type", error_type)
            raise
        finally:
            duration = time.monotonic() - start_time
            common_attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            }
            operation_duration_histogram.record(duration, attributes=common_attrs)

        response_model = data.get("model", model)
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        choices = data.get("choices", [])
        finish_reasons = [c.get("finish_reason", "") for c in choices]
        response_id = data.get("id", "")

        span.set_attribute("gen_ai.response.model", response_model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        span.set_attribute("gen_ai.response.id", response_id)

        token_usage_histogram.record(input_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "input",
        })
        token_usage_histogram.record(output_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "output",
        })

        if ENABLE_CONTENT_EVENTS and choices:
            output_messages = [
                {"role": "assistant", "content": c.get("message", {}).get("content", "")}
                for c in choices
            ]
            span.add_event("gen_ai.output.messages", attributes={
                "gen_ai.output.messages": json.dumps(output_messages),
            })

        return data
