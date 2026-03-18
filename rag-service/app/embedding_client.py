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


def embed(
    base_url: str,
    model: str,
    provider: str,
    text: str,
) -> list[float]:
    with tracer.start_as_current_span(
        f"embed {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "embed",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
        },
    ) as span:
        start_time = time.monotonic()
        try:
            response = requests.post(
                f"{base_url}/embeddings",
                json={"model": model, "input": text},
                timeout=60,
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
                "gen_ai.operation.name": "embed",
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            }
            operation_duration_histogram.record(duration, attributes=common_attrs)

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

        token_usage_histogram.record(input_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "input",
        })

        embedding = data["data"][0]["embedding"]
        return embedding
