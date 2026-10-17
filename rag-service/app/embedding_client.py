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
        common_attrs = {
            "gen_ai.operation.name": "embed",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
        }
        try:
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embeddings"][0]

            input_tokens = data.get("prompt_eval_count")
            if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens >= 0:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                token_usage_histogram.record(input_tokens, attributes={
                    **common_attrs, "gen_ai.token.type": "input",
                })

            span.set_status(trace.StatusCode.OK)
            return embedding
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
