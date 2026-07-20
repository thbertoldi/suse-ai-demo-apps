import time
import requests
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
        error_type = None
        try:
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            error_type = type(e).__name__
            if hasattr(e, "response") and e.response is not None:
                error_type = str(e.response.status_code)
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.set_attribute("error.type", error_type)
            raise
        finally:
            duration = time.monotonic() - start_time
            common_attrs = {
                "gen_ai.operation.name": "embed",
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            }
            duration_attrs = {**common_attrs, "error.type": error_type} if error_type else common_attrs
            operation_duration_histogram.record(duration, attributes=duration_attrs)

        input_tokens = data.get("prompt_eval_count", 0)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

        token_usage_histogram.record(input_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "input",
        })

        embedding = data["embeddings"][0]
        return embedding
