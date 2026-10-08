from app.otel_instrumentation import invoke_agent_span


def test_agent_invocation_identifies_its_configured_model_provider(monkeypatch, span_exporter):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    with invoke_agent_span("review-agent", "qwen-demo"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.agent.name"] == "review-agent"
    assert span.attributes["gen_ai.request.model"] == "qwen-demo"
    assert span.attributes["gen_ai.provider.name"] == "vllm"


def test_model_endpoint_identity_excludes_credentials_and_url_path(monkeypatch, span_exporter):
    monkeypatch.setenv("LLM_BASE_URL", "https://test-user:test-password@models.example/v1?token=test-token")
    with invoke_agent_span("review-agent", "qwen-demo"):
        pass
    attrs = span_exporter.get_finished_spans()[0].attributes
    assert attrs["server.address"] == "models.example"
    assert attrs["server.port"] == 443
    assert "test-password" not in str(attrs)
    assert "test-token" not in str(attrs)


def test_tool_duration_histogram_resolves_submillisecond_calls(monkeypatch):
    import importlib.util
    from pathlib import Path
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from app import otel_instrumentation

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(metrics, "get_meter", provider.get_meter)
    spec = importlib.util.spec_from_file_location(
        "isolated_instrumentation", Path(otel_instrumentation.__file__)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tool_duration_histogram.record(0.0002)
    data = reader.get_metrics_data()
    point = data.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points[0]
    occupied = next(i for i, count in enumerate(point.bucket_counts) if count)
    assert point.explicit_bounds[occupied] < 0.001
    assert point.count == 1
    assert point.sum == 0.0002
    provider.shutdown()
