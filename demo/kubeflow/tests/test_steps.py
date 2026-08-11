from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import steps


@pytest.fixture(scope="module", autouse=True)
def _provider():
    trace.set_tracer_provider(TracerProvider())


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()


class FakeResponse:
    status_code = 201

    def raise_for_status(self):
        pass

    def json(self):
        return {"id": "1"}


def test_run_register_posts_to_registry(monkeypatch):
    calls = []
    monkeypatch.setattr(steps.httpx, "post",
                        lambda url, json=None, timeout=None: calls.append(url) or FakeResponse())
    steps.run_register("http://model-registry-service:8080", model_name="iris", version="v1")
    assert any("model-registry" in u for u in calls)
    assert any(u.endswith("/api/model_registry/v1alpha3/registered_models") for u in calls)


def test_run_deploy_emits_kserve_span(span_exporter, monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setattr(steps, "_kserve_api", lambda: fake_api)
    steps.run_deploy("sklearn-iris", "kserve-test")

    spans = span_exporter.get_finished_spans()
    match = [s for s in spans if s.attributes.get("kserve.inference.service") == "sklearn-iris"]
    assert match, "expected a span tagged kserve.inference.service=sklearn-iris"
