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


def test_run_register_idempotent_on_conflict(monkeypatch):
    """run_register should tolerate model/version already existing (conflict), recover the id, and complete."""
    post_calls = []
    get_calls = []

    class ConflictResponse:
        status_code = 409

        def raise_for_status(self):
            raise steps.httpx.HTTPStatusError("conflict", request=None, response=self)

        def json(self):
            return {}

    class ListResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"items": [{"name": "iris", "id": "7"}]}

    class VersionOkResponse:
        status_code = 201

        def raise_for_status(self):
            pass

        def json(self):
            return {"id": "v1-id"}

    def fake_post(url, json=None, timeout=None):
        post_calls.append(url)
        if url.endswith("/registered_models"):
            return ConflictResponse()
        elif "/versions" in url:
            return VersionOkResponse()
        return FakeResponse()

    def fake_get(url, timeout=None):
        get_calls.append(url)
        return ListResponse()

    monkeypatch.setattr(steps.httpx, "post", fake_post)
    monkeypatch.setattr(steps.httpx, "get", fake_get)

    steps.run_register("http://model-registry-service:8080", model_name="iris", version="v1")

    assert any(u.endswith("/api/model_registry/v1alpha3/registered_models") for u in post_calls), "must POST to registered_models"
    assert any(u.endswith("/api/model_registry/v1alpha3/registered_models") for u in get_calls), "must GET existing models on conflict"
    assert any("/registered_models/7/versions" in u for u in post_calls), "must POST version using recovered id=7"
