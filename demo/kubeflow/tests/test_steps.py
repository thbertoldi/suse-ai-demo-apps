from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import steps


MODEL_URI = "s3://mlpipeline/runs/demo/model"


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
    def __init__(self, identifier="1", status_code=201, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"id": identifier}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise steps.httpx.HTTPStatusError(
                "request failed",
                request=None,
                response=self,
            )

    def json(self):
        return self._body


def test_run_register_creates_model_version_and_artifact(monkeypatch):
    calls = []
    responses = iter([FakeResponse("1"), FakeResponse("2"), FakeResponse("3")])

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return next(responses)

    monkeypatch.setattr(steps.httpx, "post", fake_post)
    version_id = steps.run_register(
        "http://model-registry-service:8080",
        MODEL_URI,
        model_name="iris",
        version="v1",
        source_namespace="profile",
    )

    assert version_id == "2"
    assert calls[0][0].endswith("/registered_models")
    assert calls[1][0].endswith("/registered_models/1/versions")
    assert calls[2][0].endswith("/model_versions/2/artifacts")
    assert calls[2][1]["artifactType"] == "model-artifact"
    assert calls[2][1]["uri"] == MODEL_URI
    assert all(headers == {"Authorization": "Bearer demo"} for _, _, headers in calls)


def _patch_deploy_dependencies(monkeypatch, fake_api):
    monkeypatch.setattr(steps, "_kserve_api", lambda: fake_api)
    monkeypatch.setattr(steps, "_ensure_kserve_storage", lambda *args, **kwargs: None)


def test_run_deploy_emits_kserve_span(span_exporter, monkeypatch):
    fake_api = MagicMock()
    _patch_deploy_dependencies(monkeypatch, fake_api)
    steps.run_deploy(
        "sklearn-iris",
        "profile",
        MODEL_URI,
        access_key="access",
        secret_key="secret",
    )

    spans = span_exporter.get_finished_spans()
    assert any(
        span.attributes.get("kserve.inference.service") == "sklearn-iris"
        for span in spans
    )
    manifest = fake_api.create_namespaced_custom_object.call_args.kwargs["body"]
    assert manifest["spec"]["predictor"]["model"]["storageUri"] == MODEL_URI


def test_run_deploy_translates_kfp_minio_uri_for_kserve(monkeypatch):
    fake_api = MagicMock()
    _patch_deploy_dependencies(monkeypatch, fake_api)

    steps.run_deploy(
        "sklearn-iris",
        "profile",
        "minio://mlpipeline/runs/demo/model",
        access_key="access",
        secret_key="secret",
    )

    manifest = fake_api.create_namespaced_custom_object.call_args.kwargs["body"]
    assert (
        manifest["spec"]["predictor"]["model"]["storageUri"]
        == "s3://mlpipeline/runs/demo/model"
    )


def test_run_deploy_patches_only_on_conflict(monkeypatch):
    fake_api = MagicMock()
    fake_api.create_namespaced_custom_object.side_effect = ApiException(status=409)
    _patch_deploy_dependencies(monkeypatch, fake_api)
    steps.run_deploy(
        "sklearn-iris",
        "profile",
        MODEL_URI,
        access_key="access",
        secret_key="secret",
    )
    fake_api.patch_namespaced_custom_object.assert_called_once()


def test_run_deploy_does_not_hide_forbidden_error(monkeypatch):
    fake_api = MagicMock()
    fake_api.create_namespaced_custom_object.side_effect = ApiException(status=403)
    _patch_deploy_dependencies(monkeypatch, fake_api)
    with pytest.raises(ApiException) as exc_info:
        steps.run_deploy(
            "sklearn-iris",
            "profile",
            MODEL_URI,
            access_key="access",
            secret_key="secret",
        )
    assert exc_info.value.status == 403
    fake_api.patch_namespaced_custom_object.assert_not_called()


def test_run_register_recovers_existing_ids(monkeypatch):
    post_calls = []
    get_calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        post_calls.append(url)
        if url.endswith("/registered_models") or url.endswith("/versions"):
            return FakeResponse(status_code=409)
        return FakeResponse("artifact-id")

    def fake_get(url, headers=None, timeout=None):
        get_calls.append(url)
        if url.endswith("/registered_models"):
            return FakeResponse(body={"items": [{"name": "iris", "id": "7"}]})
        return FakeResponse(body={"items": [{"name": "v1", "id": "8"}]})

    monkeypatch.setattr(steps.httpx, "post", fake_post)
    monkeypatch.setattr(steps.httpx, "get", fake_get)
    assert steps.run_register(
        "http://model-registry-service:8080",
        MODEL_URI,
        model_name="iris",
        version="v1",
        source_namespace="profile",
    ) == "8"
    assert any(url.endswith("/model_versions/8/artifacts") for url in post_calls)
    assert len(get_calls) == 2


def test_run_register_does_not_hide_non_conflict_error(monkeypatch):
    monkeypatch.setattr(
        steps.httpx,
        "post",
        lambda url, json=None, headers=None, timeout=None: FakeResponse(status_code=403),
    )
    with pytest.raises(steps.httpx.HTTPStatusError):
        steps.run_register(
            "http://model-registry-service:8080",
            MODEL_URI,
            source_namespace="profile",
        )


def test_smoke_test_returns_prediction(monkeypatch):
    fake_api = MagicMock()
    fake_api.get_namespaced_custom_object.return_value = {
        "status": {
            "components": {
                "predictor": {
                    "latestCreatedRevision": "sklearn-iris-predictor-00002",
                    "latestReadyRevision": "sklearn-iris-predictor-00002",
                }
            }
        }
    }
    monkeypatch.setattr(steps, "_kserve_api", lambda: fake_api)
    fake_core = MagicMock()
    monkeypatch.setattr(steps, "_core_api", lambda: fake_core)
    monkeypatch.setattr(
        steps.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=200,
            body={"predictions": [0]},
        ),
    )
    assert steps.run_smoke_test(
        "sklearn-iris",
        "profile",
        timeout_seconds=1,
        interval_seconds=0,
    ) == [0]


def test_smoke_test_retries_until_predictor_revision_is_ready(monkeypatch):
    fake_api = MagicMock()
    fake_api.get_namespaced_custom_object.side_effect = [
        {"status": {"components": {"predictor": {}}}},
        {
            "status": {
                "components": {
                    "predictor": {
                        "latestCreatedRevision": "sklearn-iris-predictor-00003",
                        "latestReadyRevision": "sklearn-iris-predictor-00003"
                    }
                }
            }
        },
    ]
    monkeypatch.setattr(steps, "_kserve_api", lambda: fake_api)
    fake_core = MagicMock()
    monkeypatch.setattr(steps, "_core_api", lambda: fake_core)
    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        return FakeResponse(status_code=200, body={"predictions": [0]})

    monkeypatch.setattr(steps.httpx, "post", fake_post)
    assert steps.run_smoke_test(
        "sklearn-iris",
        "profile",
        timeout_seconds=1,
        interval_seconds=0,
    ) == [0]
    assert urls == [
        "http://suse-ai-sklearn-iris.profile.svc.cluster.local/"
        "v1/models/sklearn-iris:predict"
    ]
    service = fake_core.create_namespaced_service.call_args.kwargs["body"]
    assert (
        service["spec"]["selector"]["serving.knative.dev/revision"]
        == "sklearn-iris-predictor-00003"
    )


def test_smoke_test_waits_for_latest_created_revision(monkeypatch):
    fake_api = MagicMock()
    fake_api.get_namespaced_custom_object.side_effect = [
        {
            "status": {
                "components": {
                    "predictor": {
                        "latestCreatedRevision": "sklearn-iris-predictor-00004",
                        "latestReadyRevision": "sklearn-iris-predictor-00003",
                    }
                }
            }
        },
        {
            "status": {
                "components": {
                    "predictor": {
                        "latestCreatedRevision": "sklearn-iris-predictor-00004",
                        "latestReadyRevision": "sklearn-iris-predictor-00004",
                    }
                }
            }
        },
    ]
    monkeypatch.setattr(steps, "_kserve_api", lambda: fake_api)
    fake_core = MagicMock()
    monkeypatch.setattr(steps, "_core_api", lambda: fake_core)
    monkeypatch.setattr(
        steps.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=200,
            body={"predictions": [0]},
        ),
    )

    assert steps.run_smoke_test(
        "sklearn-iris",
        "profile",
        timeout_seconds=1,
        interval_seconds=0,
    ) == [0]
    assert fake_api.get_namespaced_custom_object.call_count == 2
    service = fake_core.create_namespaced_service.call_args.kwargs["body"]
    assert (
        service["spec"]["selector"]["serving.knative.dev/revision"]
        == "sklearn-iris-predictor-00004"
    )
