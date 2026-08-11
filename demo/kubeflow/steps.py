"""Executable, observable steps for the Iris KFP lifecycle."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from typing import Iterator

import httpx
from kubernetes.client.exceptions import ApiException
from opentelemetry import trace
from opentelemetry.trace import SpanKind

import iris_lib
import otel_boot


tracer = otel_boot.tracer


@contextmanager
def _observe_step(step_name: str) -> Iterator[trace.Span]:
    started = time.monotonic()
    outcome = "success"
    with tracer.start_as_current_span(
        f"kubeflow.pipeline.step {step_name}",
        kind=SpanKind.INTERNAL,
        attributes={"suse.ai.kubeflow.pipeline.step.name": step_name},
    ) as span:
        try:
            yield span
            span.set_status(trace.StatusCode.OK)
        except Exception as exc:
            outcome = "error"
            span.set_status(trace.StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__name__)
            raise
        finally:
            attributes = {"step": step_name, "outcome": outcome}
            otel_boot.step_runs.add(1, attributes)
            otel_boot.step_duration.record(
                time.monotonic() - started,
                attributes,
            )


def _registry_headers() -> dict[str, str]:
    token = os.environ.get("MODEL_REGISTRY_BEARER_TOKEN", "demo").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _items(response: httpx.Response) -> list[dict]:
    body = response.json()
    return body if isinstance(body, list) else body.get("items", [])


def run_prepare(dataset_path: str) -> int:
    with _observe_step("prepare-dataset") as span:
        rows = iris_lib.prepare_iris_dataset(dataset_path)
        span.set_attribute("suse.ai.kubeflow.dataset.rows", rows)
        return rows


def run_train(dataset_path: str, model_path: str) -> dict:
    with _observe_step("train-model") as span:
        result = iris_lib.train_iris_model(dataset_path, model_path)
        accuracy = float(result["accuracy"])
        span.set_attribute("suse.ai.kubeflow.model.accuracy", accuracy)
        span.set_attribute("suse.ai.kubeflow.evaluation.rows", result["test_rows"])
        otel_boot.model_accuracy.record(
            accuracy,
            {"model_name": "iris", "model_format": "sklearn"},
        )
        return result


def run_gate(accuracy: float, minimum_accuracy: float) -> None:
    with _observe_step("quality-gate") as span:
        span.set_attribute("suse.ai.kubeflow.model.accuracy", accuracy)
        span.set_attribute("suse.ai.kubeflow.model.minimum_accuracy", minimum_accuracy)
        if accuracy < minimum_accuracy:
            raise RuntimeError(
                f"model accuracy {accuracy:.4f} is below the required "
                f"{minimum_accuracy:.4f}"
            )


def _find_named_id(url: str, name: str, headers: dict[str, str]) -> str:
    response = httpx.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    for item in _items(response):
        if item.get("name") == name:
            return str(item.get("id", ""))
    return ""


def run_register(
    registry_url: str,
    model_uri: str,
    model_name: str = "iris",
    version: str = "v1",
    source_namespace: str = "kubeflow-user-example-com",
    source_run_id: str = "",
) -> str:
    """Register a model, version, and physical ModelArtifact idempotently."""
    base = registry_url.rstrip("/")
    rm_url = f"{base}/api/model_registry/v1alpha3/registered_models"
    headers = _registry_headers()

    with _observe_step("register-model") as span:
        span.set_attribute("url.full", rm_url)
        span.set_attribute("suse.ai.kubeflow.model.uri", model_uri)
        with tracer.start_as_current_span(
            "register iris model artifact",
            kind=SpanKind.CLIENT,
            attributes={"url.full": rm_url},
        ):
            response = httpx.post(
                rm_url,
                json=iris_lib.build_registered_model_payload(
                    model_name,
                    "SUSE AI observability Iris demo model",
                ),
                headers=headers,
                timeout=30,
            )

            try:
                response.raise_for_status()
                registered_model_id = str(response.json().get("id", ""))
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code != 409:
                    raise
                registered_model_id = _find_named_id(rm_url, model_name, headers)

            if not registered_model_id:
                raise RuntimeError(
                    f"failed to determine registered model id for {model_name!r}"
                )

            versions_url = f"{rm_url}/{registered_model_id}/versions"
            version_response = httpx.post(
                versions_url,
                json=iris_lib.build_model_version_payload(
                    model_name,
                    version,
                    model_uri,
                    registered_model_id,
                ),
                headers=headers,
                timeout=30,
            )
            try:
                version_response.raise_for_status()
                model_version_id = str(version_response.json().get("id", ""))
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code != 409:
                    raise
                model_version_id = _find_named_id(versions_url, version, headers)

            if not model_version_id:
                raise RuntimeError(
                    f"failed to determine model version id for {model_name!r} {version!r}"
                )

            artifacts_url = (
                f"{base}/api/model_registry/v1alpha3/model_versions/"
                f"{model_version_id}/artifacts"
            )
            artifact_response = httpx.post(
                artifacts_url,
                json=iris_lib.build_model_artifact_payload(
                    model_name,
                    version,
                    model_uri,
                    source_namespace=source_namespace,
                    source_run_id=source_run_id,
                ),
                headers=headers,
                timeout=30,
            )
            try:
                artifact_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code != 409:
                    raise

        span.set_attribute("suse.ai.kubeflow.registry.model_version_id", model_version_id)
        return model_version_id


def _kserve_api():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def _core_api():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api()


def _ensure_kserve_storage(
    namespace: str,
    *,
    service_account_name: str,
    secret_name: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> None:
    core = _core_api()
    secret = iris_lib.build_kserve_storage_secret(
        secret_name,
        namespace,
        access_key,
        secret_key,
        endpoint,
    )
    try:
        core.create_namespaced_secret(namespace=namespace, body=secret)
    except ApiException as exc:
        if exc.status != 409:
            raise
        core.patch_namespaced_secret(
            name=secret_name,
            namespace=namespace,
            body=secret,
        )

    service_account = iris_lib.build_model_service_account(
        service_account_name,
        namespace,
        secret_name,
    )
    try:
        core.create_namespaced_service_account(
            namespace=namespace,
            body=service_account,
        )
    except ApiException as exc:
        if exc.status != 409:
            raise
        core.patch_namespaced_service_account(
            name=service_account_name,
            namespace=namespace,
            body=service_account,
        )


def run_deploy(
    name: str,
    namespace: str,
    storage_uri: str,
    *,
    service_account_name: str = "suse-ai-kserve-model",
    storage_secret_name: str = "suse-ai-kserve-s3",
    s3_endpoint: str = "seaweedfs.kubeflow:9000",
    access_key: str | None = None,
    secret_key: str | None = None,
) -> None:
    """Deploy the exact KFP Model artifact to KServe."""
    # KFP's S3-compatible artifact repository serializes URIs with a
    # ``minio://`` scheme.  KServe uses the same object and credentials, but
    # its storage initializer only accepts the standard ``s3://`` scheme.
    kserve_storage_uri = (
        f"s3://{storage_uri.removeprefix('minio://')}"
        if storage_uri.startswith("minio://")
        else storage_uri
    )
    resolved_access_key = access_key or os.environ.get("S3_ACCESS_KEY", "")
    resolved_secret_key = secret_key or os.environ.get("S3_SECRET_KEY", "")
    if not resolved_access_key or not resolved_secret_key:
        raise RuntimeError("S3_ACCESS_KEY and S3_SECRET_KEY are required for KServe")

    with _observe_step("deploy-model") as span:
        span.set_attribute("kserve.inference.service", name)
        span.set_attribute("suse.ai.kubeflow.model.uri", storage_uri)
        span.set_attribute("suse.ai.kserve.storage.uri", kserve_storage_uri)
        _ensure_kserve_storage(
            namespace,
            service_account_name=service_account_name,
            secret_name=storage_secret_name,
            endpoint=s3_endpoint,
            access_key=resolved_access_key,
            secret_key=resolved_secret_key,
        )
        manifest = iris_lib.build_inference_service_manifest(
            name,
            namespace,
            kserve_storage_uri,
            service_account_name,
        )
        api = _kserve_api()
        with tracer.start_as_current_span(
            "deploy sklearn-iris",
            kind=SpanKind.CLIENT,
            attributes={"kserve.inference.service": name},
        ):
            try:
                api.create_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    body=manifest,
                )
            except ApiException as exc:
                if exc.status != 409:
                    raise
                api.patch_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=name,
                    body=manifest,
                )


def run_smoke_test(
    name: str,
    namespace: str,
    *,
    prediction_service_name: str = "suse-ai-sklearn-iris",
    timeout_seconds: int = 300,
    interval_seconds: float = 5,
) -> list:
    payload = {"instances": [[5.1, 3.5, 1.4, 0.2]]}
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    with _observe_step("smoke-test") as span:
        span.set_attribute("kserve.inference.service", name)
        while time.monotonic() < deadline:
            try:
                inference_service = _kserve_api().get_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=name,
                )
                predictor = (
                    inference_service.get("status", {})
                    .get("components", {})
                    .get("predictor", {})
                )
                created_revision = predictor.get("latestCreatedRevision")
                ready_revision = predictor.get("latestReadyRevision")
                if not created_revision:
                    raise RuntimeError("KServe predictor has no created revision yet")
                if ready_revision != created_revision:
                    raise RuntimeError("KServe predictor has no ready revision yet")
                service = iris_lib.build_prediction_service(
                    prediction_service_name,
                    namespace,
                    name,
                    created_revision,
                )
                core = _core_api()
                try:
                    core.create_namespaced_service(namespace=namespace, body=service)
                except ApiException as exc:
                    if exc.status != 409:
                        raise
                    core.patch_namespaced_service(
                        name=prediction_service_name,
                        namespace=namespace,
                        body=service,
                    )
                # Generated KServe addresses resolve through Kubeflow's OIDC
                # gateway. This service bypasses that browser-facing route and
                # its revision selector prevents stale models after updates.
                url = (
                    f"http://{prediction_service_name}.{namespace}.svc.cluster.local/"
                    f"v1/models/{name}:predict"
                )
                span.set_attribute("url.full", url)
                with tracer.start_as_current_span(
                    "predict sklearn-iris",
                    kind=SpanKind.CLIENT,
                    attributes={
                        "kserve.inference.service": name,
                        "url.full": url,
                    },
                ):
                    response = httpx.post(url, json=payload, timeout=15)
                    response.raise_for_status()
                    predictions = response.json().get("predictions", [])
                if not predictions:
                    raise RuntimeError("KServe response contained no predictions")
                otel_boot.smoke_tests.add(
                    1,
                    {"inference_service": name, "outcome": "success"},
                )
                span.set_attribute("suse.ai.kubeflow.prediction", str(predictions[0]))
                return predictions
            except Exception as exc:  # service readiness is intentionally retried
                last_error = exc
                time.sleep(interval_seconds)

        otel_boot.smoke_tests.add(
            1,
            {"inference_service": name, "outcome": "error"},
        )
        raise RuntimeError(
            f"KServe smoke test did not succeed within {timeout_seconds}s: {last_error}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-path", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--dataset-path", required=True)
    train.add_argument("--model-path", required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--accuracy", required=True, type=float)
    gate.add_argument("--minimum-accuracy", required=True, type=float)

    register = subparsers.add_parser("register")
    register.add_argument(
        "--registry-url",
        default=os.environ.get("MODEL_REGISTRY_URL", ""),
    )
    register.add_argument("--model-uri", required=True)
    register.add_argument("--model-name", default="iris")
    register.add_argument("--version", required=True)
    register.add_argument("--source-namespace", required=True)
    register.add_argument("--source-run-id", default="")

    deploy = subparsers.add_parser("deploy")
    deploy.add_argument("--name", default="sklearn-iris")
    deploy.add_argument("--namespace", required=True)
    deploy.add_argument("--storage-uri", required=True)
    deploy.add_argument("--s3-endpoint", default="seaweedfs.kubeflow:9000")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--name", default="sklearn-iris")
    smoke.add_argument("--namespace", required=True)
    smoke.add_argument(
        "--prediction-service-name",
        default="suse-ai-sklearn-iris",
    )
    smoke.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def main() -> None:
    args = _parser().parse_args()
    providers = otel_boot.init_telemetry()
    try:
        if args.command == "prepare":
            print(f"rows={run_prepare(args.dataset_path)}")
        elif args.command == "train":
            print(run_train(args.dataset_path, args.model_path))
        elif args.command == "gate":
            run_gate(args.accuracy, args.minimum_accuracy)
        elif args.command == "register":
            print(
                "model_version_id="
                + run_register(
                    args.registry_url,
                    args.model_uri,
                    args.model_name,
                    args.version,
                    args.source_namespace,
                    args.source_run_id,
                )
            )
        elif args.command == "deploy":
            run_deploy(
                args.name,
                args.namespace,
                args.storage_uri,
                s3_endpoint=args.s3_endpoint,
            )
        elif args.command == "smoke":
            print(
                run_smoke_test(
                    args.name,
                    args.namespace,
                    prediction_service_name=args.prediction_service_name,
                    timeout_seconds=args.timeout_seconds,
                )
            )
    finally:
        providers.shutdown()


if __name__ == "__main__":
    main()
