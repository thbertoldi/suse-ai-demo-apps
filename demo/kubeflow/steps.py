import os
import sys

import httpx
from opentelemetry import trace
from opentelemetry.trace import SpanKind

import iris_lib
from otel_boot import init_tracer

tracer = trace.get_tracer("kubeflow-pipelines")


def run_train() -> float:
    with tracer.start_as_current_span("train iris", kind=SpanKind.INTERNAL) as span:
        acc = iris_lib.train_iris()
        span.set_attribute("iris.accuracy", acc)
        return acc


def run_register(registry_url: str, model_name: str = "iris", version: str = "v1") -> None:
    base = registry_url.rstrip("/")
    rm_url = f"{base}/api/model_registry/v1alpha3/registered_models"
    resp = httpx.post(rm_url, json=iris_lib.build_registered_model_payload(model_name, "demo iris model"), timeout=30)
    resp.raise_for_status()
    rm_id = resp.json().get("id", "")
    mv_url = f"{base}/api/model_registry/v1alpha3/registered_models/{rm_id}/versions"
    resp2 = httpx.post(mv_url, json=iris_lib.build_model_version_payload(model_name, version, iris_lib.IRIS_STORAGE_URI), timeout=30)
    resp2.raise_for_status()


def _kserve_api():
    from kubernetes import client, config
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    return client.CustomObjectsApi()


def run_deploy(name: str = "sklearn-iris", namespace: str = "kserve-test") -> None:
    manifest = iris_lib.build_inference_service_manifest(name, namespace, iris_lib.IRIS_STORAGE_URI)
    api = _kserve_api()
    with tracer.start_as_current_span(
        "deploy sklearn-iris",
        kind=SpanKind.CLIENT,
        attributes={"kserve.inference.service": name},
    ):
        try:
            api.create_namespaced_custom_object(
                group="serving.kserve.io", version="v1beta1",
                namespace=namespace, plural="inferenceservices", body=manifest,
            )
        except Exception:
            api.patch_namespaced_custom_object(
                group="serving.kserve.io", version="v1beta1",
                namespace=namespace, plural="inferenceservices", name=name, body=manifest,
            )


def main():
    init_tracer()
    global tracer
    tracer = trace.get_tracer("kubeflow-pipelines")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        print("accuracy:", run_train())
    elif cmd == "register":
        run_register(os.environ["MODEL_REGISTRY_URL"])
    elif cmd == "deploy":
        run_deploy(os.environ.get("ISVC_NAME", "sklearn-iris"),
                   os.environ.get("ISVC_NAMESPACE", "kserve-test"))
    else:
        raise SystemExit(f"unknown command: {cmd!r}")
    trace.get_tracer_provider().shutdown()


if __name__ == "__main__":
    main()
