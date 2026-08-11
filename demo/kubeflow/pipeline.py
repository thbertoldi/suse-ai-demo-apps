import os

from kfp import dsl
from kfp import compiler

IMAGE = os.environ.get("IRIS_PIPELINE_IMAGE", "ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:latest")
OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://opentelemetry-collector.observability.svc.cluster.local:4317",
)
KFP_RESOURCE_ATTRS = (
    "suse.ai.component.name=kubeflow-pipelines,"
    "suse.ai.component.type=workflow-engine,"
    "suse.ai.managed=true"
)
MODEL_REGISTRY_URL = os.environ.get(
    "MODEL_REGISTRY_URL", "http://model-registry-service.kubeflow.svc.cluster.local:8080"
)


@dsl.container_component
def train():
    return dsl.ContainerSpec(image=IMAGE, command=["python", "-m", "steps"], args=["train"])


@dsl.container_component
def register():
    return dsl.ContainerSpec(image=IMAGE, command=["python", "-m", "steps"], args=["register"])


@dsl.container_component
def deploy():
    return dsl.ContainerSpec(image=IMAGE, command=["python", "-m", "steps"], args=["deploy"])


def _with_otel(task, extra: dict | None = None):
    task.set_env_variable("OTEL_EXPORTER_OTLP_ENDPOINT", OTLP_ENDPOINT)
    task.set_env_variable("OTEL_RESOURCE_ATTRIBUTES", KFP_RESOURCE_ATTRS)
    for k, v in (extra or {}).items():
        task.set_env_variable(k, v)
    return task


@dsl.pipeline(name="iris-lifecycle", description="Train, register, and deploy the iris model")
def iris_pipeline():
    t = _with_otel(train())
    r = _with_otel(register(), {"MODEL_REGISTRY_URL": MODEL_REGISTRY_URL}).after(t)
    _with_otel(deploy(), {"ISVC_NAME": "sklearn-iris", "ISVC_NAMESPACE": "kserve-test"}).after(r)


def compile_pipeline(path: str = "iris_pipeline.yaml"):
    compiler.Compiler().compile(iris_pipeline, path)


if __name__ == "__main__":
    compile_pipeline()
    print("compiled to iris_pipeline.yaml")
