"""KFP definition for a real Iris train/register/deploy lifecycle."""

import os

from kfp import compiler, dsl, kubernetes


IMAGE = os.environ.get(
    "IRIS_PIPELINE_IMAGE",
    "ghcr.io/thbertoldi/suse-ai-demo-iris-pipeline:latest",
)
OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://opentelemetry-collector.observability.svc.cluster.local:4317",
)
PIPELINE_NAMESPACE = os.environ.get(
    "KFP_NAMESPACE",
    "kubeflow-user-example-com",
)
KFP_RESOURCE_ATTRS = (
    "service.namespace=" + PIPELINE_NAMESPACE + ","
    "suse.ai.component.name=kubeflow-pipelines,"
    "suse.ai.component.type=workflow-engine,"
    "suse.ai.managed=true"
)
MODEL_REGISTRY_URL = os.environ.get(
    "MODEL_REGISTRY_URL",
    "http://model-registry-service.kubeflow.svc.cluster.local:8080",
)
MODEL_REGISTRY_BEARER_TOKEN = os.environ.get(
    "MODEL_REGISTRY_BEARER_TOKEN",
    "demo",
)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def prepare_dataset(dataset: dsl.Output[dsl.Dataset]) -> int:
    import otel_boot
    import steps

    providers = otel_boot.init_telemetry()
    try:
        rows = steps.run_prepare(dataset.path)
        dataset.metadata["rows"] = rows
        dataset.metadata["feature_names"] = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        return rows
    finally:
        providers.shutdown()


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def train_model(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    classification_metrics: dsl.Output[dsl.ClassificationMetrics],
) -> float:
    import otel_boot
    import steps

    providers = otel_boot.init_telemetry()
    try:
        result = steps.run_train(dataset.path, model.path)
        accuracy = float(result["accuracy"])
        metrics.log_metric("accuracy", accuracy)
        metrics.log_metric("test_rows", int(result["test_rows"]))
        classification_metrics.log_confusion_matrix(
            categories=["setosa", "versicolor", "virginica"],
            matrix=result["confusion_matrix"],
        )
        model.metadata["framework"] = "scikit-learn"
        model.metadata["model_format"] = "joblib"
        model.metadata["accuracy"] = accuracy
        return accuracy
    finally:
        providers.shutdown()


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def quality_gate(accuracy: float, minimum_accuracy: float):
    import otel_boot
    import steps

    providers = otel_boot.init_telemetry()
    try:
        steps.run_gate(accuracy, minimum_accuracy)
    finally:
        providers.shutdown()


@dsl.container_component
def register_model(
    model: dsl.Input[dsl.Model],
    registry_url: str,
    model_name: str,
    model_version: str,
    source_namespace: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "steps"],
        args=[
            "register",
            "--registry-url",
            registry_url,
            "--model-uri",
            model.uri,
            "--model-name",
            model_name,
            "--version",
            model_version,
            "--source-namespace",
            source_namespace,
            "--source-run-id",
            model_version,
        ],
    )


@dsl.container_component
def deploy_model(
    model: dsl.Input[dsl.Model],
    inference_service_name: str,
    target_namespace: str,
    s3_endpoint: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "steps"],
        args=[
            "deploy",
            "--name",
            inference_service_name,
            "--namespace",
            target_namespace,
            "--storage-uri",
            model.uri,
            "--s3-endpoint",
            s3_endpoint,
        ],
    )


@dsl.container_component
def smoke_test(
    inference_service_name: str,
    target_namespace: str,
    prediction_service_name: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "steps"],
        args=[
            "smoke",
            "--name",
            inference_service_name,
            "--namespace",
            target_namespace,
            "--prediction-service-name",
            prediction_service_name,
            "--timeout-seconds",
            "300",
        ],
    )


def _with_otel(task: dsl.PipelineTask, extra: dict[str, str] | None = None):
    task.set_env_variable("OTEL_EXPORTER_OTLP_ENDPOINT", OTLP_ENDPOINT)
    task.set_env_variable("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    task.set_env_variable("OTEL_METRIC_EXPORT_INTERVAL", "10000")
    task.set_env_variable("OTEL_SERVICE_NAME", "kubeflow-pipelines-demo")
    task.set_env_variable("OTEL_RESOURCE_ATTRIBUTES", KFP_RESOURCE_ATTRS)
    task.set_env_variable("PYTHONUNBUFFERED", "1")
    for key, value in (extra or {}).items():
        task.set_env_variable(key, value)
    task.set_caching_options(False)
    return task


@dsl.pipeline(
    name="iris-lifecycle",
    description="Prepare, train, evaluate, register, deploy, and test Iris.",
    pipeline_root="s3://mlpipeline/suse-ai-demo",
)
def iris_pipeline(
    model_name: str = "iris",
    model_version: str = "manual",
    minimum_accuracy: float = 0.85,
    inference_service_name: str = "sklearn-iris",
    prediction_service_name: str = "suse-ai-sklearn-iris",
    target_namespace: str = "kubeflow-user-example-com",
    registry_url: str = MODEL_REGISTRY_URL,
    s3_endpoint: str = "seaweedfs.kubeflow:9000",
):
    dataset_task = _with_otel(prepare_dataset())
    train_task = _with_otel(train_model(dataset=dataset_task.outputs["dataset"]))
    gate_task = _with_otel(
        quality_gate(
            accuracy=train_task.outputs["Output"],
            minimum_accuracy=minimum_accuracy,
        )
    )
    register_task = _with_otel(
        register_model(
            model=train_task.outputs["model"],
            registry_url=registry_url,
            model_name=model_name,
            model_version=model_version,
            source_namespace=target_namespace,
        ),
        {"MODEL_REGISTRY_BEARER_TOKEN": MODEL_REGISTRY_BEARER_TOKEN},
    ).after(gate_task)
    deploy_task = _with_otel(
        deploy_model(
            model=train_task.outputs["model"],
            inference_service_name=inference_service_name,
            target_namespace=target_namespace,
            s3_endpoint=s3_endpoint,
        )
    ).after(register_task)
    kubernetes.use_secret_as_env(
        deploy_task,
        secret_name="mlpipeline-minio-artifact",
        secret_key_to_env={
            "accesskey": "S3_ACCESS_KEY",
            "secretkey": "S3_SECRET_KEY",
        },
    )
    _with_otel(
        smoke_test(
            inference_service_name=inference_service_name,
            target_namespace=target_namespace,
            prediction_service_name=prediction_service_name,
        )
    ).after(deploy_task)


def compile_pipeline(path: str = "iris_pipeline.yaml") -> None:
    compiler.Compiler().compile(iris_pipeline, path)


if __name__ == "__main__":
    compile_pipeline()
    print("compiled to iris_pipeline.yaml")
