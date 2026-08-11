"""Pure, unit-testable helpers for the Iris model lifecycle demo."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def prepare_iris_dataset(output_path: str) -> int:
    """Write the sklearn Iris dataset as a portable KFP directory artifact."""
    from sklearn.datasets import load_iris

    artifact_dir = Path(output_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / "iris.csv"
    iris = load_iris()
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*FEATURE_NAMES, "target"])
        for features, target in zip(iris.data, iris.target, strict=True):
            writer.writerow([*map(float, features), int(target)])
    return len(iris.data)


def train_iris_model(dataset_path: str, model_path: str) -> dict[str, Any]:
    """Train and persist a model, returning bounded evaluation data."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    source = Path(dataset_path) / "iris.csv"
    rows = np.genfromtxt(source, delimiter=",", skip_header=1)
    features, targets = rows[:, :4], rows[:, 4].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=42,
        stratify=targets,
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    artifact_dir = Path(model_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=list(range(len(TARGET_NAMES))),
        ).tolist(),
        "test_rows": int(len(y_test)),
    }


def train_iris() -> float:
    """Compatibility helper used by unit tests and local experimentation."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="suse-ai-iris-") as directory:
        dataset_path = str(Path(directory) / "dataset")
        model_path = str(Path(directory) / "model")
        prepare_iris_dataset(dataset_path)
        return float(train_iris_model(dataset_path, model_path)["accuracy"])


def build_registered_model_payload(name: str, description: str = "") -> dict:
    return {"name": name, "description": description}


def build_model_version_payload(
    name: str,
    version: str,
    storage_uri: str,
    registered_model_id: str,
) -> dict:
    return {
        "name": version,
        "description": f"{name} version {version}",
        "registeredModelId": registered_model_id,
        "customProperties": {
            "storageUri": {
                "metadataType": "MetadataStringValue",
                "string_value": storage_uri,
            },
        },
    }


def build_model_artifact_payload(
    model_name: str,
    version: str,
    storage_uri: str,
    *,
    source_namespace: str,
    source_run_id: str = "",
    service_account_name: str = "suse-ai-kserve-model",
    storage_key: str = "suse-ai-kserve-s3",
) -> dict:
    """Build the v0.3.x Model Registry ModelArtifact request body."""
    payload = {
        "artifactType": "model-artifact",
        "name": f"{model_name}-{version}",
        "description": f"KFP artifact for {model_name} version {version}",
        "modelFormatName": "sklearn",
        "modelFormatVersion": "1",
        "storageKey": storage_key,
        "storagePath": storage_uri,
        "serviceAccountName": service_account_name,
        "modelSourceKind": "pipelines",
        "modelSourceClass": "pipelinerun",
        "modelSourceGroup": source_namespace,
        "modelSourceName": f"iris-lifecycle/{version}",
        "uri": storage_uri,
        "state": "LIVE",
    }
    if source_run_id:
        payload["modelSourceId"] = source_run_id
    return payload


def build_kserve_storage_secret(
    name: str,
    namespace: str,
    access_key: str,
    secret_key: str,
    endpoint: str,
) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/s3-endpoint": endpoint,
                "serving.kserve.io/s3-usehttps": "0",
                "serving.kserve.io/s3-region": "us-east-1",
                "serving.kserve.io/s3-useanoncredential": "false",
            },
        },
        "type": "Opaque",
        "stringData": {
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
        },
    }


def build_model_service_account(
    name: str,
    namespace: str,
    secret_name: str,
    image_pull_secret_name: str = "suse-ai-registry",
) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": name, "namespace": namespace},
        "secrets": [{"name": secret_name}],
        # The SUSE KServe installation uses product images from the protected
        # registry.  Profile service accounts do not inherit pull secrets.
        "imagePullSecrets": [{"name": image_pull_secret_name}],
    }


def build_inference_service_manifest(
    name: str,
    namespace: str,
    storage_uri: str,
    service_account_name: str = "suse-ai-kserve-model",
) -> dict:
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/enable-prometheus-scraping": "true",
                "autoscaling.knative.dev/min-scale": "1",
            },
        },
        "spec": {
            "predictor": {
                "serviceAccountName": service_account_name,
                "model": {
                    "modelFormat": {"name": "sklearn"},
                    "storageUri": storage_uri,
                },
            },
        },
    }


def build_prediction_service(
    name: str,
    namespace: str,
    inference_service_name: str,
    revision: str,
) -> dict:
    """Build a stable service that targets only the latest ready revision."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "suse-ai-demo-kserve-direct",
                "app.kubernetes.io/part-of": "suse-ai-demo",
            },
        },
        "spec": {
            "selector": {
                "serving.kserve.io/inferenceservice": inference_service_name,
                "serving.knative.dev/revision": revision,
                "component": "predictor",
            },
            "ports": [
                {
                    "name": "http",
                    "port": 80,
                    "protocol": "TCP",
                    "targetPort": "queue-port",
                }
            ],
        },
    }
