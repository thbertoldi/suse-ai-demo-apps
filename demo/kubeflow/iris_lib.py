"""Pure, unit-testable helpers for the iris lifecycle pipeline."""

IRIS_STORAGE_URI = "gs://kfserving-examples/models/sklearn/1.0/model"


def train_iris() -> float:
    """Train a LogisticRegression on the iris dataset; return test accuracy."""
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def build_registered_model_payload(name: str, description: str = "") -> dict:
    return {"name": name, "description": description}


def build_model_version_payload(name: str, version: str, storage_uri: str) -> dict:
    return {
        "name": version,
        "description": f"{name} version {version}",
        "customProperties": {
            "storageUri": {"metadataType": "MetadataStringValue", "string_value": storage_uri},
        },
    }


def build_inference_service_manifest(name: str, namespace: str, storage_uri: str) -> dict:
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
                "model": {
                    "modelFormat": {"name": "sklearn"},
                    "storageUri": storage_uri,
                }
            }
        },
    }
