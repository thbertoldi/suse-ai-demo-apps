from pathlib import Path

import iris_lib


MODEL_URI = "s3://mlpipeline/runs/demo/model"


def test_prepare_and_train_create_real_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    model = tmp_path / "model"
    assert iris_lib.prepare_iris_dataset(str(dataset)) == 150
    result = iris_lib.train_iris_model(str(dataset), str(model))
    assert 0.8 <= result["accuracy"] <= 1.0
    assert result["test_rows"] > 0
    assert len(result["confusion_matrix"]) == 3
    assert (dataset / "iris.csv").is_file()
    assert (model / "model.joblib").is_file()


def test_train_iris_returns_reasonable_accuracy():
    assert 0.8 <= iris_lib.train_iris() <= 1.0


def test_registered_model_payload_shape():
    payload = iris_lib.build_registered_model_payload("iris", "demo model")
    assert payload == {"name": "iris", "description": "demo model"}


def test_model_version_and_artifact_payloads():
    version = iris_lib.build_model_version_payload("iris", "v1", MODEL_URI, "7")
    assert version["registeredModelId"] == "7"
    assert version["customProperties"]["storageUri"]["string_value"] == MODEL_URI

    artifact = iris_lib.build_model_artifact_payload(
        "iris",
        "v1",
        MODEL_URI,
        source_namespace="profile",
        source_run_id="run-1",
    )
    assert artifact["artifactType"] == "model-artifact"
    assert artifact["uri"] == MODEL_URI
    assert artifact["modelFormatName"] == "sklearn"
    assert artifact["modelSourceId"] == "run-1"
    assert artifact["modelSourceGroup"] == "profile"


def test_kserve_objects_reference_the_pipeline_artifact():
    secret = iris_lib.build_kserve_storage_secret(
        "s3-creds",
        "profile",
        "access",
        "secret",
        "seaweedfs.kubeflow:9000",
    )
    assert secret["stringData"]["AWS_ACCESS_KEY_ID"] == "access"
    assert secret["metadata"]["annotations"]["serving.kserve.io/s3-usehttps"] == "0"

    service_account = iris_lib.build_model_service_account(
        "model-sa",
        "profile",
        "s3-creds",
    )
    assert service_account["secrets"] == [{"name": "s3-creds"}]
    assert service_account["imagePullSecrets"] == [{"name": "suse-ai-registry"}]

    manifest = iris_lib.build_inference_service_manifest(
        "sklearn-iris",
        "profile",
        MODEL_URI,
        "model-sa",
    )
    predictor = manifest["spec"]["predictor"]
    assert predictor["serviceAccountName"] == "model-sa"
    assert predictor["model"]["storageUri"] == MODEL_URI
    assert predictor["model"]["modelFormat"]["name"] == "sklearn"

    service = iris_lib.build_prediction_service(
        "stable-predictor",
        "profile",
        "sklearn-iris",
        "sklearn-iris-predictor-00002",
    )
    assert service["spec"]["selector"] == {
        "serving.kserve.io/inferenceservice": "sklearn-iris",
        "serving.knative.dev/revision": "sklearn-iris-predictor-00002",
        "component": "predictor",
    }
    assert service["spec"]["ports"][0]["targetPort"] == "queue-port"
