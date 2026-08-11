import iris_lib


def test_train_iris_returns_reasonable_accuracy():
    acc = iris_lib.train_iris()
    assert 0.8 <= acc <= 1.0


def test_registered_model_payload_shape():
    p = iris_lib.build_registered_model_payload("iris", "demo model")
    assert p["name"] == "iris"
    assert p["description"] == "demo model"


def test_model_version_payload_shape():
    p = iris_lib.build_model_version_payload("iris", "v1", iris_lib.IRIS_STORAGE_URI)
    assert p["name"] == "v1"
    assert p["customProperties"]["storageUri"]["string_value"] == iris_lib.IRIS_STORAGE_URI


def test_inference_service_manifest_shape():
    m = iris_lib.build_inference_service_manifest("sklearn-iris", "kserve-test", iris_lib.IRIS_STORAGE_URI)
    assert m["apiVersion"] == "serving.kserve.io/v1beta1"
    assert m["kind"] == "InferenceService"
    assert m["metadata"]["name"] == "sklearn-iris"
    assert m["metadata"]["namespace"] == "kserve-test"
    assert m["spec"]["predictor"]["model"]["storageUri"] == iris_lib.IRIS_STORAGE_URI
    assert m["spec"]["predictor"]["model"]["modelFormat"]["name"] == "sklearn"
