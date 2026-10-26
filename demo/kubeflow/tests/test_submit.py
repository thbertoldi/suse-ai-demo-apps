from unittest.mock import MagicMock
from types import SimpleNamespace

import submit


def test_set_user_header_configures_all_kfp_apis():
    client = MagicMock()
    apis = []
    for name in (
        "_experiment_api",
        "_healthz_api",
        "_pipelines_api",
        "_recurring_run_api",
        "_run_api",
        "_upload_api",
    ):
        api = MagicMock()
        api.api_client.default_headers = {}
        setattr(client, name, api)
        apis.append(api)

    submit._set_user_header(client, "user@example.com")

    assert all(
        api.api_client.default_headers["kubeflow-userid"] == "user@example.com"
        for api in apis
    )


def test_upload_versioned_pipeline_creates_pipeline_and_version(tmp_path):
    package = tmp_path / "pipeline.yaml"
    package.write_text("pipelineInfo: {}")
    client = MagicMock()
    client.list_pipelines.return_value = SimpleNamespace(pipelines=[])
    client.upload_pipeline.return_value = SimpleNamespace(pipeline_id="pipeline-1")
    client.upload_pipeline_version.return_value = SimpleNamespace(
        pipeline_version_id="version-1"
    )

    pipeline_id, version_id = submit._upload_versioned_pipeline(
        client,
        package,
        "iris-lifecycle",
        "demo-1",
        "profile",
    )

    assert (pipeline_id, version_id) == ("pipeline-1", "version-1")
    client.upload_pipeline.assert_called_once()
    client.upload_pipeline_version.assert_called_once()


def test_upload_versioned_pipeline_reuses_namespaced_pipeline(tmp_path):
    package = tmp_path / "pipeline.yaml"
    package.write_text("pipelineInfo: {}")
    client = MagicMock()
    client.list_pipelines.return_value = SimpleNamespace(
        pipelines=[
            SimpleNamespace(
                pipeline_id="pipeline-1",
                display_name="iris-lifecycle",
            )
        ]
    )
    client.upload_pipeline_version.return_value = SimpleNamespace(
        pipeline_version_id="version-2"
    )

    pipeline_id, version_id = submit._upload_versioned_pipeline(
        client,
        package,
        "iris-lifecycle",
        "demo-2",
        "profile",
    )

    assert (pipeline_id, version_id) == ("pipeline-1", "version-2")
    client.list_pipelines.assert_called_once_with(
        page_size=100,
        namespace="profile",
    )
    client.upload_pipeline.assert_not_called()
