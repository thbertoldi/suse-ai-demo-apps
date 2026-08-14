import os
import tempfile
from pathlib import Path

import pipeline


def test_pipeline_compiles_with_kfp_component_env():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pipeline.yaml")
        pipeline.compile_pipeline(out)
        text = open(out).read()
    assert "suse.ai.component.name=kubeflow-pipelines" in text
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in text
    assert "MODEL_REGISTRY_BEARER_TOKEN" in text
    assert "s3://mlpipeline/suse-ai-demo" in text
    assert "mlpipeline-minio-artifact" in text
    assert "prepare-dataset" in text
    assert "train-model" in text
    assert "register-model" in text
    assert "deploy-model" in text
    assert "smoke-test" in text
    assert "--model-uri" in text
    assert "--storage-uri" in text


def test_runtime_image_includes_kfp_lightweight_component_executor():
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text()
    assert "kfp==2.17.0" in requirements.splitlines()
