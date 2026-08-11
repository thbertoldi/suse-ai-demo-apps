import os
import tempfile

import pipeline


def test_pipeline_compiles_with_kfp_component_env():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pipeline.yaml")
        pipeline.compile_pipeline(out)
        text = open(out).read()
    assert "suse.ai.component.name=kubeflow-pipelines" in text
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in text
