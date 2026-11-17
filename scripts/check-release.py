"""Reject inconsistent release metadata before building or publishing images."""
import os
from pathlib import Path
import re

import yaml

root = Path(__file__).resolve().parents[1]
chart = yaml.safe_load((root / "helm/suse-ai-demo/Chart.yaml").read_text())
version = chart["version"]
assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version), version
assert chart["appVersion"] == version, "Chart and application versions must match"
if os.environ.get("RELEASE_TAG"):
    assert os.environ["RELEASE_TAG"] == "v" + version, "Tag does not match chart version"
values = yaml.safe_load((root / "helm/suse-ai-demo/values.yaml").read_text())
for service in ("gateway", "ragService", "llmService", "agentService", "trafficGen", "training"):
    assert values[service]["image"]["tag"] in ("", version), f"Unpinned {service} image"
assert f"suse-ai-demo-iris-pipeline:{version}" in (root / "demo/kubeflow/pipeline.py").read_text()
assert (root / "docs/releases" / f"v{version}.md").is_file(), "Release notes are missing"
print(f"Release metadata agrees on {version}")
