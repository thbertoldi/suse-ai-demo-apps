"""Check the rendered installation contract rather than template text."""
from pathlib import Path
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "helm/suse-ai-demo"


def render(*args):
    result = subprocess.run(["helm", "template", "demo", str(CHART), "--namespace", "demo-tests", *args],
                            capture_output=True, text=True, check=True)
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def container(doc):
    spec = doc["spec"]
    if doc["kind"] == "CronJob":
        spec = spec["jobTemplate"]["spec"]
    return spec["template"]["spec"]["containers"][0]


class ChartTests(unittest.TestCase):
    def test_default_install_needs_no_kubeflow_namespace(self):
        docs = render()
        self.assertFalse(any(doc["kind"] in ("Role", "RoleBinding", "CronJob") for doc in docs))
        workloads = [doc for doc in docs if doc["kind"] == "Deployment"]
        self.assertEqual({doc["metadata"]["name"] for doc in workloads},
                         {"gateway", "rag-service", "llm-service", "agent-service", "traffic-gen"})
        version = yaml.safe_load((CHART / "Chart.yaml").read_text())["appVersion"]
        for doc in workloads:
            c = container(doc)
            self.assertTrue(c["image"].endswith(":" + version))
            env = {item["name"]: item for item in c["env"]}
            self.assertIn("service.namespace=demo-tests", env["OTEL_RESOURCE_ATTRIBUTES"]["value"])
            self.assertIn("service.instance.id=$(POD_UID)", env["OTEL_RESOURCE_ATTRIBUTES"]["value"])
            self.assertIn("suse.ai.demo=true", env["OTEL_RESOURCE_ATTRIBUTES"]["value"])
            self.assertEqual(env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"], "metadata.uid")

    def test_native_products_have_images_secrets_tls_and_native_identity(self):
        docs = render("-f", str(ROOT / "examples/native-products.yaml"))
        workloads = {doc["metadata"]["name"]: doc for doc in docs if doc["kind"] in ("Deployment", "CronJob")}
        self.assertTrue({"rag-qdrant", "rag-opensearch", "product-traffic", "pytorch-training"} <= workloads.keys())
        for name in ("rag-qdrant", "rag-opensearch", "product-traffic", "pytorch-training"):
            env = {v["name"]: v for v in container(workloads[name])["env"]}
            self.assertEqual(env["OTEL_SERVICE_NAME"]["value"], name)
            self.assertIn("service.instance.id=$(POD_UID)", env["OTEL_RESOURCE_ATTRIBUTES"]["value"])
        search = container(workloads["rag-opensearch"])
        env = {v["name"]: v for v in search["env"]}
        self.assertEqual(env["VECTOR_DB_PASSWORD"]["valueFrom"]["secretKeyRef"]["name"], "opensearch-auth")
        self.assertEqual(env["OPENSEARCH_INDEX_REPLICAS"]["value"], "0")
        self.assertEqual(env["OPENSEARCH_CA_CERT"]["value"], "/etc/opensearch-ca/ca.crt")
        self.assertEqual(search["volumeMounts"][0]["mountPath"], "/etc/opensearch-ca")
        self.assertEqual(container(workloads["product-traffic"])["command"], ["python", "-m", "app.product_traffic"])
        self.assertEqual(workloads["pytorch-training"]["spec"]["concurrencyPolicy"], "Forbid")

    def test_authenticated_rag_and_agent_and_digest_override(self):
        digest = "sha256:" + "a" * 64
        docs = render("--set", "ragService.llm.existingSecret=provider-key",
                      "--set", "agentService.llm.existingSecret=provider-key",
                      "--set", "ragService.image.digest=" + digest)
        for doc in docs:
            if doc["kind"] != "Deployment" or doc["metadata"]["name"] not in ("rag-service", "agent-service"):
                continue
            c = container(doc)
            key = next(env for env in c["env"] if env["name"] == "LLM_API_KEY")
            self.assertEqual(key["valueFrom"]["secretKeyRef"]["name"], "provider-key")
            if doc["metadata"]["name"] == "rag-service":
                self.assertTrue(c["image"].endswith("@" + digest))

    def test_optional_kubeflow_and_bounded_product_job(self):
        docs = render("-f", str(ROOT / "examples/kubeflow.yaml"))
        self.assertEqual(sum(doc["kind"] in ("Role", "RoleBinding") for doc in docs), 2)
        docs = render("-f", str(ROOT / "examples/native-products.yaml"), "--set", "productTraffic.maxRounds=1")
        product = next(doc for doc in docs if doc["metadata"]["name"] == "product-traffic")
        self.assertEqual(product["kind"], "Job")
        self.assertEqual(product["spec"]["backoffLimit"], 0)
        self.assertEqual(product["spec"]["template"]["spec"]["restartPolicy"], "Never")

    def test_incomplete_product_configuration_fails_early(self):
        with self.assertRaises(subprocess.CalledProcessError):
            render("--set", "productTraffic.enabled=true")
        with self.assertRaises(subprocess.CalledProcessError):
            render("--set", "training.enabled=true")
