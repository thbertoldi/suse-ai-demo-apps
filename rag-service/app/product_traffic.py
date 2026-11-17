"""Exercise selected native products using the dependencies in the RAG image."""

import logging
import os
import signal
import threading
import time
from urllib.parse import urlparse

import grpc
import requests
from opentelemetry import metrics, trace
from opentelemetry.propagate import inject

from .generated import demo_pb2, demo_pb2_grpc
from .llm_client import chat_completion
from .otel_setup import setup_otel, shutdown_otel

logger = logging.getLogger(__name__)
SCENARIOS = ("vllm", "litellm-vllm", "qdrant", "opensearch", "mlflow", "mcpo")


class ProductTraffic:
    def __init__(self, config):
        self.config = config
        self.names = tuple(name.strip() for name in config.get("PRODUCT_SCENARIOS", "").split(",") if name.strip())
        if not self.names or any(name not in SCENARIOS for name in self.names):
            raise ValueError("PRODUCT_SCENARIOS must select from " + ",".join(SCENARIOS))
        required = {
            "vllm": ("VLLM_BASE_URL", "VLLM_MODEL"),
            "litellm-vllm": ("LITELLM_BASE_URL", "LITELLM_MODEL", "LITELLM_API_KEY"),
            "qdrant": ("QDRANT_RAG_ADDR",),
            "opensearch": ("OPENSEARCH_RAG_ADDR",),
            "mlflow": ("MLFLOW_URL",),
            "mcpo": ("MCPO_URL",),
        }
        for name in self.names:
            for key in required[name]:
                if not config.get(key):
                    raise ValueError(f"{key} is required for {name}")
        self.tracer = trace.get_tracer("suse.ai.demo.products")
        meter = metrics.get_meter("suse.ai.demo.products")
        self.runs = meter.create_counter("suse.ai.demo.scenario.runs", unit="{run}")
        self.duration = meter.create_histogram("suse.ai.demo.scenario.duration", unit="s")
        self.session = requests.Session()
        self.channels = {}
        self.experiment = None

    def http(self, base, path, payload=None, *, method="POST", params=None, token=""):
        endpoint = urlparse(base)
        with self.tracer.start_as_current_span(path, kind=trace.SpanKind.CLIENT, attributes={
            "server.address": endpoint.hostname or "",
            "server.port": endpoint.port or (443 if endpoint.scheme == "https" else 80),
            "http.request.method": method,
            "url.path": path,
        }) as span:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            inject(headers)
            response = self.session.request(method, base.rstrip("/") + path,
                                            json=payload, params=params, headers=headers, timeout=30)
            span.set_attribute("http.response.status_code", response.status_code)
            response.raise_for_status()
            return response.json()

    def mlflow_run(self):
        base = self.config["MLFLOW_URL"]
        token = self.config.get("MLFLOW_TRACKING_TOKEN", "")

        def api(path, payload=None, **kwargs):
            return self.http(base, "/api/2.0/mlflow/" + path, payload, token=token, **kwargs)

        if self.experiment is None:
            name = self.config.get("MLFLOW_EXPERIMENT", "native-product-traffic")
            try:
                result = api("experiments/get-by-name", method="GET", params={"experiment_name": name})
                self.experiment = result["experiment"]["experiment_id"]
            except requests.HTTPError as error:
                if error.response.status_code != 404:
                    raise
                self.experiment = api("experiments/create", {"name": name})["experiment_id"]
        started = time.monotonic()
        info = api("runs/create", {"experiment_id": self.experiment, "run_name": "api-roundtrip"})["run"]["info"]
        run_id = info["run_id"]
        try:
            now = int(time.time() * 1000)
            api("runs/log-metric", {"run_id": run_id, "key": "create_latency_seconds",
                                   "value": time.monotonic() - started, "timestamp": now, "step": 0})
            api("runs/update", {"run_id": run_id, "status": "FINISHED", "end_time": now})
        except Exception:
            try:
                api("runs/update", {"run_id": run_id, "status": "FAILED", "end_time": int(time.time() * 1000)})
            except Exception:
                logger.warning("Could not mark MLflow run failed")
            raise
        return {"run_id": run_id}

    def retrieve(self, name):
        if name not in self.channels:
            self.channels[name] = grpc.insecure_channel(self.config[name.upper() + "_RAG_ADDR"])
        result = demo_pb2_grpc.RAGServiceStub(self.channels[name]).Retrieve(
            demo_pb2.RetrieveRequest(query="What is a Kubernetes pod?", top_k=3), timeout=90)
        if not result.sources:
            raise RuntimeError(f"{name} returned no documents")
        return {"sources": len(result.sources)}

    def operation(self, name):
        if name in ("qdrant", "opensearch"):
            return self.retrieve(name)
        if name == "mlflow":
            return self.mlflow_run()
        if name == "mcpo":
            return self.http(self.config["MCPO_URL"], "/time/get_current_time", {"timezone": "UTC"},
                             token=self.config.get("MCPO_API_KEY", ""))
        prefix, provider = ("VLLM", "vllm") if name == "vllm" else ("LITELLM", "litellm")
        return chat_completion(
            self.config[prefix + "_BASE_URL"], self.config[prefix + "_MODEL"], provider,
            [{"role": "user", "content": "Explain a Kubernetes pod in one sentence."}],
            max_tokens=40, api_key=self.config.get(prefix + "_API_KEY", ""))

    def run(self, name):
        started = time.monotonic()
        outcome = "success"
        with self.tracer.start_as_current_span("scenario " + name) as span:
            try:
                self.operation(name)
                logger.info("scenario=%s outcome=success", name)
            except Exception as error:
                outcome = "error"
                span.record_exception(error)
                span.set_status(trace.StatusCode.ERROR, type(error).__name__)
                logger.warning("scenario=%s outcome=error error=%s", name, type(error).__name__)
        attributes = {"scenario": name, "outcome": outcome}
        self.runs.add(1, attributes)
        self.duration.record(time.monotonic() - started, attributes)
        return outcome == "success"

    def close(self):
        for channel in self.channels.values():
            channel.close()
        self.session.close()


def main():
    logging.basicConfig(level=logging.INFO)
    interval = float(os.environ.get("PRODUCT_INTERVAL_SECONDS", "15"))
    rounds = int(os.environ.get("PRODUCT_MAX_ROUNDS", "0"))
    if interval < 0 or rounds < 0:
        raise ValueError("Interval and maximum rounds must be nonnegative")
    providers = setup_otel(os.environ.get("OTEL_SERVICE_NAME", "product-traffic"))
    traffic = None
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    failed = False
    try:
        traffic = ProductTraffic(os.environ)
        completed = 0
        while not stop.is_set() and (rounds == 0 or completed < rounds):
            for name in traffic.names:
                if stop.is_set():
                    break
                failed = not traffic.run(name) or failed
                stop.wait(interval)
            completed += 1
    finally:
        if traffic is not None:
            traffic.close()
        shutdown_otel(*providers)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
