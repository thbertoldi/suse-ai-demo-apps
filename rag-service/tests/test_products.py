import http.server
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import grpc
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import product_traffic
from app.generated import demo_pb2, demo_pb2_grpc


class Handler(http.server.BaseHTTPRequestHandler):
    calls = []
    fail_metrics = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.reply(404, {"error_code": "RESOURCE_DOES_NOT_EXIST"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self.calls.append((self.path, body, self.headers.get("Authorization"), self.headers.get("traceparent")))
        if self.path.endswith("experiments/create"):
            result = {"experiment_id": "1"}
        elif self.path.endswith("runs/create"):
            result = {"run": {"info": {"run_id": "test-run"}}}
        elif self.path.endswith("runs/log-metric") and self.fail_metrics:
            return self.reply(503, {"error": "fixture"})
        elif self.path.endswith("chat/completions"):
            result = {"model": body["model"], "choices": [{"message": {"content": "fixture"}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
        else:
            result = {"ok": True}
        self.reply(200, result)

    def reply(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Retrieval(demo_pb2_grpc.RAGServiceServicer):
    def Retrieve(self, request, context):
        return demo_pb2.RetrieveResponse(answer="fixture", sources=["doc-1"])


class ProductTests(unittest.TestCase):
    def test_rag_keeps_embedding_provider_separate_from_generation_proxy(self):
        from app import grpc_server
        store = Mock()
        store.search.return_value = []
        with patch.dict(os.environ, {"LLM_PROVIDER": "litellm", "EMBEDDING_PROVIDER": "ollama"}), \
                patch.object(grpc_server, "embed", return_value=[1.0]) as embedding, \
                patch.object(grpc_server, "chat_completion", return_value={"choices": [{"message": {"content": "fixture"}}]}) as chat:
            service = grpc_server.RAGServiceServicer(store)
            result = service.Retrieve(demo_pb2.RetrieveRequest(query="fixture", top_k=3), Mock())
        self.assertEqual(result.answer, "fixture")
        self.assertEqual(embedding.call_args.kwargs["provider"], "ollama")
        self.assertEqual(chat.call_args.kwargs["provider"], "litellm")

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.grpc_server = grpc.server(ThreadPoolExecutor(max_workers=2))
        demo_pb2_grpc.add_RAGServiceServicer_to_server(Retrieval(), cls.grpc_server)
        cls.grpc_port = cls.grpc_server.add_insecure_port("127.0.0.1:0")
        cls.grpc_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.grpc_server.stop(0).wait()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.calls = []
        Handler.fail_metrics = False

    def test_all_selected_scenarios_use_real_http_and_grpc(self):
        provider, exporter = TracerProvider(), InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        config = {
            "PRODUCT_SCENARIOS": ",".join(product_traffic.SCENARIOS),
            "VLLM_BASE_URL": self.base + "/v1", "VLLM_MODEL": "direct",
            "LITELLM_BASE_URL": self.base + "/v1", "LITELLM_MODEL": "proxy",
            "LITELLM_API_KEY": "test-key", "MLFLOW_URL": self.base, "MCPO_URL": self.base,
            "QDRANT_RAG_ADDR": f"127.0.0.1:{self.grpc_port}",
            "OPENSEARCH_RAG_ADDR": f"127.0.0.1:{self.grpc_port}",
        }
        with patch.object(product_traffic.trace, "get_tracer", provider.get_tracer):
            traffic = product_traffic.ProductTraffic(config)
        try:
            self.assertTrue(all(traffic.run(name) for name in traffic.names))
            chats = [call for call in Handler.calls if call[0].endswith("chat/completions")]
            self.assertEqual([(c[1]["model"], c[2]) for c in chats], [("direct", None), ("proxy", "Bearer test-key")])
            updates = [c[1]["status"] for c in Handler.calls if c[0].endswith("runs/update")]
            self.assertEqual(updates, ["FINISHED"])
            mcpo = next(c for c in Handler.calls if c[0] == "/time/get_current_time")
            self.assertEqual(mcpo[1], {"timezone": "UTC"})
            self.assertTrue(mcpo[3].startswith("00-"))
            scenarios = [s for s in exporter.get_finished_spans() if s.name.startswith("scenario ")]
            self.assertEqual(len(scenarios), 6)
        finally:
            traffic.close()
            provider.shutdown()

    def test_mlflow_failure_is_reported_and_run_is_closed(self):
        Handler.fail_metrics = True
        traffic = product_traffic.ProductTraffic({"PRODUCT_SCENARIOS": "mlflow", "MLFLOW_URL": self.base})
        try:
            self.assertFalse(traffic.run("mlflow"))
            updates = [c[1]["status"] for c in Handler.calls if c[0].endswith("runs/update")]
            self.assertEqual(updates, ["FAILED"])
        finally:
            traffic.close()

    def test_only_selected_products_need_configuration(self):
        traffic = product_traffic.ProductTraffic({"PRODUCT_SCENARIOS": "mcpo", "MCPO_URL": self.base})
        try:
            self.assertTrue(traffic.run("mcpo"))
        finally:
            traffic.close()
        for config in ({}, {"PRODUCT_SCENARIOS": "unknown"}, {"PRODUCT_SCENARIOS": "litellm-vllm"}):
            with self.assertRaises(ValueError):
                product_traffic.ProductTraffic(config)
