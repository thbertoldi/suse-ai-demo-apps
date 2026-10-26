"""Database traces identify the destination without exporting connection secrets."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vector_base", ROOT / "rag-service/app/vectordb/base.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Store(module.VectorStore):
    def search(self, *args): pass
    def upsert(self, *args): pass
    def health(self): pass
    def collection_exists(self): pass
    def create_collection(self, *args): pass
    def count(self): pass


class DatabaseEndpointsTest(unittest.TestCase):
    def test_explicit_endpoint_and_operation_without_url_secrets(self):
        for product, port in (("milvus", 19530), ("qdrant", 6333), ("opensearch", 9200)):
            store = Store(f"https://user:password@database.example:{port}/private?token=secret", "documents")
            for operation in ("search", "upsert"):
                attrs = store.span_attributes(product, operation)
                self.assertEqual(attrs["db.system.name"], product)
                self.assertEqual(attrs["db.operation.name"], operation)
                self.assertEqual(attrs["server.address"], "database.example")
                self.assertEqual(attrs["server.port"], port)
                self.assertNotIn("secret", str(attrs))
                self.assertNotIn("password", str(attrs))
                self.assertNotIn("private", str(attrs))

    def test_ipv6_and_unknown_port(self):
        self.assertEqual(Store("http://[2001:db8::1]:19530", "docs").span_attributes("milvus", "search")["server.address"],
                         "2001:db8::1")
        attrs = Store("https://database.example", "docs").span_attributes("milvus", "search")
        self.assertNotIn("server.port", attrs, "Do not guess a transport port from a product or URL scheme")
        self.assertEqual(Store("database.example:19530", "docs").span_attributes("milvus", "search")["server.port"],
                         19530)

    def test_each_store_emits_client_spans_for_search_and_upsert(self):
        # Exercise the real wrapper methods and SDK. Database responses are
        # stubbed; live Milvus retrieval is covered by the deployed demo.
        package = ModuleType("database_test")
        package.__path__ = []
        models = SimpleNamespace(Distance=Mock(), PointStruct=Mock(), VectorParams=Mock())
        modules = {"database_test": package, "database_test.base": module,
                   "pymilvus": SimpleNamespace(MilvusClient=Mock()),
                   "qdrant_client": SimpleNamespace(QdrantClient=Mock(), models=models),
                   "qdrant_client.models": models,
                   "opensearchpy": SimpleNamespace(OpenSearch=Mock())}
        for product, class_name in (("milvus", "MilvusVectorStore"), ("qdrant", "QdrantVectorStore"),
                                    ("opensearch", "OpenSearchVectorStore")):
            with self.subTest(product=product):
                provider, exporter = TracerProvider(), InMemorySpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(exporter))
                spec = importlib.util.spec_from_file_location("database_test." + product,
                    ROOT / "rag-service/app/vectordb" / (product + ".py"))
                implementation = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, modules), patch.object(trace, "get_tracer", provider.get_tracer):
                    spec.loader.exec_module(implementation)
                store = getattr(implementation, class_name).__new__(getattr(implementation, class_name))
                module.VectorStore.__init__(store, "http://database.example:19530", "docs")
                store._client = Mock()
                store._client.search.return_value = [[]] if product == "milvus" else {"hits": {"hits": []}}
                store._client.query_points.return_value = SimpleNamespace(points=[])
                try:
                    self.assertEqual(store.search([0.1, 0.2]), [])
                    store.upsert([module.Document(id="1", content="test", embedding=[0.1, 0.2])])
                    spans = exporter.get_finished_spans()
                    self.assertEqual(len(spans), 2)
                    self.assertEqual({s.attributes["db.operation.name"] for s in spans}, {"search", "upsert"})
                    for span in spans:
                        self.assertEqual(span.kind, trace.SpanKind.CLIENT)
                        self.assertEqual(span.attributes["db.system.name"], product)
                        self.assertEqual(span.attributes["server.address"], "database.example")
                        self.assertEqual(span.attributes["server.port"], 19530)
                    upsert = next(s for s in spans if s.attributes["db.operation.name"] == "upsert")
                    self.assertEqual(upsert.attributes["db.operation.batch.size"], 1)
                    self.assertNotIn("db.operation.batch_size", upsert.attributes)
                    exporter.clear()
                    store._client.search.side_effect = RuntimeError("database unavailable")
                    store._client.query_points.side_effect = RuntimeError("database unavailable")
                    with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                        store.search([0.1, 0.2])
                    failed = exporter.get_finished_spans()[0]
                    self.assertEqual(failed.status.status_code, trace.StatusCode.ERROR)
                    self.assertEqual(failed.attributes["error.type"], "RuntimeError")
                    self.assertEqual(failed.attributes["server.address"], "database.example")
                    self.assertTrue(any(event.name == "exception" for event in failed.events))
                finally:
                    provider.shutdown()


if __name__ == "__main__":
    unittest.main()
