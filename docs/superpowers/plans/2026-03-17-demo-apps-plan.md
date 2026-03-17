# SUSE AI Demo Apps Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG pipeline of 4 instrumented services (Go gateway, Python RAG, Python LLM, traffic generator) that produce OpenTelemetry traces and metrics following GenAI semantic conventions.

**Architecture:** A Go gateway receives gRPC requests and routes them to either a Python RAG service (which queries Qdrant + Ollama) or a Python LLM service (which queries vLLM). A traffic generator loops predefined queries. All services export telemetry via OTLP to a collector.

**Tech Stack:** Go 1.22+, Python 3.12+, gRPC/protobuf, OpenTelemetry (Go SDK + Python SDK), Qdrant client, requests library, Docker, Helm 3, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-03-17-demo-apps-design.md`

---

### Task 1: Proto Definition and Code Generation Setup

**Files:**
- Create: `proto/demo.proto`
- Create: `proto/Makefile`

This task creates the shared protobuf definition and the tooling to generate Go and Python stubs.

- [ ] **Step 1: Create `proto/demo.proto`**

```protobuf
syntax = "proto3";
package demo;
option go_package = "github.com/suse/suse-ai-demo-apps/gateway/pb";

service DemoService {
  rpc Query(QueryRequest) returns (QueryResponse);
  rpc Chat(ChatRequest) returns (ChatResponse);
}

service RAGService {
  rpc Retrieve(RetrieveRequest) returns (RetrieveResponse);
}

service LLMService {
  rpc Generate(GenerateRequest) returns (GenerateResponse);
}

message QueryRequest {
  string query = 1;
  int32 top_k = 2;
}

message QueryResponse {
  string answer = 1;
  repeated string sources = 2;
  string model = 3;
}

message ChatRequest {
  string message = 1;
}

message ChatResponse {
  string reply = 1;
  string model = 2;
}

message RetrieveRequest {
  string query = 1;
  int32 top_k = 2;
}

message RetrieveResponse {
  string answer = 1;
  repeated string sources = 2;
  string model = 3;
}

message GenerateRequest {
  string prompt = 1;
}

message GenerateResponse {
  string text = 1;
  string model = 2;
}
```

- [ ] **Step 2: Create `proto/Makefile` for code generation**

```makefile
.PHONY: all go python clean

all: go python

go:
	protoc --go_out=../gateway --go-grpc_out=../gateway \
		--go_opt=paths=source_relative --go-grpc_opt=paths=source_relative \
		demo.proto

python:
	python -m grpc_tools.protoc -I. \
		--python_out=../rag-service/app/generated --grpc_python_out=../rag-service/app/generated \
		demo.proto
	python -m grpc_tools.protoc -I. \
		--python_out=../llm-service/app/generated --grpc_python_out=../llm-service/app/generated \
		demo.proto
	python -m grpc_tools.protoc -I. \
		--python_out=../traffic-gen/generated --grpc_python_out=../traffic-gen/generated \
		demo.proto
	# Fix imports in generated files
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../rag-service/app/generated/demo_pb2_grpc.py
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../llm-service/app/generated/demo_pb2_grpc.py
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../traffic-gen/generated/demo_pb2_grpc.py

clean:
	rm -f ../gateway/pb/*.go
	rm -f ../rag-service/app/generated/demo_pb2*.py
	rm -f ../llm-service/app/generated/demo_pb2*.py
	rm -f ../traffic-gen/generated/demo_pb2*.py
```

- [ ] **Step 3: Create output directories with `__init__.py`**

Create these empty directories with `__init__.py` files:
- `rag-service/app/generated/__init__.py`
- `llm-service/app/generated/__init__.py`
- `traffic-gen/generated/__init__.py`
- `gateway/pb/` (empty dir, Go code will go here)

- [ ] **Step 4: Generate code**

Run: `cd /home/thbertoldi/suse/suse-ai-demo-apps/proto && make all`

Verify: `ls ../gateway/pb/demo*.go` shows `demo.pb.go` and `demo_grpc.pb.go`
Verify: `ls ../rag-service/app/generated/demo_pb2*.py` shows `demo_pb2.py` and `demo_pb2_grpc.py`

- [ ] **Step 5: Commit**

```bash
git add proto/ gateway/pb/ rag-service/app/generated/ llm-service/app/generated/ traffic-gen/generated/
git commit -m "feat: add proto definition and generated code for all services"
```

---

### Task 2: Python Shared OTel Setup Module (for RAG and LLM services)

**Files:**
- Create: `rag-service/app/otel_setup.py`
- Create: `llm-service/app/otel_setup.py`

Both Python services need the same OTel bootstrap. We write it once and copy it (they are separate deployables, not a shared library).

- [ ] **Step 1: Create `rag-service/app/otel_setup.py`**

```python
import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer, GrpcInstrumentorClient
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator


def setup_otel(service_name: str) -> tuple[TracerProvider, MeterProvider]:
    resource = Resource.create({"service.name": service_name})

    # Traces
    trace_exporter = OTLPSpanExporter(insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_exporter = OTLPMetricExporter(insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Propagation
    set_global_textmap(CompositeHTTPPropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    # Auto-instrument gRPC and HTTP requests
    GrpcInstrumentorServer().instrument()
    GrpcInstrumentorClient().instrument()
    RequestsInstrumentor().instrument()

    return tracer_provider, meter_provider


def shutdown_otel(tracer_provider: TracerProvider, meter_provider: MeterProvider):
    tracer_provider.shutdown()
    meter_provider.shutdown()
```

- [ ] **Step 2: Copy to `llm-service/app/otel_setup.py`**

The file is identical. Copy `rag-service/app/otel_setup.py` to `llm-service/app/otel_setup.py`.

- [ ] **Step 3: Create `rag-service/app/__init__.py` and `llm-service/app/__init__.py`**

Both are empty files.

- [ ] **Step 4: Commit**

```bash
git add rag-service/app/otel_setup.py rag-service/app/__init__.py \
       llm-service/app/otel_setup.py llm-service/app/__init__.py
git commit -m "feat: add OpenTelemetry setup module for Python services"
```

---

### Task 3: LLM Client with GenAI Instrumentation

**Files:**
- Create: `rag-service/app/llm_client.py`
- Create: `llm-service/app/llm_client.py`

The LLM client calls `/v1/chat/completions` and produces GenAI spans and metrics. Both services use the same client code.

- [ ] **Step 1: Create `rag-service/app/llm_client.py`**

```python
import json
import os
import time
import requests
from opentelemetry import trace, metrics

tracer = trace.get_tracer("gen_ai")
meter = metrics.get_meter("gen_ai")

token_usage_histogram = meter.create_histogram(
    name="gen_ai.client.token.usage",
    description="Token usage per GenAI call",
    unit="{token}",
)

operation_duration_histogram = meter.create_histogram(
    name="gen_ai.client.operation.duration",
    description="Duration of GenAI operations",
    unit="s",
)

ENABLE_CONTENT_EVENTS = os.environ.get("ENABLE_OTEL_CONTENT_EVENTS", "false").lower() == "true"


def chat_completion(
    base_url: str,
    model: str,
    provider: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    with tracer.start_as_current_span(
        f"chat {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
            "gen_ai.request.max_tokens": max_tokens,
            "gen_ai.request.temperature": temperature,
        },
    ) as span:
        if ENABLE_CONTENT_EVENTS:
            span.add_event("gen_ai.input.messages", attributes={
                "gen_ai.input.messages": json.dumps(messages),
            })

        start_time = time.monotonic()
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            error_type = type(e).__name__
            if hasattr(e, "response") and e.response is not None:
                error_type = str(e.response.status_code)
            span.set_attribute("error.type", error_type)
            raise
        finally:
            duration = time.monotonic() - start_time
            common_attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            }
            operation_duration_histogram.record(duration, attributes=common_attrs)

        response_model = data.get("model", model)
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        choices = data.get("choices", [])
        finish_reasons = [c.get("finish_reason", "") for c in choices]
        response_id = data.get("id", "")

        span.set_attribute("gen_ai.response.model", response_model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        span.set_attribute("gen_ai.response.id", response_id)

        token_usage_histogram.record(input_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "input",
        })
        token_usage_histogram.record(output_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "output",
        })

        if ENABLE_CONTENT_EVENTS and choices:
            output_messages = [
                {"role": "assistant", "content": c.get("message", {}).get("content", "")}
                for c in choices
            ]
            span.add_event("gen_ai.output.messages", attributes={
                "gen_ai.output.messages": json.dumps(output_messages),
            })

        return data
```

- [ ] **Step 2: Copy to `llm-service/app/llm_client.py`**

The file is identical. Copy it.

- [ ] **Step 3: Commit**

```bash
git add rag-service/app/llm_client.py llm-service/app/llm_client.py
git commit -m "feat: add LLM client with GenAI semantic convention instrumentation"
```

---

### Task 4: Embedding Client with GenAI Instrumentation

**Files:**
- Create: `rag-service/app/embedding_client.py`

- [ ] **Step 1: Create `rag-service/app/embedding_client.py`**

```python
import time
import requests
from opentelemetry import trace, metrics

tracer = trace.get_tracer("gen_ai")
meter = metrics.get_meter("gen_ai")

token_usage_histogram = meter.create_histogram(
    name="gen_ai.client.token.usage",
    description="Token usage per GenAI call",
    unit="{token}",
)

operation_duration_histogram = meter.create_histogram(
    name="gen_ai.client.operation.duration",
    description="Duration of GenAI operations",
    unit="s",
)


def embed(
    base_url: str,
    model: str,
    provider: str,
    text: str,
) -> list[float]:
    with tracer.start_as_current_span(
        f"embed {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "embed",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
        },
    ) as span:
        start_time = time.monotonic()
        try:
            response = requests.post(
                f"{base_url}/embeddings",
                json={"model": model, "input": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            error_type = type(e).__name__
            if hasattr(e, "response") and e.response is not None:
                error_type = str(e.response.status_code)
            span.set_attribute("error.type", error_type)
            raise
        finally:
            duration = time.monotonic() - start_time
            common_attrs = {
                "gen_ai.operation.name": "embed",
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            }
            operation_duration_histogram.record(duration, attributes=common_attrs)

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

        token_usage_histogram.record(input_tokens, attributes={
            **common_attrs, "gen_ai.token.type": "input",
        })

        embedding = data["data"][0]["embedding"]
        return embedding
```

- [ ] **Step 2: Commit**

```bash
git add rag-service/app/embedding_client.py
git commit -m "feat: add embedding client with GenAI semantic convention instrumentation"
```

---

### Task 5: Vector DB Repository Pattern (Qdrant)

**Files:**
- Create: `rag-service/app/vectordb/__init__.py`
- Create: `rag-service/app/vectordb/base.py`
- Create: `rag-service/app/vectordb/qdrant.py`
- Create: `rag-service/app/vectordb/factory.py`

- [ ] **Step 1: Create `rag-service/app/vectordb/__init__.py`**

```python
from .base import VectorStore, Document
from .factory import create_vector_store

__all__ = ["VectorStore", "Document", "create_vector_store"]
```

- [ ] **Step 2: Create `rag-service/app/vectordb/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class VectorStore(ABC):
    def __init__(self, url: str, collection_name: str):
        self._url = url
        self._collection_name = collection_name

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        ...

    @abstractmethod
    def upsert(self, documents: list[Document]) -> None:
        ...

    @abstractmethod
    def health(self) -> bool:
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        ...

    @abstractmethod
    def create_collection(self, vector_size: int) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
```

- [ ] **Step 3: Create `rag-service/app/vectordb/qdrant.py`**

```python
from opentelemetry import trace
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .base import Document, VectorStore

tracer = trace.get_tracer("vectordb")


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, collection_name: str):
        super().__init__(url, collection_name)
        self._client = QdrantClient(url=url)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        with tracer.start_as_current_span(
            f"search {self._collection_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system": "qdrant",
                "db.operation.name": "search",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                results = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True,
                )
                docs = []
                for point in results.points:
                    payload = point.payload or {}
                    docs.append(Document(
                        id=str(point.id),
                        content=payload.get("content", ""),
                        metadata=payload.get("metadata", {}),
                        score=point.score,
                    ))
                span.set_attribute("db.response.returned_rows", len(docs))
                return docs
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def upsert(self, documents: list[Document]) -> None:
        with tracer.start_as_current_span(
            f"upsert {self._collection_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system": "qdrant",
                "db.operation.name": "upsert",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                points = [
                    PointStruct(
                        id=idx,
                        vector=doc.embedding,
                        payload={"content": doc.content, "metadata": doc.metadata},
                    )
                    for idx, doc in enumerate(documents)
                ]
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
                span.set_attribute("db.operation.batch_size", len(documents))
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def health(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
            return True
        except Exception:
            return False

    def create_collection(self, vector_size: int) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count
```

- [ ] **Step 4: Create `rag-service/app/vectordb/factory.py`**

```python
from .base import VectorStore
from .qdrant import QdrantVectorStore

_REGISTRY: dict[str, type[VectorStore]] = {
    "qdrant": QdrantVectorStore,
}


def create_vector_store(db_type: str, url: str, collection_name: str) -> VectorStore:
    cls = _REGISTRY.get(db_type)
    if cls is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown vector DB type '{db_type}'. Supported: {supported}")
    return cls(url=url, collection_name=collection_name)
```

- [ ] **Step 5: Commit**

```bash
git add rag-service/app/vectordb/
git commit -m "feat: add vector DB repository pattern with Qdrant implementation"
```

---

### Task 6: Seed Data for RAG Service

**Files:**
- Create: `rag-service/app/seed_data.py`

- [ ] **Step 1: Create `rag-service/app/seed_data.py`**

```python
import logging
from app.embedding_client import embed
from app.vectordb.base import Document, VectorStore

logger = logging.getLogger(__name__)

SEED_DOCUMENTS = [
    {
        "content": "A Kubernetes pod is the smallest deployable unit in Kubernetes. A pod represents a single instance of a running process in your cluster. Pods contain one or more containers, such as Docker containers. When a pod runs multiple containers, the containers are managed as a single entity and share the pod's resources.",
        "metadata": {"topic": "kubernetes", "subtopic": "pods"},
    },
    {
        "content": "A Kubernetes Deployment provides declarative updates for Pods and ReplicaSets. You describe a desired state in a Deployment, and the Deployment controller changes the actual state to the desired state at a controlled rate. You can define Deployments to create new ReplicaSets, or to remove existing Deployments and adopt all their resources with new Deployments.",
        "metadata": {"topic": "kubernetes", "subtopic": "deployments"},
    },
    {
        "content": "A Kubernetes Service is an abstraction which defines a logical set of Pods and a policy by which to access them. Services enable loose coupling between dependent Pods. A Service is defined using YAML or JSON, like all Kubernetes objects.",
        "metadata": {"topic": "kubernetes", "subtopic": "services"},
    },
    {
        "content": "Linux containers are a technology that allows you to package and isolate applications with their entire runtime environment. This makes it easy to move the contained application between environments while retaining full functionality. Containers share the host OS kernel and therefore do not require an OS per application.",
        "metadata": {"topic": "containers", "subtopic": "basics"},
    },
    {
        "content": "A container runtime is the software responsible for running containers. It manages the complete lifecycle of containers including image transfer, storage, execution, supervision, and networking. Common container runtimes include containerd, CRI-O, and Docker Engine.",
        "metadata": {"topic": "containers", "subtopic": "runtime"},
    },
    {
        "content": "Container images are lightweight, standalone, executable packages that include everything needed to run a piece of software, including the code, runtime, system tools, libraries, and settings. Images are built from Dockerfiles and stored in container registries.",
        "metadata": {"topic": "containers", "subtopic": "images"},
    },
    {
        "content": "OpenTelemetry is a collection of APIs, SDKs, and tools used to instrument, generate, collect, and export telemetry data (metrics, logs, and traces) to help you analyze your software's performance and behavior. It is a CNCF project and provides a vendor-neutral specification.",
        "metadata": {"topic": "observability", "subtopic": "opentelemetry"},
    },
    {
        "content": "Distributed tracing is a method used to profile and monitor applications built using a microservices architecture. It helps pinpoint where failures occur and what causes poor performance by tracking requests as they flow through distributed systems.",
        "metadata": {"topic": "observability", "subtopic": "tracing"},
    },
]


def seed_if_needed(
    store: VectorStore,
    embedding_base_url: str,
    embedding_model: str,
    provider: str,
) -> None:
    if store.collection_exists() and store.count() > 0:
        logger.info("Collection already populated, skipping seed")
        return

    logger.info("Seeding vector database with sample documents...")

    # Generate embeddings for seed documents
    documents = []
    for i, doc_data in enumerate(SEED_DOCUMENTS):
        embedding = embed(
            base_url=embedding_base_url,
            model=embedding_model,
            provider=provider,
            text=doc_data["content"],
        )
        documents.append(Document(
            id=str(i),
            content=doc_data["content"],
            embedding=embedding,
            metadata=doc_data["metadata"],
        ))

    # Create collection if needed (use first embedding's dimension)
    if not store.collection_exists():
        vector_size = len(documents[0].embedding)
        store.create_collection(vector_size)
        logger.info(f"Created collection with vector_size={vector_size}")

    store.upsert(documents)
    logger.info(f"Seeded {len(documents)} documents")
```

- [ ] **Step 2: Commit**

```bash
git add rag-service/app/seed_data.py
git commit -m "feat: add seed data module for RAG service vector DB bootstrap"
```

---

### Task 7: RAG Service gRPC Server and Main

**Files:**
- Create: `rag-service/app/grpc_server.py`
- Create: `rag-service/app/main.py`
- Create: `rag-service/requirements.txt`

- [ ] **Step 1: Create `rag-service/app/grpc_server.py`**

```python
import os
import logging

import grpc

from app.generated import demo_pb2, demo_pb2_grpc
from app.llm_client import chat_completion
from app.embedding_client import embed
from app.vectordb import VectorStore

logger = logging.getLogger(__name__)


class RAGServiceServicer(demo_pb2_grpc.RAGServiceServicer):
    def __init__(self, store: VectorStore):
        self._store = store
        self._llm_base_url = os.environ.get("LLM_BASE_URL", "http://ollama:11434/v1")
        self._llm_model = os.environ.get("LLM_MODEL", "llama3")
        self._llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
        self._embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", "http://ollama:11434/v1")
        self._embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    def Retrieve(self, request, context):
        top_k = request.top_k if request.top_k > 0 else 3
        query = request.query

        try:
            # 1. Generate embedding for the query
            query_embedding = embed(
                base_url=self._embedding_base_url,
                model=self._embedding_model,
                provider=self._llm_provider,
                text=query,
            )

            # 2. Search vector DB
            docs = self._store.search(query_embedding, top_k=top_k)
            sources = [doc.content for doc in docs]

            # 3. Build prompt with context
            if sources:
                context_text = "\n\n".join(sources)
                prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            else:
                prompt = query

            # 4. Call LLM
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If no context is provided, answer based on your general knowledge."},
                {"role": "user", "content": prompt},
            ]
            result = chat_completion(
                base_url=self._llm_base_url,
                model=self._llm_model,
                provider=self._llm_provider,
                messages=messages,
            )

            answer = result["choices"][0]["message"]["content"]
            model = result.get("model", self._llm_model)

            return demo_pb2.RetrieveResponse(
                answer=answer,
                sources=sources,
                model=model,
            )
        except Exception as e:
            logger.exception("Error in Retrieve")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.RetrieveResponse()
```

- [ ] **Step 2: Create `rag-service/app/main.py`**

```python
import os
import signal
import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from app.generated import demo_pb2_grpc
from app.grpc_server import RAGServiceServicer
from app.otel_setup import setup_otel, shutdown_otel
from app.vectordb import create_vector_store
from app.seed_data import seed_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def serve():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "rag-service")
    tracer_provider, meter_provider = setup_otel(service_name)

    listen_addr = os.environ.get("GRPC_LISTEN_ADDR", "[::]:50052")

    # Create vector store
    store = create_vector_store(
        db_type=os.environ.get("VECTOR_DB_TYPE", "qdrant"),
        url=os.environ.get("VECTOR_DB_URL", "http://qdrant:6333"),
        collection_name=os.environ.get("VECTOR_DB_COLLECTION", "demo-docs"),
    )

    # Seed data if needed
    try:
        seed_if_needed(
            store=store,
            embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", "http://ollama:11434/v1"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
            provider=os.environ.get("LLM_PROVIDER", "ollama"),
        )
    except Exception:
        logger.exception("Failed to seed data (will retry on next restart)")

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_RAGServiceServicer_to_server(RAGServiceServicer(store), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("demo.RAGService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"RAG service listening on {listen_addr}")

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(grace=5)
        shutdown_otel(tracer_provider, meter_provider)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
```

- [ ] **Step 3: Create `rag-service/requirements.txt`**

```
grpcio==1.68.1
grpcio-tools==1.68.1
grpcio-health-checking==1.68.1
protobuf==5.29.2
requests==2.32.3
qdrant-client==1.12.1
opentelemetry-api==1.29.0
opentelemetry-sdk==1.29.0
opentelemetry-exporter-otlp-proto-grpc==1.29.0
opentelemetry-instrumentation-grpc==0.50b0
opentelemetry-instrumentation-requests==0.50b0
```

- [ ] **Step 4: Commit**

```bash
git add rag-service/
git commit -m "feat: add RAG service gRPC server with health check and main entrypoint"
```

---

### Task 8: LLM Service gRPC Server and Main

**Files:**
- Create: `llm-service/app/grpc_server.py`
- Create: `llm-service/app/main.py`
- Create: `llm-service/requirements.txt`

- [ ] **Step 1: Create `llm-service/app/grpc_server.py`**

```python
import os
import logging

import grpc

from app.generated import demo_pb2, demo_pb2_grpc
from app.llm_client import chat_completion

logger = logging.getLogger(__name__)


class LLMServiceServicer(demo_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        self._base_url = os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1")
        self._model = os.environ.get("LLM_MODEL", "llama3")
        self._provider = os.environ.get("LLM_PROVIDER", "vllm")

    def Generate(self, request, context):
        try:
            messages = [
                {"role": "user", "content": request.prompt},
            ]
            result = chat_completion(
                base_url=self._base_url,
                model=self._model,
                provider=self._provider,
                messages=messages,
            )
            text = result["choices"][0]["message"]["content"]
            model = result.get("model", self._model)

            return demo_pb2.GenerateResponse(text=text, model=model)
        except Exception as e:
            logger.exception("Error in Generate")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.GenerateResponse()
```

- [ ] **Step 2: Create `llm-service/app/main.py`**

```python
import os
import signal
import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from app.generated import demo_pb2_grpc
from app.grpc_server import LLMServiceServicer
from app.otel_setup import setup_otel, shutdown_otel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def serve():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "llm-service")
    tracer_provider, meter_provider = setup_otel(service_name)

    listen_addr = os.environ.get("GRPC_LISTEN_ADDR", "[::]:50053")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceServicer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("demo.LLMService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"LLM service listening on {listen_addr}")

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(grace=5)
        shutdown_otel(tracer_provider, meter_provider)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
```

- [ ] **Step 3: Create `llm-service/requirements.txt`**

```
grpcio==1.68.1
grpcio-tools==1.68.1
grpcio-health-checking==1.68.1
protobuf==5.29.2
requests==2.32.3
opentelemetry-api==1.29.0
opentelemetry-sdk==1.29.0
opentelemetry-exporter-otlp-proto-grpc==1.29.0
opentelemetry-instrumentation-grpc==0.50b0
opentelemetry-instrumentation-requests==0.50b0
```

- [ ] **Step 4: Commit**

```bash
git add llm-service/
git commit -m "feat: add LLM service gRPC server with health check and main entrypoint"
```

---

### Task 9: Gateway (Go)

**Files:**
- Create: `gateway/go.mod`
- Create: `gateway/main.go`

- [ ] **Step 1: Create `gateway/go.mod`**

Run:
```bash
cd /home/thbertoldi/suse/suse-ai-demo-apps/gateway
go mod init github.com/suse/suse-ai-demo-apps/gateway
```

Then add dependencies:
```bash
go get google.golang.org/grpc@v1.68.1
go get google.golang.org/protobuf@v1.36.1
go get go.opentelemetry.io/otel@v1.33.0
go get go.opentelemetry.io/otel/sdk@v1.33.0
go get go.opentelemetry.io/otel/sdk/metric@v1.33.0
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@v1.33.0
go get go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc@v1.33.0
go get go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc@v0.58.0
go get google.golang.org/grpc/health@v1.68.1
```

- [ ] **Step 2: Create `gateway/main.go`**

```go
package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthgrpc "google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/suse/suse-ai-demo-apps/gateway/pb"
)

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// --- OTel setup ---

func setupOTel(ctx context.Context, serviceName string) (func(), error) {
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(semconv.SchemaURL, semconv.ServiceName(serviceName)),
	)
	if err != nil {
		return nil, err
	}

	traceExp, err := otlptracegrpc.New(ctx, otlptracegrpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(traceExp), sdktrace.WithResource(res))
	otel.SetTracerProvider(tp)

	metricExp, err := otlpmetricgrpc.New(ctx, otlpmetricgrpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp)), sdkmetric.WithResource(res))
	otel.SetMeterProvider(mp)

	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return func() {
		_ = tp.Shutdown(ctx)
		_ = mp.Shutdown(ctx)
	}, nil
}

// --- Gateway service ---

type gatewayServer struct {
	pb.UnimplementedDemoServiceServer
	ragConn *grpc.ClientConn
	llmConn *grpc.ClientConn
}

func (s *gatewayServer) Query(ctx context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
	client := pb.NewRAGServiceClient(s.ragConn)
	resp, err := client.Retrieve(ctx, &pb.RetrieveRequest{
		Query: req.Query,
		TopK:  req.TopK,
	})
	if err != nil {
		return nil, err
	}
	return &pb.QueryResponse{
		Answer:  resp.Answer,
		Sources: resp.Sources,
		Model:   resp.Model,
	}, nil
}

func (s *gatewayServer) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	client := pb.NewLLMServiceClient(s.llmConn)
	resp, err := client.Generate(ctx, &pb.GenerateRequest{
		Prompt: req.Message,
	})
	if err != nil {
		return nil, err
	}
	return &pb.ChatResponse{
		Reply: resp.Text,
		Model: resp.Model,
	}, nil
}

func main() {
	ctx := context.Background()

	serviceName := envOrDefault("OTEL_SERVICE_NAME", "gateway")
	shutdownOTel, err := setupOTel(ctx, serviceName)
	if err != nil {
		log.Fatalf("failed to setup OTel: %v", err)
	}
	defer shutdownOTel()

	listenAddr := envOrDefault("GRPC_LISTEN_ADDR", ":50051")
	ragAddr := envOrDefault("RAG_SERVICE_ADDR", "rag-service:50052")
	llmAddr := envOrDefault("LLM_SERVICE_ADDR", "llm-service:50053")

	// Client connections with OTel interceptors
	ragConn, err := grpc.NewClient(ragAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
	)
	if err != nil {
		log.Fatalf("failed to connect to RAG service: %v", err)
	}
	defer ragConn.Close()

	llmConn, err := grpc.NewClient(llmAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
	)
	if err != nil {
		log.Fatalf("failed to connect to LLM service: %v", err)
	}
	defer llmConn.Close()

	// gRPC server with OTel interceptor
	srv := grpc.NewServer(grpc.StatsHandler(otelgrpc.NewServerHandler()))

	pb.RegisterDemoServiceServer(srv, &gatewayServer{ragConn: ragConn, llmConn: llmConn})

	// Health service
	healthSrv := healthgrpc.NewServer()
	healthpb.RegisterHealthServer(srv, healthSrv)
	healthSrv.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("demo.DemoService", healthpb.HealthCheckResponse_SERVING)

	lis, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	go func() {
		log.Printf("Gateway listening on %s", listenAddr)
		if err := srv.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	// Graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	<-sigCh
	log.Println("Shutting down...")
	srv.GracefulStop()
}
```

- [ ] **Step 3: Generate proto and tidy modules**

```bash
cd /home/thbertoldi/suse/suse-ai-demo-apps/proto && make go
cd /home/thbertoldi/suse/suse-ai-demo-apps/gateway && go mod tidy
```

Verify: `go build ./...` succeeds

- [ ] **Step 4: Commit**

```bash
git add gateway/
git commit -m "feat: add Go gateway with gRPC routing, OTel instrumentation, and health checks"
```

---

### Task 10: Traffic Generator

**Files:**
- Create: `traffic-gen/main.py`
- Create: `traffic-gen/requirements.txt`

- [ ] **Step 1: Create `traffic-gen/main.py`**

```python
import os
import random
import signal
import logging
import time

import grpc
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

from generated import demo_pb2, demo_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAG_QUERIES = [
    "What is a Kubernetes pod?",
    "How do Linux containers work?",
    "What is a container runtime?",
    "Explain Kubernetes deployments",
    "What are container images?",
    "What is OpenTelemetry?",
    "Explain distributed tracing",
    "What is a Kubernetes service?",
]

CHAT_MESSAGES = [
    "Explain microservices in one sentence",
    "What is observability?",
    "What is the difference between monitoring and observability?",
    "Explain the CAP theorem briefly",
    "What is a service mesh?",
    "What are the benefits of containerization?",
]

running = True


def shutdown(signum, frame):
    global running
    logger.info("Received shutdown signal")
    running = False


def main():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "traffic-gen")
    resource = Resource.create({"service.name": service_name})

    trace_exporter = OTLPSpanExporter(insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    set_global_textmap(CompositeHTTPPropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    GrpcInstrumentorClient().instrument()

    gateway_addr = os.environ.get("GATEWAY_ADDR", "gateway:50051")
    interval = int(os.environ.get("INTERVAL_SECONDS", "5"))

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    channel = grpc.insecure_channel(gateway_addr)
    stub = demo_pb2_grpc.DemoServiceStub(channel)

    # Interleave RAG and Chat queries
    queries = []
    for i in range(max(len(RAG_QUERIES), len(CHAT_MESSAGES))):
        if i < len(RAG_QUERIES):
            queries.append(("rag", RAG_QUERIES[i]))
        if i < len(CHAT_MESSAGES):
            queries.append(("chat", CHAT_MESSAGES[i]))

    idx = 0
    logger.info(f"Starting traffic generation to {gateway_addr} every ~{interval}s")

    while running:
        query_type, query_text = queries[idx % len(queries)]
        idx += 1

        try:
            if query_type == "rag":
                logger.info(f"Sending RAG query: {query_text}")
                resp = stub.Query(demo_pb2.QueryRequest(query=query_text, top_k=3), timeout=120)
                logger.info(f"RAG response model={resp.model}, sources={len(resp.sources)}")
            else:
                logger.info(f"Sending Chat message: {query_text}")
                resp = stub.Chat(demo_pb2.ChatRequest(message=query_text), timeout=120)
                logger.info(f"Chat response model={resp.model}")
        except grpc.RpcError as e:
            logger.warning(f"gRPC error: {e.code()} {e.details()}")
        except Exception as e:
            logger.warning(f"Error: {e}")

        # Sleep with jitter
        jitter = random.uniform(0, 2)
        sleep_time = interval + jitter
        start = time.monotonic()
        while running and (time.monotonic() - start) < sleep_time:
            time.sleep(0.5)

    logger.info("Shutting down OTel...")
    channel.close()
    tracer_provider.shutdown()
    meter_provider.shutdown()
    logger.info("Done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `traffic-gen/requirements.txt`**

```
grpcio==1.68.1
grpcio-tools==1.68.1
protobuf==5.29.2
opentelemetry-api==1.29.0
opentelemetry-sdk==1.29.0
opentelemetry-exporter-otlp-proto-grpc==1.29.0
opentelemetry-instrumentation-grpc==0.50b0
```

- [ ] **Step 3: Commit**

```bash
git add traffic-gen/
git commit -m "feat: add traffic generator with round-robin RAG/Chat queries"
```

---

### Task 11: Dockerfiles

**Files:**
- Create: `gateway/Dockerfile`
- Create: `rag-service/Dockerfile`
- Create: `llm-service/Dockerfile`
- Create: `traffic-gen/Dockerfile`

- [ ] **Step 1: Create `gateway/Dockerfile`**

```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /gateway .

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=builder /gateway /gateway
ENTRYPOINT ["/gateway"]
```

- [ ] **Step 2: Create `rag-service/Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
```

- [ ] **Step 3: Create `llm-service/Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
```

- [ ] **Step 4: Create `traffic-gen/Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

- [ ] **Step 5: Create `.dockerignore`** at repo root

```
.git
.github
docs
helm
*.md
__pycache__
*.pyc
.env
```

- [ ] **Step 6: Commit**

```bash
git add gateway/Dockerfile rag-service/Dockerfile llm-service/Dockerfile traffic-gen/Dockerfile .dockerignore
git commit -m "feat: add Dockerfiles for all services"
```

---

### Task 12: Helm Chart

**Files:**
- Create: `helm/suse-ai-demo/Chart.yaml`
- Create: `helm/suse-ai-demo/values.yaml`
- Create: `helm/suse-ai-demo/templates/gateway-deployment.yaml`
- Create: `helm/suse-ai-demo/templates/gateway-service.yaml`
- Create: `helm/suse-ai-demo/templates/rag-service-deployment.yaml`
- Create: `helm/suse-ai-demo/templates/rag-service-service.yaml`
- Create: `helm/suse-ai-demo/templates/llm-service-deployment.yaml`
- Create: `helm/suse-ai-demo/templates/llm-service-service.yaml`
- Create: `helm/suse-ai-demo/templates/traffic-gen-deployment.yaml`

- [ ] **Step 1: Create `helm/suse-ai-demo/Chart.yaml`**

```yaml
apiVersion: v2
name: suse-ai-demo
description: SUSE AI Demo Apps - RAG pipeline with OpenTelemetry GenAI instrumentation
type: application
version: 0.1.0
appVersion: "0.1.0"
```

- [ ] **Step 2: Create `helm/suse-ai-demo/values.yaml`**

```yaml
gateway:
  image:
    repository: ghcr.io/suse/suse-ai-demo-gateway
    tag: latest
    pullPolicy: IfNotPresent
  replicas: 1
  ragServiceAddr: rag-service:50052
  llmServiceAddr: llm-service:50053
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 256Mi

ragService:
  image:
    repository: ghcr.io/suse/suse-ai-demo-rag-service
    tag: latest
    pullPolicy: IfNotPresent
  replicas: 1
  llm:
    baseUrl: http://ollama:11434/v1
    model: llama3
    provider: ollama
  embedding:
    baseUrl: http://ollama:11434/v1
    model: nomic-embed-text
  vectorDb:
    type: qdrant
    url: http://qdrant:6333
    collection: demo-docs
  enableContentEvents: "false"
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 256Mi

llmService:
  image:
    repository: ghcr.io/suse/suse-ai-demo-llm-service
    tag: latest
    pullPolicy: IfNotPresent
  replicas: 1
  llm:
    baseUrl: http://vllm:8000/v1
    model: llama3
    provider: vllm
  enableContentEvents: "false"
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 256Mi

trafficGen:
  image:
    repository: ghcr.io/suse/suse-ai-demo-traffic-gen
    tag: latest
    pullPolicy: IfNotPresent
  enabled: true
  gatewayAddr: gateway:50051
  intervalSeconds: "5"
  resources:
    requests:
      cpu: 50m
      memory: 64Mi
    limits:
      cpu: 200m
      memory: 128Mi

otel:
  exporterEndpoint: http://otel-collector:4317
```

- [ ] **Step 3: Create `helm/suse-ai-demo/templates/gateway-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
  labels:
    app: gateway
spec:
  replicas: {{ .Values.gateway.replicas }}
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
    spec:
      containers:
        - name: gateway
          image: "{{ .Values.gateway.image.repository }}:{{ .Values.gateway.image.tag }}"
          imagePullPolicy: {{ .Values.gateway.image.pullPolicy }}
          ports:
            - containerPort: 50051
              protocol: TCP
          env:
            - name: GRPC_LISTEN_ADDR
              value: ":50051"
            - name: RAG_SERVICE_ADDR
              value: {{ .Values.gateway.ragServiceAddr | quote }}
            - name: LLM_SERVICE_ADDR
              value: {{ .Values.gateway.llmServiceAddr | quote }}
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: {{ .Values.otel.exporterEndpoint | quote }}
            - name: OTEL_SERVICE_NAME
              value: "gateway"
          readinessProbe:
            grpc:
              port: 50051
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 50051
            initialDelaySeconds: 5
            periodSeconds: 15
          resources:
            {{- toYaml .Values.gateway.resources | nindent 12 }}
```

- [ ] **Step 4: Create `helm/suse-ai-demo/templates/gateway-service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gateway
spec:
  selector:
    app: gateway
  ports:
    - port: 50051
      targetPort: 50051
      protocol: TCP
```

- [ ] **Step 5: Create `helm/suse-ai-demo/templates/rag-service-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  labels:
    app: rag-service
spec:
  replicas: {{ .Values.ragService.replicas }}
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
        - name: rag-service
          image: "{{ .Values.ragService.image.repository }}:{{ .Values.ragService.image.tag }}"
          imagePullPolicy: {{ .Values.ragService.image.pullPolicy }}
          ports:
            - containerPort: 50052
              protocol: TCP
          env:
            - name: GRPC_LISTEN_ADDR
              value: "[::]:50052"
            - name: LLM_BASE_URL
              value: {{ .Values.ragService.llm.baseUrl | quote }}
            - name: LLM_MODEL
              value: {{ .Values.ragService.llm.model | quote }}
            - name: LLM_PROVIDER
              value: {{ .Values.ragService.llm.provider | quote }}
            - name: EMBEDDING_BASE_URL
              value: {{ .Values.ragService.embedding.baseUrl | quote }}
            - name: EMBEDDING_MODEL
              value: {{ .Values.ragService.embedding.model | quote }}
            - name: VECTOR_DB_TYPE
              value: {{ .Values.ragService.vectorDb.type | quote }}
            - name: VECTOR_DB_URL
              value: {{ .Values.ragService.vectorDb.url | quote }}
            - name: VECTOR_DB_COLLECTION
              value: {{ .Values.ragService.vectorDb.collection | quote }}
            - name: ENABLE_OTEL_CONTENT_EVENTS
              value: {{ .Values.ragService.enableContentEvents | quote }}
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: {{ .Values.otel.exporterEndpoint | quote }}
            - name: OTEL_SERVICE_NAME
              value: "rag-service"
          readinessProbe:
            grpc:
              port: 50052
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 50052
            initialDelaySeconds: 10
            periodSeconds: 15
          resources:
            {{- toYaml .Values.ragService.resources | nindent 12 }}
```

- [ ] **Step 6: Create `helm/suse-ai-demo/templates/rag-service-service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  selector:
    app: rag-service
  ports:
    - port: 50052
      targetPort: 50052
      protocol: TCP
```

- [ ] **Step 7: Create `helm/suse-ai-demo/templates/llm-service-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
  labels:
    app: llm-service
spec:
  replicas: {{ .Values.llmService.replicas }}
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      containers:
        - name: llm-service
          image: "{{ .Values.llmService.image.repository }}:{{ .Values.llmService.image.tag }}"
          imagePullPolicy: {{ .Values.llmService.image.pullPolicy }}
          ports:
            - containerPort: 50053
              protocol: TCP
          env:
            - name: GRPC_LISTEN_ADDR
              value: "[::]:50053"
            - name: LLM_BASE_URL
              value: {{ .Values.llmService.llm.baseUrl | quote }}
            - name: LLM_MODEL
              value: {{ .Values.llmService.llm.model | quote }}
            - name: LLM_PROVIDER
              value: {{ .Values.llmService.llm.provider | quote }}
            - name: ENABLE_OTEL_CONTENT_EVENTS
              value: {{ .Values.llmService.enableContentEvents | quote }}
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: {{ .Values.otel.exporterEndpoint | quote }}
            - name: OTEL_SERVICE_NAME
              value: "llm-service"
          readinessProbe:
            grpc:
              port: 50053
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 50053
            initialDelaySeconds: 5
            periodSeconds: 15
          resources:
            {{- toYaml .Values.llmService.resources | nindent 12 }}
```

- [ ] **Step 8: Create `helm/suse-ai-demo/templates/llm-service-service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-service
  ports:
    - port: 50053
      targetPort: 50053
      protocol: TCP
```

- [ ] **Step 9: Create `helm/suse-ai-demo/templates/traffic-gen-deployment.yaml`**

```yaml
{{- if .Values.trafficGen.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: traffic-gen
  labels:
    app: traffic-gen
spec:
  replicas: 1
  selector:
    matchLabels:
      app: traffic-gen
  template:
    metadata:
      labels:
        app: traffic-gen
    spec:
      containers:
        - name: traffic-gen
          image: "{{ .Values.trafficGen.image.repository }}:{{ .Values.trafficGen.image.tag }}"
          imagePullPolicy: {{ .Values.trafficGen.image.pullPolicy }}
          env:
            - name: GATEWAY_ADDR
              value: {{ .Values.trafficGen.gatewayAddr | quote }}
            - name: INTERVAL_SECONDS
              value: {{ .Values.trafficGen.intervalSeconds | quote }}
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: {{ .Values.otel.exporterEndpoint | quote }}
            - name: OTEL_SERVICE_NAME
              value: "traffic-gen"
          resources:
            {{- toYaml .Values.trafficGen.resources | nindent 12 }}
{{- end }}
```

- [ ] **Step 10: Validate Helm chart**

Run: `helm lint helm/suse-ai-demo/`
Expected: "0 chart(s) failed"

- [ ] **Step 11: Commit**

```bash
git add helm/
git commit -m "feat: add Helm chart for all demo services"
```

---

### Task 13: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/build.yaml`

- [ ] **Step 1: Create `.github/workflows/build.yaml`**

```yaml
name: Build and Push Container Images

on:
  push:
    branches: [main]
    tags: ["v*"]

env:
  REGISTRY: ghcr.io

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    strategy:
      matrix:
        include:
          - service: gateway
            context: ./gateway
            image: suse-ai-demo-gateway
          - service: rag-service
            context: ./rag-service
            image: suse-ai-demo-rag-service
          - service: llm-service
            context: ./llm-service
            image: suse-ai-demo-llm-service
          - service: traffic-gen
            context: ./traffic-gen
            image: suse-ai-demo-traffic-gen
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ matrix.image }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: ${{ matrix.context }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Commit**

```bash
git add .github/
git commit -m "feat: add GitHub Actions workflow for building container images"
```

---

### Task 14: Final Verification

- [ ] **Step 1: Verify project structure**

Run: `find /home/thbertoldi/suse/suse-ai-demo-apps -type f -not -path '*/.git/*' | sort`

Verify all expected files exist per the spec.

- [ ] **Step 2: Validate Helm chart**

Run: `helm lint helm/suse-ai-demo/`
Run: `helm template test helm/suse-ai-demo/` — verify output renders correctly

- [ ] **Step 3: Verify Go builds**

Run: `cd gateway && go build ./...`

- [ ] **Step 4: Verify Python syntax**

Run: `python -m py_compile rag-service/app/main.py`
Run: `python -m py_compile llm-service/app/main.py`
Run: `python -m py_compile traffic-gen/main.py`

- [ ] **Step 5: Verify Docker builds (dry run)**

Run: `docker build --check gateway/` (if available, otherwise skip)

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address issues found during verification"
```
