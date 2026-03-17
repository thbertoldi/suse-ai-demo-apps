# SUSE AI Demo Apps — Design Spec

## Overview

A set of demo applications forming a RAG (Retrieval-Augmented Generation) pipeline, instrumented with OpenTelemetry following GenAI semantic conventions. The apps call each other over gRPC and HTTP, producing rich distributed traces and metrics suitable for observability demos.

## Architecture

```
                    gRPC                  gRPC                HTTP
  [Traffic Gen] ──────> [Gateway (Go)] ──────> [RAG (Python)] ──────> [Qdrant]
                              │                      │
                              │                      │ HTTP (OpenAI-compat API)
                              │                      ├──────> [Ollama]
                              │                      │
                              │ gRPC                 │ (embedding)
                              └──────> [LLM Svc (Py)]
                                              │
                                              │ HTTP (OpenAI-compat API)
                                              └──────> [vLLM]
```

### Services We Build

1. **Gateway (Go)** — gRPC server that receives user queries and routes them to either the RAG service or the LLM service based on request type. Provides request validation and entry-point tracing. Uses `otelgrpc` for both server and client interceptors to ensure proper context propagation on outbound gRPC calls. Registers the standard gRPC health checking service (`grpc.health.v1.Health`).

2. **RAG Service (Python)** — Receives queries via gRPC, generates embeddings (via `/v1/embeddings` endpoint), searches Qdrant for relevant context, constructs a prompt with retrieved documents, and calls Ollama (OpenAI-compatible `/v1/chat/completions`) to generate answers. Registers `grpc.health.v1.Health`.

3. **LLM Service (Python)** — Receives queries via gRPC and calls vLLM (OpenAI-compatible `/v1/chat/completions`) directly for general-purpose Q&A without retrieval. Registers `grpc.health.v1.Health`.

4. **Traffic Generator (Python)** — A loop that sends gRPC requests to the Gateway with predefined queries at configurable intervals, producing continuous telemetry data.

**Why two LLM backends?** The split between Ollama (RAG service) and vLLM (LLM service) is intentional — it demonstrates distributed tracing across heterogeneous GenAI providers, showing how OTel captures `gen_ai.provider.name` differences in the same trace topology. Both use the OpenAI-compatible API, so the code is identical; only the endpoint and provider name differ.

### External Dependencies (not built, assumed present)

- Qdrant (vector database)
- Ollama (LLM backend for RAG service)
- vLLM (LLM backend for LLM service)
- OpenTelemetry Collector (receives traces and metrics via OTLP/gRPC)

## Inter-Service Communication

- **gRPC** between Traffic Gen → Gateway → RAG Service / LLM Service
- **HTTP** for outbound calls to LLM backends (OpenAI-compatible `/v1/chat/completions`) and Qdrant
- **W3C TraceContext** propagation across all gRPC and HTTP calls

### Proto Definition

A single `proto/demo.proto` defines the following services and messages:

```protobuf
syntax = "proto3";
package demo;
option go_package = "github.com/suse/suse-ai-demo-apps/proto/demo";

// Gateway-facing API (called by traffic generator and external clients)
service DemoService {
  rpc Query(QueryRequest) returns (QueryResponse);   // routes to RAG service
  rpc Chat(ChatRequest) returns (ChatResponse);       // routes to LLM service
}

// Internal: RAG service
service RAGService {
  rpc Retrieve(RetrieveRequest) returns (RetrieveResponse);
}

// Internal: LLM service
service LLMService {
  rpc Generate(GenerateRequest) returns (GenerateResponse);
}

message QueryRequest {
  string query = 1;            // user's natural language question
  int32 top_k = 2;             // number of documents to retrieve (default: 3)
}

message QueryResponse {
  string answer = 1;           // LLM-generated answer
  repeated string sources = 2; // retrieved document snippets used as context
  string model = 3;            // model that generated the answer
}

message ChatRequest {
  string message = 1;          // user's message
}

message ChatResponse {
  string reply = 1;            // LLM-generated reply
  string model = 2;            // model that generated the reply
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

## OpenTelemetry Instrumentation

### Traces — GenAI Semantic Conventions

**Chat/completion calls** produce spans following `gen_ai.*` conventions:
- **Span name**: `chat {model_name}`
- **Attributes**:
  - `gen_ai.operation.name` — `chat`
  - `gen_ai.request.model` — requested model name
  - `gen_ai.response.model` — actual model used
  - `gen_ai.provider.name` — `ollama` or `vllm`
  - `gen_ai.usage.input_tokens` — prompt token count
  - `gen_ai.usage.output_tokens` — completion token count
  - `gen_ai.response.finish_reasons` — e.g., `["stop"]`
  - `gen_ai.response.id` — completion ID
- **Events** (opt-in via `ENABLE_OTEL_CONTENT_EVENTS`):
  - `gen_ai.input.messages` — input conversation
  - `gen_ai.output.messages` — output completion

**Embedding calls** (RAG service only) produce spans with:
- **Span name**: `embed {model_name}`
- **Attributes**:
  - `gen_ai.operation.name` — `embed`
  - `gen_ai.request.model` — embedding model name
  - `gen_ai.provider.name` — reuses `LLM_PROVIDER` value (since embeddings and chat use the same Ollama backend)
  - `gen_ai.usage.input_tokens` — token count from embedding response
- Embedding logic lives in a dedicated `embedding_client.py` module in the RAG service

Vector DB operations produce spans with:
- `db.system` — `qdrant`
- `db.operation.name` — `search`, `upsert`
- `db.collection.name` — collection name

gRPC spans are auto-instrumented:
- Go: `otelgrpc` interceptors
- Python: `opentelemetry-instrumentation-grpc`

HTTP spans for outbound LLM/embedding calls are auto-instrumented:
- Python: `opentelemetry-instrumentation-requests`

### Metrics — GenAI Semantic Conventions

- `gen_ai.client.token.usage` — histogram of token counts (attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.token.type` [input/output])
- `gen_ai.client.operation.duration` — histogram of LLM call duration (attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.provider.name`)
- Standard HTTP/gRPC server and client metrics from auto-instrumentation

### Export

All services export telemetry via OTLP/gRPC to an OpenTelemetry Collector. The endpoint is configured via `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Error Handling

Errors are propagated as gRPC status codes and recorded on spans:
- LLM/embedding HTTP errors → span status set to `ERROR`, `error.type` attribute set (e.g., `timeout`, `connection_error`, `500`)
- Vector DB errors → span status set to `ERROR` with `error.type`
- gRPC errors → auto-instrumentation records the status code on the span
- Zero Qdrant results → not an error; the RAG service proceeds with an empty context and the LLM generates an answer based solely on the query
- All services flush the OTel SDK on shutdown (`TracerProvider.shutdown()` / `TracerProvider.Shutdown()`) to ensure in-flight spans are exported
- The traffic generator handles SIGTERM to stop its loop and flush telemetry before exiting

## Vector DB Repository Pattern

The RAG service uses a repository pattern for vector DB access, allowing easy expansion to other backends.

```
rag-service/app/vectordb/
├── base.py       # Abstract VectorStore class
├── qdrant.py     # Qdrant implementation
└── factory.py    # Factory: reads VECTOR_DB_TYPE, returns implementation
```

**`VectorStore` interface** (collection name is set at construction time via config, not per-call):
- `search(query_embedding: list[float], top_k: int) -> list[Document]`
- `upsert(documents: list[Document]) -> None`
- `health() -> bool`

**`Document` model:**
- `id: str`
- `content: str`
- `embedding: list[float]`
- `metadata: dict`
- `score: float` (populated on search results)

To add Milvus or OpenSearch later: implement `VectorStore` in a new file and register it in `factory.py`.

### Data Seeding

Qdrant must be populated before the RAG pipeline can return meaningful results. The RAG service includes a **seed mode**: on startup, if the collection does not exist or is empty, it creates the collection and upserts a bundled set of sample documents (hardcoded in a `seed_data.py` module). The sample documents cover a few topics (e.g., Kubernetes basics, Linux container concepts) so the RAG pipeline has content to retrieve. This runs once automatically — no separate job or manual step needed.

## Configuration

All services are configured via environment variables.

### Gateway (Go)
| Variable | Default | Description |
|---|---|---|
| `GRPC_LISTEN_ADDR` | `:50051` | gRPC listen address |
| `RAG_SERVICE_ADDR` | `rag-service:50052` | RAG service gRPC address |
| `LLM_SERVICE_ADDR` | `llm-service:50053` | LLM service gRPC address |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector endpoint |
| `OTEL_SERVICE_NAME` | `gateway` | OTel service name |

### RAG Service (Python)
| Variable | Default | Description |
|---|---|---|
| `GRPC_LISTEN_ADDR` | `[::]:50052` | gRPC listen address |
| `LLM_BASE_URL` | `http://ollama:11434/v1` | LLM OpenAI-compat endpoint |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `ollama` | Provider name for OTel attributes |
| `EMBEDDING_BASE_URL` | `http://ollama:11434/v1` | Embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `VECTOR_DB_TYPE` | `qdrant` | Vector DB backend |
| `VECTOR_DB_URL` | `http://qdrant:6333` | Vector DB URL |
| `VECTOR_DB_COLLECTION` | `demo-docs` | Collection name |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector endpoint |
| `OTEL_SERVICE_NAME` | `rag-service` | OTel service name |
| `ENABLE_OTEL_CONTENT_EVENTS` | `false` | Enable gen_ai input/output message events |

### LLM Service (Python)
| Variable | Default | Description |
|---|---|---|
| `GRPC_LISTEN_ADDR` | `[::]:50053` | gRPC listen address |
| `LLM_BASE_URL` | `http://vllm:8000/v1` | LLM OpenAI-compat endpoint |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `vllm` | Provider name for OTel attributes |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector endpoint |
| `OTEL_SERVICE_NAME` | `llm-service` | OTel service name |
| `ENABLE_OTEL_CONTENT_EVENTS` | `false` | Enable gen_ai input/output message events |

### Traffic Generator (Python)
| Variable | Default | Description |
|---|---|---|
| `GATEWAY_ADDR` | `gateway:50051` | Gateway gRPC address |
| `INTERVAL_SECONDS` | `5` | Delay between requests |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector endpoint |
| `OTEL_SERVICE_NAME` | `traffic-gen` | OTel service name |

### Traffic Generator Query Set

The traffic generator sends a mix of requests to exercise both code paths:
- **RAG queries** (via `DemoService.Query`): predefined questions that match the seeded documents, e.g., "What is a Kubernetes pod?", "How do Linux containers work?", "What is a container runtime?"
- **Chat messages** (via `DemoService.Chat`): general questions that go directly to the LLM, e.g., "Explain microservices in one sentence", "What is observability?"
- Queries are selected **round-robin** from the list, alternating between RAG and Chat requests, with a configurable delay between each request
- A small random jitter (0-2s) is added to the interval to avoid perfectly regular traffic patterns

## Project Structure

```
suse-ai-demo-apps/
├── proto/
│   └── demo.proto
├── gateway/
│   ├── main.go
│   ├── go.mod
│   ├── go.sum
│   └── Dockerfile
├── rag-service/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── otel_setup.py
│   │   ├── llm_client.py
│   │   ├── embedding_client.py
│   │   ├── seed_data.py
│   │   ├── grpc_server.py
│   │   └── vectordb/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── qdrant.py
│   │       └── factory.py
│   ├── requirements.txt
│   └── Dockerfile
├── llm-service/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── otel_setup.py
│   │   ├── llm_client.py
│   │   └── grpc_server.py
│   ├── requirements.txt
│   └── Dockerfile
├── traffic-gen/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── helm/
│   └── suse-ai-demo/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── gateway-deployment.yaml
│           ├── gateway-service.yaml
│           ├── rag-service-deployment.yaml
│           ├── rag-service-service.yaml
│           ├── llm-service-deployment.yaml
│           ├── llm-service-service.yaml
│           └── traffic-gen-deployment.yaml
└── .github/
    └── workflows/
        └── build.yaml
```

## Helm Chart

The chart deploys only the 4 services we build. External dependencies (Qdrant, Ollama, vLLM, OTel Collector) are assumed to already be present in the cluster.

Each service deployment includes:
- **Readiness probe**: gRPC health check for Gateway/RAG/LLM services (using `grpc_health_v1`); not applicable for traffic-gen
- **Liveness probe**: TCP check on the gRPC port for Gateway/RAG/LLM services
- **Resource requests**: `cpu: 100m, memory: 128Mi` for all services; `limits: cpu: 500m, memory: 256Mi`
- `OTEL_SERVICE_NAME` is hardcoded per-template (not configurable in values.yaml) since service names are fixed identities

### values.yaml

```yaml
gateway:
  image:
    repository: ghcr.io/suse/suse-ai-demo-gateway
    tag: latest
  replicas: 1
  ragServiceAddr: rag-service:50052
  llmServiceAddr: llm-service:50053

ragService:
  image:
    repository: ghcr.io/suse/suse-ai-demo-rag-service
    tag: latest
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
  enableContentEvents: false

llmService:
  image:
    repository: ghcr.io/suse/suse-ai-demo-llm-service
    tag: latest
  replicas: 1
  llm:
    baseUrl: http://vllm:8000/v1
    model: llama3
    provider: vllm
  enableContentEvents: false

trafficGen:
  image:
    repository: ghcr.io/suse/suse-ai-demo-traffic-gen
    tag: latest
  gatewayAddr: gateway:50051
  intervalSeconds: 5
  enabled: true

otel:
  exporterEndpoint: http://otel-collector:4317
```

## GitHub Actions

Single workflow `.github/workflows/build.yaml`:
- **Triggers**: push to `main`, tags matching `v*`
- **Matrix**: builds all 4 images in parallel (`gateway`, `rag-service`, `llm-service`, `traffic-gen`)
- **Steps**: checkout → login to GHCR → docker build → docker push
- **Tags**: `latest` on main branch, semver on version tags
- **Caching**: Docker layer caching via `docker/build-push-action`
