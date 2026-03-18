# SUSE AI Demo Apps

A set of microservices forming a **RAG (Retrieval-Augmented Generation) pipeline**, fully instrumented with [OpenTelemetry](https://opentelemetry.io/) following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Designed for demoing observability in AI-powered applications.

## Architecture

```
                    gRPC                  gRPC                HTTP
  [Traffic Gen] -------> [Gateway (Go)] -------> [RAG (Python)] -------> [Qdrant]
                              |                      |
                              |                      | HTTP (OpenAI-compat)
                              |                      '-------> [Ollama]
                              |
                              | gRPC
                              '-------> [LLM Service (Python)]
                                              |
                                              | HTTP (OpenAI-compat)
                                              '-------> [vLLM]
```

| Service | Language | Description | Port |
|---------|----------|-------------|------|
| **Gateway** | Go | gRPC entry point. Routes `Query` requests to the RAG service and `Chat` requests to the LLM service. | 50051 |
| **RAG Service** | Python | Embeds the user query, searches Qdrant for relevant documents, builds a context-augmented prompt, and calls Ollama for an answer. | 50052 |
| **LLM Service** | Python | Forwards prompts directly to vLLM for general-purpose Q&A (no retrieval). | 50053 |
| **Traffic Generator** | Python | Sends a round-robin mix of RAG and Chat queries to the Gateway at a configurable interval, producing continuous telemetry. | - |

Two different LLM backends (Ollama and vLLM) are used intentionally to demonstrate distributed tracing across heterogeneous GenAI providers.

## OpenTelemetry Instrumentation

All services export **traces and metrics** via OTLP/gRPC to an OpenTelemetry Collector.

### Traces

- **GenAI spans** on every LLM and embedding call: `chat {model}`, `embed {model}`
  - Attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.response.id`
  - Optional content events: `gen_ai.input.messages`, `gen_ai.output.messages` (enable with `ENABLE_OTEL_CONTENT_EVENTS=true`)
- **Vector DB spans**: `db.system=qdrant`, `db.operation.name`, `db.collection.name`
- **gRPC spans**: auto-instrumented on all server and client connections
- **HTTP spans**: auto-instrumented on outbound LLM/embedding calls
- **W3C TraceContext** propagation across all hops

### Metrics

- `gen_ai.client.token.usage` — histogram of token counts by model, operation, and token type
- `gen_ai.client.operation.duration` — histogram of LLM/embedding call duration
- Standard gRPC and HTTP server/client metrics from auto-instrumentation

## Prerequisites

- **Go 1.25+** (for building the gateway)
- **Python 3.12+** (for building the Python services)
- **protoc** with `protoc-gen-go`, `protoc-gen-go-grpc`, and `grpcio-tools` (for regenerating proto stubs)
- **Docker** (for building container images)
- **Helm 3** (for deploying to Kubernetes)

External services (not included, must be running separately):
- [Qdrant](https://qdrant.tech/) — vector database
- [Ollama](https://ollama.ai/) — LLM backend for the RAG service
- [vLLM](https://docs.vllm.ai/) — LLM backend for the LLM service
- An [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) — receives traces and metrics

## Building

### Container Images

Each service has a Dockerfile. Build them from the repo root:

```bash
docker build -t suse-ai-demo-gateway ./gateway
docker build -t suse-ai-demo-rag-service ./rag-service
docker build -t suse-ai-demo-llm-service ./llm-service
docker build -t suse-ai-demo-traffic-gen ./traffic-gen
```

Images are also built automatically via GitHub Actions on every push to `main` and on version tags, and published to `ghcr.io`.

### Building Locally (without Docker)

**Gateway (Go):**

```bash
cd gateway
go build -o gateway .
```

**Python services:**

```bash
cd rag-service  # or llm-service, traffic-gen
pip install -r requirements.txt
```

### Regenerating Proto Stubs

If you modify `proto/demo.proto`:

```bash
cd proto
make all
```

This regenerates Go stubs in `gateway/pb/` and Python stubs in `{rag,llm}-service/app/generated/` and `traffic-gen/generated/`.

## Running Locally

Each service is configured via environment variables. Set them to point at your local instances:

```bash
# Terminal 1: RAG service
cd rag-service
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3
export LLM_PROVIDER=ollama
export EMBEDDING_BASE_URL=http://localhost:11434/v1
export EMBEDDING_MODEL=nomic-embed-text
export VECTOR_DB_URL=http://localhost:6333
export VECTOR_DB_COLLECTION=demo-docs
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python -m app.main

# Terminal 2: LLM service
cd llm-service
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=llama3
export LLM_PROVIDER=vllm
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python -m app.main

# Terminal 3: Gateway
cd gateway
export RAG_SERVICE_ADDR=localhost:50052
export LLM_SERVICE_ADDR=localhost:50053
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
./gateway

# Terminal 4: Traffic generator
cd traffic-gen
export GATEWAY_ADDR=localhost:50051
export INTERVAL_SECONDS=10
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python main.py
```

## Deploying with Helm

The Helm chart deploys the 4 services we build. External dependencies (Qdrant, Ollama, vLLM, OTel Collector) must already be running in the cluster.

```bash
helm install demo ./helm/suse-ai-demo \
  --set ragService.llm.baseUrl=http://ollama:11434/v1 \
  --set ragService.vectorDb.url=http://qdrant:6333 \
  --set llmService.llm.baseUrl=http://vllm:8000/v1 \
  --set otel.exporterEndpoint=http://otel-collector:4317
```

See [`helm/suse-ai-demo/values.yaml`](helm/suse-ai-demo/values.yaml) for all configurable values.

## Configuration Reference

### Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_LISTEN_ADDR` | `:50051` | gRPC listen address |
| `RAG_SERVICE_ADDR` | `rag-service:50052` | RAG service address |
| `LLM_SERVICE_ADDR` | `llm-service:50053` | LLM service address |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector |
| `OTEL_SERVICE_NAME` | `gateway` | Service name for telemetry |

### RAG Service

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_LISTEN_ADDR` | `[::]:50052` | gRPC listen address |
| `LLM_BASE_URL` | `http://ollama:11434/v1` | LLM endpoint (OpenAI-compat) |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `ollama` | Provider name for OTel attributes |
| `EMBEDDING_BASE_URL` | `http://ollama:11434/v1` | Embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `VECTOR_DB_TYPE` | `qdrant` | Vector DB backend |
| `VECTOR_DB_URL` | `http://qdrant:6333` | Vector DB address |
| `VECTOR_DB_COLLECTION` | `demo-docs` | Collection name |
| `ENABLE_OTEL_CONTENT_EVENTS` | `false` | Log input/output messages as span events |

### LLM Service

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_LISTEN_ADDR` | `[::]:50053` | gRPC listen address |
| `LLM_BASE_URL` | `http://vllm:8000/v1` | LLM endpoint (OpenAI-compat) |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `vllm` | Provider name for OTel attributes |
| `ENABLE_OTEL_CONTENT_EVENTS` | `false` | Log input/output messages as span events |

### Traffic Generator

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_ADDR` | `gateway:50051` | Gateway gRPC address |
| `INTERVAL_SECONDS` | `5` | Delay between requests |

## Project Structure

```
suse-ai-demo-apps/
├── proto/                        # Protobuf definitions + Makefile
│   ├── demo.proto
│   └── Makefile
├── gateway/                      # Go gRPC gateway
│   ├── main.go
│   ├── pb/                       # Generated Go proto stubs
│   └── Dockerfile
├── rag-service/                  # Python RAG service
│   ├── app/
│   │   ├── main.py
│   │   ├── otel_setup.py         # OTel bootstrap
│   │   ├── llm_client.py         # GenAI-instrumented LLM client
│   │   ├── embedding_client.py   # GenAI-instrumented embedding client
│   │   ├── grpc_server.py        # RAGService gRPC handler
│   │   ├── seed_data.py          # Vector DB bootstrap data
│   │   ├── generated/            # Generated Python proto stubs
│   │   └── vectordb/             # Vector DB repository pattern
│   │       ├── base.py           # Abstract VectorStore interface
│   │       ├── qdrant.py         # Qdrant implementation
│   │       └── factory.py        # Backend factory
│   └── Dockerfile
├── llm-service/                  # Python LLM service
│   ├── app/
│   │   ├── main.py
│   │   ├── otel_setup.py
│   │   ├── llm_client.py
│   │   ├── grpc_server.py
│   │   └── generated/
│   └── Dockerfile
├── traffic-gen/                  # Python traffic generator
│   ├── main.py
│   ├── generated/
│   └── Dockerfile
├── helm/
│   └── suse-ai-demo/            # Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
└── .github/
    └── workflows/
        └── build.yaml            # CI: build & push images to GHCR
```

## Extending the Vector DB Backend

The RAG service uses a repository pattern for vector DB access. To add a new backend (e.g., Milvus or OpenSearch):

1. Create a new file `rag-service/app/vectordb/milvus.py` implementing the `VectorStore` interface from `base.py`
2. Register it in `factory.py`:
   ```python
   from .milvus import MilvusVectorStore
   _REGISTRY["milvus"] = MilvusVectorStore
   ```
3. Set `VECTOR_DB_TYPE=milvus` and the appropriate `VECTOR_DB_URL`

## License

See [LICENSE](LICENSE).
