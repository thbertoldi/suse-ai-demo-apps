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
                              |
                              | gRPC
                              '-------> [Agent Service (Python)]
                                              |
                                              | HTTP (OpenAI-compat)
                                              +-------> [vLLM / Ollama]
                                              |
                                              | gRPC (search_docs tool)
                                              +-------> [RAG Service]
                                              | HTTP (predict tool)
                                              +-------> [KServe / Iris]
                                              | HTTP (list_models tool)
                                              '-------> [Kubeflow Model Registry]

  [Kubeflow Pipelines] ---> [Model Registry] ---> [KServe / Iris revision]
```

| Service | Language | Description | Port |
|---------|----------|-------------|------|
| **Gateway** | Go | gRPC entry point. Routes `Query` requests to the RAG service and `Chat` requests to the LLM service. | 50051 |
| **RAG Service** | Python | Embeds the user query, searches Milvus, Qdrant, or OpenSearch for relevant documents, builds a context-augmented prompt, and calls Ollama for an answer. | 50052 |
| **LLM Service** | Python | Forwards prompts to an OpenAI-compatible endpoint for general-purpose Q&A (no retrieval). | 50053 |
| **Traffic Generator** | Python | Sends RAG, Chat, and Agent scenarios to the Gateway, including deterministic Kubeflow lifecycle calls. | - |
| **Agent Service** | Python | LangGraph agent with document, calculation, time, KServe prediction, and Model Registry tools. Produces `invoke_agent` and `execute_tool` OTel spans. | 50054 |
| **Iris pipeline** | Python/KFP | Trains and evaluates an Iris model, registers its exact artifact, deploys it to KServe, and smoke-tests the ready revision. | - |

The default Helm profile uses Milvus and Ollama. The optional native-products profile adds Qdrant, OpenSearch, vLLM, LiteLLM, MLflow, mcpo and real CPU training. Products and the Collector are installed separately.

## OpenTelemetry Instrumentation

All services export **traces and metrics** via OTLP/gRPC to an OpenTelemetry Collector.

### Traces

- **GenAI spans** on every LLM and embedding call: `chat {model}`, `embed {model}`
  - Attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.response.id`
  - Optional content attributes: `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` (enable with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`)
- **Vector DB spans**: `db.system.name`, `db.operation.name`, `db.collection.name`, `server.address`, and explicit `server.port`; upserts also record `db.operation.batch.size`
- **gRPC spans**: auto-instrumented on all server and client connections
- **HTTP spans**: auto-instrumented on outbound LLM/embedding calls
- **Agent spans**: `invoke_agent {agent_name}` with `gen_ai.agent.name`, `gen_ai.agent.id`
- **Tool execution spans**: `execute_tool {tool_name}` with `gen_ai.tool.name`, `gen_ai.tool.type`, `gen_ai.tool.call.id`
- **Kubeflow lifecycle spans**: step, Model Registry, KServe deployment, and
  prediction spans carrying SUSE AI product-relation attributes
- **W3C TraceContext** propagation across all hops

### Metrics

- `gen_ai.client.token.usage` — histogram of token counts by model, operation, and token type
- `gen_ai.client.operation.duration` — histogram of LLM/embedding call duration
- Standard gRPC and HTTP server/client metrics from auto-instrumentation
- `suse.ai.agent.*` — tool rate/duration and agent iterations/outcomes
- `suse.ai.rag.*` — retrieval yield, no-hit count, context size, and request duration
- `suse.ai.demo.scenario.*` — generated scenario outcomes and end-to-end duration
- `suse.ai.kubeflow.*` — pipeline step duration/outcome, model accuracy, and
  deployment smoke-test outcome

LLM and embedding durations use seconds with subsecond histogram buckets. Each
HTTP provider attempt records one duration, including failed attempts with the
same `error.type` as its GenAI span. Token observations use only reported
nonnegative integer counts: missing usage stays absent, and a reported zero is
preserved. Agent tool arguments and results follow the same content-capture
opt-in. `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` takes precedence
over the legacy `ENABLE_OTEL_CONTENT_EVENTS` variable, which remains a fallback
when the standard variable is unset. Content capture is disabled by default.

Run the client regressions with the real OTel SDK:

```bash
python -m pip install -r tests/requirements.txt
python -m unittest discover -s tests -v
python -m pip install -r agent-service/requirements-dev.txt
PYTHONPATH=agent-service python -m pytest agent-service/tests -q
```

## Prerequisites

- **Go 1.25+** (for building the gateway)
- **Python 3.12+** (for building the Python services)
- **protoc** with `protoc-gen-go`, `protoc-gen-go-grpc`, and `grpcio-tools` (for regenerating proto stubs)
- **Docker** (for building container images)
- **Helm 3** (for deploying to Kubernetes)
- **Kubeflow Pipelines, Model Registry, and KServe** (for the optional real
  model-lifecycle demo)

External services (not included, must be running separately):
- Milvus (the chart default), Qdrant, or OpenSearch — vector database
- Ollama — generation and `/api/embed` embeddings in the default profile
- Optionally vLLM or LiteLLM — OpenAI-compatible generation endpoints
- An [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) — receives traces and metrics

Pull every configured Ollama model before starting traffic. The default RAG
configuration requires both `llama3.2` and `nomic-embed-text`:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Building

### Container Images

Each service has a Dockerfile. Build them from the repo root:

```bash
docker build -t suse-ai-demo-gateway ./gateway
docker build -t suse-ai-demo-rag-service ./rag-service
docker build -t suse-ai-demo-llm-service ./llm-service
docker build -t suse-ai-demo-agent-service ./agent-service
docker build -t suse-ai-demo-traffic-gen ./traffic-gen
docker build -t suse-ai-demo-iris-pipeline:0.2.0 ./demo/kubeflow
docker build -t suse-ai-demo-training:0.2.0 ./demo/training
```

Pull requests run regression checks and build/smoke-test all seven images. Main and version-tag builds publish to `ghcr.io` only after validation. Tag releases use one version across the chart, services, training and Iris pipeline; chart image defaults use `appVersion`, with per-image tag/digest overrides. See [the release procedure](docs/releasing.md).

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

This regenerates Go stubs in `gateway/pb/` and Python stubs in `{rag,llm,agent}-service/app/generated/` and `traffic-gen/generated/`.

## Running Locally

Each service is configured via environment variables. Set them to point at your local instances:

```bash
# Terminal 1: RAG service
cd rag-service
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3
export LLM_PROVIDER=ollama
export EMBEDDING_BASE_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
export VECTOR_DB_URL=http://localhost:6333
export VECTOR_DB_COLLECTION=demo_docs
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

# Terminal 5: Agent service
cd agent-service
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=llama3
export LLM_PROVIDER=vllm
export RAG_SERVICE_ADDR=localhost:50052
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python -m app.main
```

## Deploying with Helm

The default chart deploys gateway, RAG, LLM, agent and traffic generation. It
requires an existing vector database, generation/embedding models and an OTLP
Collector. Kubeflow traffic and RBAC are disabled by default. Use one demo release
per namespace: service names are intentionally stable.

Configure your endpoints in a copy of
[`values.yaml`](helm/suse-ai-demo/values.yaml), then install:

```bash
helm upgrade --install demo ./helm/suse-ai-demo \
  --namespace suse-ai-demo --create-namespace -f my-values.yaml --wait
kubectl -n suse-ai-demo logs deployment/traffic-gen --tail=20
```

For example, a Qdrant install needs **both** `ragService.vectorDb.type=qdrant`
and its `ragService.vectorDb.url`; changing the URL alone keeps the Milvus client.
Pull the configured Ollama generation and embedding models before installing.
OTel `service.namespace` defaults to the release namespace. Pod UID, name and
node come from the Downward API; configure cluster identity in the Collector.

For authenticated inference, create an existing Secret with an `api-key` key
and set `llmService.llm.existingSecret`, `ragService.llm.existingSecret`, or
`agentService.llm.existingSecret`. Each supports `apiKeyKey` to select another key.
Credentials are read from Secret references and never included in values examples.

### Native product traffic and training

[`examples/native-products.yaml`](examples/native-products.yaml) is an opt-in
profile. Edit its DNS names/models to match your installed products. It adds two
independent RAG services, a product traffic workload using the RAG image, and a
CPU PyTorch CronJob that trains for 200 epochs and uploads/downloads its checkpoint
through MLflow. It preserves the primary Milvus RAG service.

Create the referenced Secrets and CA ConfigMap in the demo namespace:

```bash
kubectl -n suse-ai-demo create secret generic litellm-api \
  --from-file=api-key=/path/to/litellm-key
kubectl -n suse-ai-demo create secret generic opensearch-auth \
  --from-file=username=/path/to/username --from-file=password=/path/to/password
kubectl -n suse-ai-demo create configmap opensearch-ca \
  --from-file=ca.crt=/path/to/opensearch-ca.crt
helm upgrade --install demo ./helm/suse-ai-demo -n suse-ai-demo \
  -f my-values.yaml -f examples/native-products.yaml --wait
```

The OpenSearch URL must match its certificate. TLS verification stays enabled.
`vectorDb.opensearch.indexReplicas: 0` is for a single-node demo; choose replicas
for a larger installation. Existing indexes retain their original engine/settings.
Each `additionalRagServices` entry inherits `ragService` and overrides only the
specified fields. Use separate collections/indexes when dimensions or models differ.

Select only installed products in `productTraffic.scenarios`; disable `training`
when MLflow is absent. MLflow must serve artifacts with `mlflow-artifacts:/` URIs
and allow the Service hostname **including its port**. Optional MLflow tokens use
`productTraffic.mlflow.existingSecret` and `training.mlflow.existingSecret` (key
`token`). Optional mcpo and vLLM keys use their `existingSecret`/`apiKeyKey` values.
For MLflow's own server telemetry, use the
[worker instrumentation recipe](demo/native-products/README.md).

```bash
kubectl -n suse-ai-demo logs deployment/product-traffic --tail=20
kubectl -n suse-ai-demo create job --from=cronjob/pytorch-training training-check
kubectl -n suse-ai-demo logs -f job/training-check
```

For a bounded acceptance run, use `productTraffic.maxRounds: 1` and
`productTraffic.intervalSeconds: 0` on a **fresh test release**. Helm renders a
Job instead of a Deployment; it exits nonzero if any selected scenario fails.
Delete that test release before switching modes, since Kubernetes cannot update
a Job's pod template in place. Continuous traffic uses `maxRounds: 0`.

### Optional Kubeflow lifecycle

After provisioning Kubeflow Pipelines, a profile namespace, Model Registry and
KServe, follow [the lifecycle runbook](demo/kubeflow/README.md) and enable
[`examples/kubeflow.yaml`](examples/kubeflow.yaml). This enables both the special
traffic and namespaced deployment RBAC. The KFP smoke step owns the stable
`suse-ai-sklearn-iris` prediction Service and updates its revision selector.

### Validation and compatibility

The compatibility baseline is SUSE AI Observability Extension **3.0.0** and
SUSE AI OpenTelemetry Collector **0.165.0**. The demo emits standard OTLP and can
also send to other Collectors; SUSE-specific views require the extension and
appropriate product profiles/endpoint associations. Installations need fresh
traffic before evaluating their charts or topology.

With Python 3.12, uv, Task, Go, Helm, ShellCheck and actionlint installed:

```bash
task validate
```

Tests use separate environments for the core services (OTel 1.44), Kubeflow
(OTel 1.40) and CPU training. Training tests execute real PyTorch optimization and
verify local MLflow HTTP and OTLP protocol exchanges. Product tests exercise all
six scenarios against local HTTP/gRPC fixtures. These do not replace acceptance
against your installed products. Release builds currently target **linux/amd64**.

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
| `LLM_API_KEY` | empty | Optional bearer credential |
| `EMBEDDING_BASE_URL` | `http://ollama:11434` | Ollama endpoint for `/api/embed` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider, independent of the generation proxy |
| `VECTOR_DB_TYPE` | `qdrant` | Vector DB backend |
| `VECTOR_DB_URL` | `http://qdrant:6333` | Vector DB address |
| `VECTOR_DB_COLLECTION` | `demo_docs` | Collection name (portable across Qdrant, Milvus, and OpenSearch) |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Capture input/output messages and system instructions as span attributes |

### LLM Service

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_LISTEN_ADDR` | `[::]:50053` | gRPC listen address |
| `LLM_BASE_URL` | `http://vllm:8000/v1` | LLM endpoint (OpenAI-compat) |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `vllm` | Provider name for OTel attributes |
| `LLM_API_KEY` | empty | Optional bearer credential |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Capture input/output messages and system instructions as span attributes |

### Agent Service

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_LISTEN_ADDR` | `[::]:50054` | gRPC listen address |
| `LLM_BASE_URL` | `http://vllm:8000/v1` | LLM endpoint (OpenAI-compat) |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `vllm` | Provider name for OTel attributes |
| `LLM_API_KEY` | empty | Optional bearer credential |
| `RAG_SERVICE_ADDR` | `rag-service:50052` | RAG service gRPC address |
| `AGENT_NAME` | `demo-agent` | Agent name for OTel attributes |
| `AGENT_MAX_ITERATIONS` | `5` | Max LLM calls per agent request |
| `DEMO_DETERMINISTIC_TOOLS` | `true` | Enable `[demo:list-models]`, `[demo:predict]`, and `[demo:lifecycle]` |
| `KSERVE_PREDICT_URL` | profile-local stable Service | KServe v1 prediction endpoint |
| `MODEL_REGISTRY_URL` | Kubeflow registry Service | Model Registry base URL |
| `MODEL_REGISTRY_BEARER_TOKEN` | `demo` | Demo Authorization header token; use a Secret outside demo environments |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Capture messages, system instructions, and tool arguments/results as span attributes |

### Traffic Generator

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_ADDR` | `gateway:50051` | Gateway gRPC address |
| `INTERVAL_SECONDS` | `5` | Delay between requests |
| `ENABLE_KUBEFLOW_SCENARIOS` | `false` | Generate KServe/Model Registry scenarios |

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
├── agent-service/                # Python agent service (LangGraph)
│   ├── app/
│   │   ├── main.py
│   │   ├── otel_setup.py
│   │   ├── otel_instrumentation.py  # GenAI span wrappers
│   │   ├── agent.py                 # LangGraph ReAct graph
│   │   ├── tools.py                 # Agent tools
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

## Open WebUI conversation traffic

`traffic-gen/webui.py` sends real chats through Open WebUI 0.6.41's
`/api/chat/completions` and `/api/chat/completed` lifecycle. It uses an ordinary
dedicated demo user and temporary `local:` chat IDs. It does not call Ollama
directly or use an administrator's token. Grant that user read access to the
chosen model (or a private model alias) and test one chat before enabling traffic.
Each attempt first lists the user's accessible models, matching the browser's
initialization of Ollama routing after an Open WebUI worker restart.

Create a private JSON file containing that user's `email` and `password`, then
mount it using an existing Kubernetes Secret. Keep this file and the Secret's
contents out of source control and terminal logs:

```bash
kubectl -n suse-ai-demo create secret generic webui-demo-user \
  --from-file=credentials.json=/private/path/credentials.json
```

Use these Helm values with the existing demo release, preserving its other values:

```yaml
webuiTraffic:
  enabled: true
  baseUrl: http://open-webui.suse-private-ai.svc.cluster.local:80
  model: llama3.2:latest # Or the private alias granted to the demo user.
  existingSecret: webui-demo-user
  intervalSeconds: 60
  maxRetries: 1
  streamEvery: 3
  maxTokens: 64
```

Build the traffic image from this checkout, or set `webuiTraffic.scriptConfigMap`
to a frozen ConfigMap containing this exact `webui.py` when using an older image.
The optional deployment has concurrency one, bounded retries (at most three),
a 120-second request timeout, no service-account token and a short token limit.
Every third request streams by default. Logs record each attempt, outcome,
duration, actual reported usage and temporary chat/message IDs; they omit
credentials and conversation content. These IDs are not exported as metric labels.

Pause/resume the optional traffic independently of the existing gateway demo:

```bash
kubectl -n suse-ai-demo scale deployment/webui-traffic --replicas=0
kubectl -n suse-ai-demo scale deployment/webui-traffic --replicas=1
```

For local validation, run `python3 -m unittest discover -s traffic-gen/tests -v`
and `helm lint helm/suse-ai-demo`. Actual GenAI instrumentation and endpoint
resolution are provided by the in-process Open WebUI integration in the
`suse-ai-observability-extension` repository; native WebUI HTTP spans alone do not
measure Ollama token usage or create a resolved client dependency.

## License

See [LICENSE](LICENSE).
